r"""Batch inference and evaluation entry point for music transcription.

Usage example::

    from torch.utils.data import DataLoader
    from audio_understanding.datasets.maestro import MAESTRO
    from audio_understanding.eval.transcription.batch_eval import batch_evaluate

    dataset = MAESTRO(root="/data/maestro-v3.0.0", split="test", crop=None)

    def my_inference_fn(data):
        # ... run model on data["audio"] ...
        return tokens  # list[str]

    results = batch_evaluate(
        dataset=dataset,
        inference_fn=my_inference_fn,
        fps=100,
    )
    print(results["note_onset"])   # {"precision": ..., "recall": ..., "f1": ...}
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from audio_understanding.eval.transcription.metrics import (
    drum_f1,
    instrument_summary,
    note_onset_f1,
    note_with_offset_f1,
    parse_tokens_to_notes,
    per_instrument_metrics,
    program_aware_f1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_scalar(value) -> bool:
    return isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_))


def _mean_scalar(values: list) -> float:
    assert len(values) > 0
    return float(np.mean([float(v) for v in values]))


def _representative_value(values: list):
    """Return a stable representative value for non-numeric leaves.

    If all values are identical, return that value.
    Otherwise, return the majority value (first occurrence wins ties).
    """
    assert len(values) > 0
    first = values[0]
    if all(v == first for v in values):
        return first

    counts: dict[object, int] = {}
    first_idx: dict[object, int] = {}
    for i, v in enumerate(values):
        counts[v] = counts.get(v, 0) + 1
        if v not in first_idx:
            first_idx[v] = i

    best = max(counts.keys(), key=lambda v: (counts[v], -first_idx[v]))
    return best


def _recursive_avg(values: list):
    """Recursively average a homogeneous list of scalars / dicts."""
    assert len(values) > 0

    first = values[0]
    if _is_scalar(first):
        return _mean_scalar(values)

    if isinstance(first, dict):
        out: dict = {}
        all_keys = set()
        for v in values:
            assert isinstance(v, dict)
            all_keys.update(v.keys())
        for k in all_keys:
            child_values = [v[k] for v in values if k in v]
            if not child_values:
                continue
            out[k] = _recursive_avg(child_values)
        return out

    # For non-numeric leaves (e.g. inst_class), keep a stable representative.
    return _representative_value(values)


def _is_single_layer_summary_value(value) -> bool:
    """Keep only scalar or one-level dict values in summary."""
    if _is_scalar(value):
        return True
    if isinstance(value, dict):
        return all(_is_scalar(v) for v in value.values())
    return False


def _merge_result_accumulator(acc: dict[str, list], result: dict) -> None:
    for k, v in result.items():
        if k not in acc:
            acc[k] = []
        acc[k].append(v)


def _finalize_result_accumulator(acc: dict[str, list]) -> dict:
    summary: dict = {}
    for k, values in acc.items():
        assert len(values) > 0
        summary[k] = _recursive_avg(values)
    return summary


def _quantize_to_fps_grid(value: float, fps: float) -> float:
    return round(value * fps) / fps


def _build_cropped_ref_notes(data: dict, fps: float, exclude_boundary: bool = False) -> list[dict]:
    """Build reference notes from data, excluding boundary-spanning notes.

    Notes that span the entire clip (MIDI2Tokens Case 3:
    note.start < start_time AND note.end > clip_end) are always excluded
    because the tokenizer produces no tokens for them.

    When exclude_boundary=True, also exclude Case 2 (left boundary,
    note.start < start_time) and Case 5 (right boundary,
    note.end > clip_end).  This keeps only Case 4 notes (fully within
    the clip), useful for debug sanity checks where roundtrip fidelity
    must be 100%.
    """
    start_time = float(data["start_time"])
    duration = float(data["duration"])
    clip_end = start_time + duration

    notes = data.get("note", [])
    note_programs = data.get("note_program", [])
    note_is_drum = data.get("note_is_drum", [])
    note_inst_class = data.get("note_inst_class", [])
    has_inst_meta = len(note_programs) == len(notes) and len(notes) > 0

    ref_notes: list[dict] = []
    for idx, note in enumerate(notes):
        n_start = float(note.start)
        n_end = float(note.end)

        if exclude_boundary:
            # Case 4 only: note fully within [start_time, clip_end]
            if n_start < start_time or n_end > clip_end:
                continue
        else:
            # Skip Case 3 only (spans entire clip, no token output)
            if n_start < start_time and n_end > clip_end:
                continue

        onset_abs = max(start_time, n_start)
        offset_abs = min(clip_end, n_end)
        if offset_abs < onset_abs:
            continue

        # Case 2 (left boundary): only offset token is emitted.
        # If offset time_index rounds to 0, the parser skips it (zero-duration
        # at clip start), so ref must also skip it for consistency.
        is_left_boundary = n_start < start_time
        if is_left_boundary:
            offset_idx = round((offset_abs - start_time) * fps)
            if offset_idx <= 0: #! 这一条的确能增加offset的准确率
                continue

        onset_rel = _quantize_to_fps_grid(onset_abs - start_time, fps)
        offset_rel = _quantize_to_fps_grid(offset_abs - start_time, fps)
        # Match MIDI2Tokens: ensure offset > onset (minimum 1 frame apart).
        # Zero-duration notes (common for drums) get pushed to onset + 1 frame.
        if offset_rel <= onset_rel:
            offset_rel = onset_rel + 1.0 / fps

        note_dict: dict = {
            "onset_time": onset_rel,
            "offset_time": offset_rel,
            "pitch": int(note.pitch),
            "velocity": int(note.velocity),
        }
        if has_inst_meta:
            note_dict["program"] = int(note_programs[idx])
            note_dict["is_drum"] = bool(note_is_drum[idx])
            note_dict["inst_class"] = note_inst_class[idx] if note_inst_class else "unknown"
        ref_notes.append(note_dict)
    return ref_notes


def _evaluate_cropped_item(
    data: dict, output_tokens: list[str], fps: float, include_program: bool,
    exclude_boundary: bool = False,
) -> dict:
    """Evaluate a single sample with boundary-aware ref construction.

    When exclude_boundary=True, only notes fully within the clip are
    evaluated (Case 4 in MIDI2Tokens).  Useful for debug sanity checks.
    """
    raw_notes = data.get("note", [])
    ref_notes = _build_cropped_ref_notes(data, fps, exclude_boundary=exclude_boundary)
    est_notes = parse_tokens_to_notes(
        tokens=output_tokens,
        fps=fps,
        include_program=include_program,
        start_time=0.0,
        clip_duration=float(data["duration"]),
        exclude_boundary=exclude_boundary,
    )
    result: dict = {
        "note_onset": note_onset_f1(ref_notes, est_notes),
        "note_offset": note_with_offset_f1(ref_notes, est_notes),
        "_diag": {
            "raw_note_count": len(raw_notes),
            "ref_note_count": len(ref_notes),
            "est_note_count": len(est_notes),
            "est_token_count": len(output_tokens),
            "start_time": float(data.get("start_time", 0)),
            "duration": float(data.get("duration", 0)),
        },
    }

    # Detect X2one scenario: instrument metadata exists but tokens lack program
    note_programs = data.get("note_program", [])
    note_is_drum = data.get("note_is_drum", [])
    notes = data.get("note", [])
    has_inst_meta = len(note_programs) == len(notes) and len(notes) > 0
    if has_inst_meta:
        result["drum"] = drum_f1(ref_notes, est_notes)

    single_program = None
    single_is_drum = False
    if has_inst_meta and not include_program:
        unique_non_drum_progs = set(
            int(p) for p, d in zip(note_programs, note_is_drum) if not d
        )
        all_drum = all(bool(d) for d in note_is_drum)
        if len(unique_non_drum_progs) == 1 or (
            len(unique_non_drum_progs) == 0 and all_drum
        ):
            single_program = int(note_programs[0])
            single_is_drum = bool(note_is_drum[0])

    if include_program or single_program is not None:
        if single_program is not None and not include_program:
            est_notes_prog = [
                {**n, "program": single_program, "is_drum": single_is_drum}
                for n in est_notes
            ]
        else:
            est_notes_prog = est_notes
        result["program_aware"] = program_aware_f1(ref_notes, est_notes_prog)
        result["per_instrument"] = per_instrument_metrics(ref_notes, est_notes_prog)

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def batch_evaluate(
    dataset,
    inference_fn: Callable[[dict], list[str]],
    fps: float,
    include_program: bool = False,
    max_samples: Optional[int] = None,
    epochs: int = 1,
    skip_empty_ref_for_averaging: bool = True,
    verbose: bool = True,
) -> dict:
    r"""Run inference and evaluation over every sample in *dataset*.

    Uses ``_evaluate_cropped_item`` for boundary-aware per-sample scoring:
    notes that span the entire clip boundary are excluded from references
    and times are quantized to the fps grid for consistent comparison.

    Args:
        dataset: a transcription dataset whose ``__getitem__`` returns dicts
            with ``start_time``, ``duration``, ``note``, etc.
        inference_fn: callable that accepts a single ``__getitem__`` output
            dict and returns a flat list of MIDI token strings (model output).
        fps: frames-per-second value used when encoding tokens (must match the
            value used during training / inference).
        include_program: whether the model output tokens include
            ``program=X`` fields.
        max_samples: optional cap on the number of samples evaluated.
        verbose: print a progress bar with ``tqdm`` when available.
    """
    assert epochs >= 1

    n_total = len(dataset)
    if max_samples is not None:
        n_total = min(n_total, max_samples)

    dataset_name = type(dataset).__name__

    # Per-sample accumulators
    result_acc: dict[str, list] = {}
    empty_ref_count = 0
    empty_pred_correct = 0
    evaluated_items = 0

    total_steps = n_total * epochs
    iterator = range(total_steps)
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"Evaluating {dataset_name}")
        except ImportError:
            pass

    for flat_i in iterator:
        i = flat_i % n_total
        data = dataset[i] #* 能够保证data就是dataset的透传
        evaluated_items += 1

        try:
            tokens = inference_fn(data)
        except Exception as exc:
            if verbose:
                print(f"[batch_evaluate] inference failed for sample {i}: {exc}")
            continue

        result = _evaluate_cropped_item(
            data=data,
            output_tokens=tokens,
            fps=fps,
            include_program=include_program,
        )

        diag = result.get("_diag", {})
        ref_note_count = int(diag.get("ref_note_count", 0))
        est_note_count = int(diag.get("est_note_count", 0))

        # Empty-reference detection must follow cropped evaluation semantics.
        # Using raw data["note"] can disagree with ref_notes(clipped) and
        # silently pollute averaged F1 denominator.
        if ref_note_count == 0:
            empty_ref_count += 1
            if est_note_count == 0:
                empty_pred_correct += 1
            if skip_empty_ref_for_averaging:
                continue

        if verbose:
            onset_f1 = result["note_onset"]["f1"]
            print(
                f"[batch_eval] sample={i} | "
                f"raw_notes={diag.get('raw_note_count')} -> "
                f"ref_notes(clipped)={diag.get('ref_note_count')} | "
                f"est_notes={diag.get('est_note_count')} | "
                f"est_tokens={diag.get('est_token_count')} | "
                f"start={diag.get('start_time'):.2f} dur={diag.get('duration'):.2f} | "
                f"onset_f1={onset_f1:.4f}"
            )

        _merge_result_accumulator(result_acc, result)

    summary: dict = {
        "n_samples": int(len(next(iter(result_acc.values()))) if result_acc else 0),
        "epochs": int(epochs),
        "evaluated_items": int(evaluated_items),
        "empty_ref_samples": int(empty_ref_count),
    }

    agg_result = _finalize_result_accumulator(result_acc)
    for k, v in agg_result.items():
        if k.startswith("_"):
            continue
        if _is_single_layer_summary_value(v):
            summary[k] = v

    if empty_ref_count > 0:
        summary["empty_audio_pred_acc"] = {
            "correct": int(empty_pred_correct),
            "total": int(empty_ref_count),
            "acc": float(empty_pred_correct / empty_ref_count),
        }

    if "per_instrument" in result_acc:
        summary["instrument_summary"] = instrument_summary(result_acc["per_instrument"])

    return summary
