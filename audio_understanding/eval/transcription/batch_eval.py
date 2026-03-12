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

def _avg_metric(metrics_list: list[dict]) -> dict:
    r"""Average precision / recall / F1 across a list of metric dicts."""
    if not metrics_list:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": float(np.mean([m["precision"] for m in metrics_list])),
        "recall":    float(np.mean([m["recall"]    for m in metrics_list])),
        "f1":        float(np.mean([m["f1"]        for m in metrics_list])),
    }


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
    skip_empty_ref_for_averaging: bool = False,
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
    onset_list: list[dict] = []
    offset_list: list[dict] = []
    program_aware_list: list[dict] = []
    per_inst_list: list[dict] = []
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

        notes = data.get("note", [])
        is_empty_ref = len(notes) == 0
        if is_empty_ref:
            empty_ref_count += 1
            est_notes = parse_tokens_to_notes(
                tokens=tokens,
                fps=fps,
                include_program=include_program,
                start_time=0.0,
            )
            if len(est_notes) == 0:
                empty_pred_correct += 1

            if skip_empty_ref_for_averaging:
                continue

        result = _evaluate_cropped_item(
            data=data,
            output_tokens=tokens,
            fps=fps,
            include_program=include_program,
        )

        if verbose:
            d = result.get("_diag", {})
            onset_f1 = result["note_onset"]["f1"]
            print(
                f"[batch_eval] sample={i} | "
                f"raw_notes={d.get('raw_note_count')} -> "
                f"ref_notes(clipped)={d.get('ref_note_count')} | "
                f"est_notes={d.get('est_note_count')} | "
                f"est_tokens={d.get('est_token_count')} | "
                f"start={d.get('start_time'):.2f} dur={d.get('duration'):.2f} | "
                f"onset_f1={onset_f1:.4f}"
            )

        onset_list.append(result["note_onset"])
        offset_list.append(result["note_offset"])

        if "program_aware" in result:
            program_aware_list.append(result["program_aware"])
        if "per_instrument" in result:
            per_inst_list.append(result["per_instrument"])

    summary: dict = {
        "n_samples": len(onset_list),
        "epochs": int(epochs),
        "evaluated_items": int(evaluated_items),
        "empty_ref_samples": int(empty_ref_count),
        "note_onset": _avg_metric(onset_list),
        "note_offset": _avg_metric(offset_list),
    }

    if empty_ref_count > 0:
        summary["empty_audio_pred_acc"] = {
            "correct": int(empty_pred_correct),
            "total": int(empty_ref_count),
            "acc": float(empty_pred_correct / empty_ref_count),
        }

    if program_aware_list:
        summary["program_aware"] = _avg_metric(program_aware_list)

    if per_inst_list:
        summary["instrument_summary"] = instrument_summary(per_inst_list)

    return summary
