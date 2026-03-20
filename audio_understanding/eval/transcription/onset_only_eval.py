from __future__ import annotations

from typing import Callable, Optional

import numpy as np

ONSET_TOL: float = 0.05


def _prf(tp: int, n_ref: int, n_est: int) -> dict[str, float]:
    precision = tp / n_est if n_est > 0 else 0.0
    recall = tp / n_ref if n_ref > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _quantize_to_fps_grid(value: float, fps: float) -> float:
    return round(value * fps) / fps


def build_cropped_ref_onsets(data: dict, fps: float, use_drum_pitch: bool) -> list[dict]:
    r"""Build onset-only references for cropped clips.

    Keep only notes with onset in [start_time, clip_end].

    When use_drum_pitch=False, drums are represented as regular pitch tokens.
    When use_drum_pitch=True, drums are represented as drum_pitch tokens.
    """
    start_time = float(data["start_time"])
    duration = float(data["duration"])
    clip_end = start_time + duration

    notes = data.get("note", [])
    note_programs = data.get("note_program")
    note_is_drum = data.get("note_is_drum")
    if note_programs is not None:
        assert len(note_programs) == len(notes)
    if note_is_drum is not None:
        assert len(note_is_drum) == len(notes)

    ref_onsets: list[dict] = []

    for idx, note in enumerate(notes):
        n_start = float(note.start)
        if n_start < start_time:
            continue
        if n_start > clip_end:
            continue

        onset_rel = _quantize_to_fps_grid(n_start - start_time, fps)
        pitch_token_key = "pitch"
        if use_drum_pitch:
            is_drum = False
            if note_is_drum is not None:
                is_drum = bool(note_is_drum[idx])
            elif note_programs is not None:
                is_drum = int(note_programs[idx]) == 128
            if is_drum:
                pitch_token_key = "drum_pitch"

        ref_onsets.append(
            {
                "onset_time": onset_rel,
                "pitch": int(note.pitch),
                "pitch_token_key": pitch_token_key,
            }
        )

    return ref_onsets


def parse_onset_tokens(tokens: list[str], fps: float) -> list[dict]:
    r"""Parse onset-only token stream to onset dicts.

    Expected event: ["time_index=<int>", "pitch=<int>" or "drum_pitch=<int>"]

    Sequence-level mixing is allowed, e.g.:
    ["time_index=10", "pitch=60", "time_index=12", "drum_pitch=38", ...]
    and multiple events can share the same time_index.
    """
    est_onsets: list[dict] = []

    i = 0
    n = len(tokens)
    while i < n:
        time_tok = tokens[i]
        if not time_tok.startswith("time_index="):
            i += 1
            continue

        if i + 1 >= n:
            break

        pitch_tok = tokens[i + 1]
        if not (pitch_tok.startswith("pitch=") or pitch_tok.startswith("drum_pitch=")):
            i += 1
            continue

        value_str = time_tok.split("=", 1)[1]
        if not value_str.isdigit():
            i += 1
            continue

        pitch_key, pitch_value = pitch_tok.split("=", 1)
        if not pitch_value.isdigit():
            i += 1
            continue

        time_index = int(value_str)
        if time_index >= 0:
            est_onsets.append(
                {
                    "onset_time": float(time_index) / fps,
                    "pitch": int(pitch_value),
                    "pitch_token_key": pitch_key,
                }
            )

        i += 2

    return est_onsets


def onset_time_f1(
    ref_onsets: list[dict],
    est_onsets: list[dict],
    onset_tol: float = ONSET_TOL,
) -> dict[str, float]:
    r"""Onset F1 with time tolerance and exact pitch-type match.

    A match requires:
    - |onset_time_ref - onset_time_est| <= onset_tol
    - same pitch value
    - same pitch token type (pitch vs drum_pitch)
    """
    if not ref_onsets or not est_onsets:
        return _prf(0, len(ref_onsets), len(est_onsets))

    ref_sorted = sorted(ref_onsets, key=lambda n: (n["onset_time"], n["pitch"], n["pitch_token_key"]))
    est_sorted = sorted(est_onsets, key=lambda n: (n["onset_time"], n["pitch"], n["pitch_token_key"]))
    matched_est = [False] * len(est_sorted)
    tp = 0

    for ref in ref_sorted:
        best_j = -1
        best_dt = float("inf")

        for j, est in enumerate(est_sorted):
            if matched_est[j]:
                continue

            dt = abs(float(ref["onset_time"]) - float(est["onset_time"]))
            if dt > onset_tol:
                continue

            if int(ref["pitch"]) != int(est["pitch"]):
                continue
            if str(ref["pitch_token_key"]) != str(est["pitch_token_key"]):
                continue

            if dt < best_dt:
                best_dt = dt
                best_j = j

        if best_j >= 0:
            matched_est[best_j] = True
            tp += 1

    fp = sum(1 for m in matched_est if not m)
    fn = len(ref_sorted) - tp
    return _prf(tp, tp + fn, tp + fp)


def evaluate_cropped_onset_item(
    data: dict,
    output_tokens: list[str],
    fps: float,
) -> dict:
    target_tokens = data.get("token", [])
    use_drum_pitch = False
    if isinstance(target_tokens, list):
        use_drum_pitch = any(
            isinstance(tok, str) and tok.startswith("drum_pitch=")
            for tok in target_tokens
        )

    ref_onsets = build_cropped_ref_onsets(
        data=data,
        fps=fps,
        use_drum_pitch=use_drum_pitch,
    )
    est_onsets = parse_onset_tokens(tokens=output_tokens, fps=fps)

    return {
        "onset_time": onset_time_f1(ref_onsets, est_onsets),
        "_diag": {
            "raw_note_count": len(data.get("note", [])),
            "ref_onset_count": len(ref_onsets),
            "est_onset_count": len(est_onsets),
            "est_token_count": len(output_tokens),
            "start_time": float(data.get("start_time", 0.0)),
            "duration": float(data.get("duration", 0.0)),
        },
    }


def batch_evaluate_onset(
    dataset,
    inference_fn: Callable[[dict], list[str]],
    fps: float,
    max_samples: Optional[int] = None,
    epochs: int = 1,
    skip_empty_ref_for_averaging: bool = True,
    verbose: bool = True,
) -> dict:
    assert epochs >= 1

    n_total = len(dataset)
    if max_samples is not None:
        n_total = min(n_total, max_samples)

    dataset_name = type(dataset).__name__
    result_acc: dict[str, list] = {}
    empty_ref_count = 0
    empty_pred_correct = 0
    evaluated_items = 0

    total_steps = n_total * epochs
    iterator = range(total_steps)
    if verbose:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc=f"Evaluating onset-only {dataset_name}")

    for flat_i in iterator:
        i = flat_i % n_total
        data = dataset[i]
        evaluated_items += 1

        tokens = inference_fn(data)
        result = evaluate_cropped_onset_item(data=data, output_tokens=tokens, fps=fps)

        diag = result["_diag"]
        ref_onset_count = int(diag["ref_onset_count"])
        est_onset_count = int(diag["est_onset_count"])

        if ref_onset_count == 0:
            empty_ref_count += 1
            if est_onset_count == 0:
                empty_pred_correct += 1
            if skip_empty_ref_for_averaging:
                continue

        for k, v in result.items():
            result_acc.setdefault(k, []).append(v)

    summary: dict = {
        "n_samples": int(len(next(iter(result_acc.values()))) if result_acc else 0),
        "epochs": int(epochs),
        "evaluated_items": int(evaluated_items),
        "empty_ref_samples": int(empty_ref_count),
    }

    if "onset_time" in result_acc:
        metrics = result_acc["onset_time"]
        summary["onset_time"] = {
            "precision": float(np.mean([m["precision"] for m in metrics])),
            "recall": float(np.mean([m["recall"] for m in metrics])),
            "f1": float(np.mean([m["f1"] for m in metrics])),
        }

    if empty_ref_count > 0:
        summary["empty_audio_pred_acc"] = {
            "correct": int(empty_pred_correct),
            "total": int(empty_ref_count),
            "acc": float(empty_pred_correct / empty_ref_count),
        }

    return summary
