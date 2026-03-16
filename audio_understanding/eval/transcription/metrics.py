r"""Music transcription evaluation metrics.

Supported metrics
-----------------
- Note onset F1 (onset + pitch match within a time tolerance)
- Note-with-offset F1 (onset + pitch + offset match)
- Program-aware F1 (onset + pitch + MIDI program match)
- Per-instrument metrics (note onset F1 grouped by program)
- Instrument summary (aggregated statistics across a full evaluation set)

Token format (as produced by MIDI2Tokens)
------------------------------------------
Events are represented as a flat list of strings.  Each note event occupies
consecutive tokens in the following order::

    note_onset:  name=note_onset  time_index=X  pitch=X  velocity=X  [program=X]
    note_offset: name=note_offset time_index=X  pitch=X              [program=X]

Time in seconds = time_index / fps.

Matching algorithm
------------------
Greedy best-first matching sorted by onset time, consistent with the
standard music transcription evaluation protocol (mir_eval style).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ONSET_TOL: float = 0.05       # 50 ms onset tolerance
OFFSET_RATIO: float = 0.2     # offset tolerance = max(OFFSET_MIN_TOL, ratio × reference_duration)
OFFSET_MIN_TOL: float = 0.05  # 50 ms minimum offset tolerance

#: Sentinel program value used to represent drums (channel 9 / is_drum=True).
#: Keeps drums separate from program 0 (Grand Piano) in per-instrument grouping.
DRUM_PROGRAM: int = 128


# ---------------------------------------------------------------------------
# Token parsing
# ---------------------------------------------------------------------------

# 这个parsing的补全对于segment/ continous 都可以
def parse_tokens_to_notes(
    tokens: list[str],
    fps: float,
    include_program: bool = False,
    start_time: float = 0.0,
    clip_duration: Optional[float] = None,
    exclude_boundary: bool = False,
    ignore_program: bool = False
) -> list[dict]:
    r"""Parse a MIDI token sequence into a list of note event dicts.

    Token order per event (matching MIDI2Tokens construction order):

        * note_onset:  ``name=note_onset``, ``time_index=X``, ``pitch=X`` or ``drum_pitch=X``,
      ``velocity=X`` [, ``program=X``]
        * note_offset: ``name=note_offset``, ``time_index=X``, ``pitch=X`` or ``drum_pitch=X``
      [, ``program=X``]

    Args:
        tokens: flat list of token strings as produced by the model.
        fps: frames per second used to encode time (time_index / fps = seconds
            from clip start).
        include_program: whether ``program=X`` tokens appear in each event.
        ignore_program: when True and include_program=True, ignore predicted
            per-instrument program IDs by normalizing all non-drum notes to
            program 0 and all drum notes to :data:`DRUM_PROGRAM`.
        start_time: absolute clip start time added to all output times.
        clip_duration: optional clip duration in seconds. When provided,
            onset-only notes are closed at ``start_time + clip_duration``.

    Returns:
        List of note dicts.  Each dict contains:

        * ``onset_time``  – float, absolute seconds
                * ``offset_time`` – float, absolute seconds (defaults to
                    ``start_time + clip_duration`` when provided, otherwise
                    ``onset_time + 0.1`` when no matching offset token is found)
        * ``pitch``       – int 0-127
        * ``velocity``    – int 0-127
        * ``program``     – int 0-127 (only when *include_program* is True)
    """
    # open_notes maps key -> list of onset dicts awaiting a matching offset.
    # key is pitch when include_program=False, otherwise (pitch, program).
    open_notes: dict[object, list[dict]] = {}
    finished: list[dict] = []
    n = len(tokens)
    i = 0

    while i < n:
        tok = tokens[i]
        if "=" not in tok:
            i += 1
            continue

        key, value = tok.split("=", 1)

        if key == "name" and value == "note_onset":
            event_len = 4 + (1 if include_program else 0)
            if i + event_len > n:
                break
            try:
                time_index = int(tokens[i + 1].split("=")[1])
                pitch_key, pitch_value = tokens[i + 2].split("=", 1)
                assert pitch_key in ["pitch", "drum_pitch"]
                pitch = int(pitch_value)
                velocity = int(tokens[i + 3].split("=")[1])
            except (ValueError, IndexError):
                i += 1
                continue
            note: dict = {
                "onset_time": start_time + time_index / fps,
                "offset_time": None,
                "pitch": pitch,
                "velocity": velocity,
            }
            if include_program:
                try:
                    note_program = int(tokens[i + 4].split("=")[1])
                except (ValueError, IndexError):
                    note_program = 0

                if ignore_program:
                    if pitch_key == "drum_pitch":
                        note["program"] = DRUM_PROGRAM
                        note["is_drum"] = True
                    else:
                        note["program"] = 0
                else:
                    note["program"] = note_program
                    if pitch_key == "drum_pitch":
                        note["is_drum"] = True
            elif pitch_key == "drum_pitch":
                note["program"] = DRUM_PROGRAM
                note["is_drum"] = True
            open_key: object = pitch
            if include_program:
                open_key = (pitch, int(note["program"]))
            open_notes.setdefault(open_key, []).append(note)
            i += event_len
            continue

        elif key == "name" and value == "note_offset":
            event_len = 3 + (1 if include_program else 0)
            if i + event_len > n:
                break
            try:
                time_index = int(tokens[i + 1].split("=")[1])
                pitch_key, pitch_value = tokens[i + 2].split("=", 1)
                assert pitch_key in ["pitch", "drum_pitch"]
                pitch = int(pitch_value)
            except (ValueError, IndexError):
                i += 1
                continue
            close_key: object = pitch
            if include_program:
                try:
                    close_program = int(tokens[i + 3].split("=")[1])
                except (ValueError, IndexError):
                    close_program = 0
                if ignore_program:
                    close_program = DRUM_PROGRAM if pitch_key == "drum_pitch" else 0
                close_key = (pitch, close_program)
            elif pitch_key == "drum_pitch":
                close_key = (pitch, DRUM_PROGRAM)

            if open_notes.get(close_key):
                # Match offset to the earliest unmatched same-pitch onset (queue/FIFO).
                # FIFO is correct for sequential overlapping notes (common in
                # piano with sustain pedal): onset_A, onset_B, offset_A, offset_B.
                note = open_notes[close_key].pop(0)
                off_time = start_time + time_index / fps
                #! clip only Ensure strictly positive duration (mirrors MIDI2Tokens bump)
                if off_time <= note["onset_time"]:
                    off_time = note["onset_time"] + 1.0 / fps
                note["offset_time"] = off_time
                finished.append(note)
            else:
                # Handle left-clipped notes represented by offset-only events.
                # MIDI2Tokens can emit note_offset without note_onset when
                # note.start < clip_start <= note.end.
                if not exclude_boundary:
                    off_time = start_time + time_index / fps
                    #! clip only Skip if offset lands on clip start (zero-duration)
                    if off_time <= start_time:
                        i += event_len
                        continue
                    note = {
                        "onset_time": start_time,
                        "offset_time": off_time,
                        "pitch": pitch,
                        "velocity": 0,
                    }
                    if include_program:
                        note["program"] = int(close_program)
                        if pitch_key == "drum_pitch":
                            note["is_drum"] = True
                    elif pitch_key == "drum_pitch":
                        note["program"] = DRUM_PROGRAM
                        note["is_drum"] = True
                    finished.append(note)
            i += event_len
            continue

        i += 1

    # Close any notes that never received a matching offset token
    if not exclude_boundary:
        for pitch_notes in open_notes.values():
            for note in pitch_notes:
                if clip_duration is not None:
                    # Quantize to fps grid for consistency with token-derived times
                    note["offset_time"] = start_time + round(float(clip_duration) * fps) / fps
                else:
                    note["offset_time"] = note["onset_time"] + 0.1  # 100 ms default
                finished.append(note)

    return finished


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prf(tp: int, n_ref: int, n_est: int) -> dict:
    r"""Return precision / recall / F1 from raw counts."""
    precision = tp / n_est if n_est > 0 else 0.0
    recall = tp / n_ref if n_ref > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _effective_program(note: dict) -> int: #* when program is missing, default to 0, therefore can compute program-aware metrics even when program info is not provided
    r"""Return the program number to use for matching.

    Drums (``is_drum=True``) are mapped to :data:`DRUM_PROGRAM` regardless of
    their stored program number, so that all drum tracks are treated as a
    single group.
    """
    if note.get("is_drum"):
        return DRUM_PROGRAM
    return note.get("program", 0)


def _match_greedy(
    ref_notes: list[dict],
    est_notes: list[dict],
    onset_tol: float,
    check_offset: bool,
    check_program: bool,
    offset_ratio: float = OFFSET_RATIO,
    offset_min_tol: float = OFFSET_MIN_TOL,
    exclude_drums: bool = True,
    drum_only: bool = False
) -> tuple[int, int, int]:
    r"""Greedy note matching sorted by onset time.

    For each reference note (processed in onset order) the closest unmatched
    estimate note that satisfies all enabled constraints is selected.

    Returns:
        ``(true_positives, false_positives, false_negatives)``
    """
    ref_sorted = sorted(ref_notes, key=lambda n: (n["onset_time"], n["pitch"]))
    est_sorted = sorted(est_notes, key=lambda n: (n["onset_time"], n["pitch"]))

    if drum_only:
        ref_sorted = [
            n for n in ref_sorted
            if n.get("is_drum") or n.get("program") == DRUM_PROGRAM
        ]
        est_sorted = [
            n for n in est_sorted
            if n.get("is_drum") or n.get("program") == DRUM_PROGRAM
        ]
    elif exclude_drums:
        ref_sorted = [
            n for n in ref_sorted
            if not n.get("is_drum") and n.get("program") != DRUM_PROGRAM
        ]
        est_sorted = [
            n for n in est_sorted
            if not n.get("is_drum") and n.get("program") != DRUM_PROGRAM
        ]

    if not ref_sorted or not est_sorted:
        return 0, len(est_sorted), len(ref_sorted)

    matched_est = [False] * len(est_sorted)
    tp = 0

    for ref in ref_sorted:
        best_j = -1
        best_dt = float("inf")

        # for unmatched est in onset order, find the closest that meets the criteria
        for j, est in enumerate(est_sorted):
            if matched_est[j]:
                continue

            # Onset tolerance
            dt = abs(ref["onset_time"] - est["onset_time"])
            if dt > onset_tol:
                continue

            # Pitch must match exactly
            if ref["pitch"] != est["pitch"]:
                continue

            # Program check (uses drum sentinel)
            if check_program:
                if _effective_program(ref) != _effective_program(est):
                    continue

            # Offset tolerance follows the transcription standard:
            # |offset_diff| <= max(0.2 * reference_duration, 50 ms)
            if check_offset:
                ref_dur = max(0.0, ref["offset_time"] - ref["onset_time"])
                off_tol = max(offset_min_tol, offset_ratio * ref_dur)
                if abs(ref["offset_time"] - est["offset_time"]) > off_tol:
                    continue

            if dt < best_dt:
                best_dt = dt
                best_j = j

        if best_j >= 0:
            matched_est[best_j] = True
            tp += 1

    fp = sum(1 for m in matched_est if not m)
    fn = len(ref_sorted) - tp
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------

def note_onset_f1(
    ref_notes: list[dict],
    est_notes: list[dict],
    onset_tol: float = ONSET_TOL,
) -> dict:
    r"""Note onset precision / recall / F1 (offset ignored).

    A match requires identical pitch and onset within *onset_tol* seconds.

    Args:
        ref_notes: reference note dicts with at least ``onset_time`` and
            ``pitch``.
        est_notes: estimated note dicts with at least ``onset_time`` and
            ``pitch``.
        onset_tol: maximum onset time difference in seconds.

    Returns:
        ``{"precision": float, "recall": float, "f1": float}``
    """
    tp, fp, fn = _match_greedy(
        ref_notes, est_notes,
        onset_tol=onset_tol,
        check_offset=False,
        check_program=False,
        exclude_drums=True,
    )
    return _prf(tp, tp + fn, tp + fp)

def drum_f1(
    ref_notes: list[dict],
    est_notes: list[dict],
    onset_tol: float = ONSET_TOL,
) -> dict:
    r"""Drum note onset precision / recall / F1 (offset ignored).

    A match requires identical pitch and onset within *onset_tol* seconds, and
    both notes must be drums (``is_drum=True`` or ``program=DRUM_PROGRAM``).

    Args:
        ref_notes: reference note dicts with at least ``onset_time``, ``pitch``,
            and either ``program`` (int) or ``is_drum`` (bool).
        est_notes: estimated note dicts with the same fields.
        onset_tol: maximum onset time difference in seconds.

    Returns:
        ``{"precision": float, "recall": float, "f1": float}``
    """
    tp, fp, fn = _match_greedy(
        ref_notes, est_notes,
        onset_tol=onset_tol,
        check_offset=False,
        check_program=False,
        exclude_drums=False,
        drum_only=True,
    )
    return _prf(tp, tp + fn, tp + fp)

def note_with_offset_f1(
    ref_notes: list[dict],
    est_notes: list[dict],
    onset_tol: float = ONSET_TOL,
    offset_ratio: float = OFFSET_RATIO,
    offset_min_tol: float = OFFSET_MIN_TOL,
) -> dict:
    r"""Note-with-offset precision / recall / F1.

    A match requires identical pitch, onset within *onset_tol*, and offset
    within ``max(offset_min_tol, offset_ratio × reference_duration)``.

    Args:
        ref_notes: reference note dicts with ``onset_time``, ``offset_time``,
            and ``pitch``.
        est_notes: estimated note dicts with ``onset_time``, ``offset_time``,
            and ``pitch``.
        onset_tol: maximum onset time difference in seconds.
        offset_ratio: fraction of note duration used as offset tolerance.
        offset_min_tol: minimum offset tolerance in seconds.

    Returns:
        ``{"precision": float, "recall": float, "f1": float}``
    """
    tp, fp, fn = _match_greedy(
        ref_notes, est_notes,
        onset_tol=onset_tol,
        check_offset=True,
        check_program=False,
        offset_ratio=offset_ratio,
        offset_min_tol=offset_min_tol,
    )
    return _prf(tp, tp + fn, tp + fp)


def program_aware_f1(
    ref_notes: list[dict],
    est_notes: list[dict],
    onset_tol: float = ONSET_TOL,
) -> dict:
    r"""Program-aware note onset precision / recall / F1.

    Like :func:`note_onset_f1` but a match also requires the same MIDI
    program.  Drums (``is_drum=True``) are mapped to the :data:`DRUM_PROGRAM`
    sentinel so that all drum tracks form a single group irrespective of their
    stored program number.

    Args:
        ref_notes: reference note dicts with ``onset_time``, ``pitch``, and
            either ``program`` (int) or ``is_drum`` (bool).
        est_notes: estimated note dicts with the same fields.
        onset_tol: maximum onset time difference in seconds.

    Returns:
        ``{"precision": float, "recall": float, "f1": float}``
    """
    tp, fp, fn = _match_greedy(
        ref_notes, est_notes,
        onset_tol=onset_tol,
        check_offset=False,
        check_program=True,
    )
    return _prf(tp, tp + fn, tp + fp)


def per_instrument_metrics(
    ref_notes: list[dict],
    est_notes: list[dict],
    onset_tol: float = ONSET_TOL,
) -> dict:
    r"""Per-instrument note onset F1 grouped by MIDI program.
    This only works when the model predicts program or if it only predicts a single program

    Notes are grouped by their effective program (see :func:`_effective_program`).
    Drums (``is_drum=True``) are aggregated under the :data:`DRUM_PROGRAM`
    sentinel key.

    Args:
        ref_notes: reference note dicts with ``onset_time``, ``pitch``,
            ``program`` / ``is_drum``, and optionally ``inst_class``.
        est_notes: estimated note dicts with the same fields.
        onset_tol: maximum onset time difference in seconds.

    Returns:
        Dict keyed by effective program (int).  Each value is::

            {
                "precision":  float,
                "recall":     float,
                "f1":         float,
                "ref_count":  int,   # reference note count for this program
                "est_count":  int,   # estimated note count for this program
                "inst_class": str,   # label from reference notes if available
            }
    """
    ref_by_prog: dict[int, list[dict]] = {}
    for note in ref_notes:
        prog = _effective_program(note)
        ref_by_prog.setdefault(prog, []).append(note)

    est_by_prog: dict[int, list[dict]] = {}
    for note in est_notes:
        prog = _effective_program(note)
        est_by_prog.setdefault(prog, []).append(note)

    all_programs = set(ref_by_prog) | set(est_by_prog)
    results: dict[int, dict] = {}

    for prog in all_programs:
        r = ref_by_prog.get(prog, [])
        e = est_by_prog.get(prog, [])
        tp, fp, fn = _match_greedy(
            r, e,
            onset_tol=onset_tol,
            check_offset=False,
            check_program=False,
        )
        metrics = _prf(tp, tp + fn, tp + fp)
        metrics["ref_count"] = len(r)
        metrics["est_count"] = len(e)
        # Pick inst_class label from reference notes when available
        if r:
            metrics["inst_class"] = r[0].get("inst_class", "unknown")
        elif e:
            metrics["inst_class"] = e[0].get("inst_class", "unknown")
        else:
            metrics["inst_class"] = "unknown"
        results[prog] = metrics

    return results


def instrument_summary(
    all_per_inst: list[dict],
    top_n: int = 10,
) -> dict:
    r"""Aggregate per-instrument metrics across all evaluation samples.

    Summarizes instruments that appear anywhere in the full evaluation set.

    Args:
        all_per_inst: list of :func:`per_instrument_metrics` outputs, one per
            evaluated sample.
        top_n: number of top instruments to report individually (ranked by
            total reference note count).

    Returns:
        Dict with the following keys:

        * ``"counts"``      – ``{program: total_ref_note_count}``
        * ``"averages"``    – ``{program: average_f1_across_samples}``
        * ``"inst_classes"``– ``{program: instrument_class_label}``
        * ``"top_10"``      – list of top-*top_n* instrument dicts, each
          containing ``program``, ``avg_f1``, ``total_count``, ``inst_class``
        * ``"others"``      – ``{"avg_f1": float, "total_count": int,
          "num_programs": int}`` aggregated over remaining instruments
    """
    counts: dict[int, int] = {}
    f1_sums: dict[int, float] = {}
    f1_samples: dict[int, int] = {}
    inst_classes: dict[int, str] = {}

    for sample_result in all_per_inst:
        for prog, metrics in sample_result.items():
            counts[prog] = counts.get(prog, 0) + metrics["ref_count"]
            f1_sums[prog] = f1_sums.get(prog, 0.0) + metrics["f1"]
            f1_samples[prog] = f1_samples.get(prog, 0) + 1
            if prog not in inst_classes and metrics.get("inst_class", "unknown") != "unknown":
                inst_classes[prog] = metrics["inst_class"]

    averages: dict[int, float] = {
        prog: f1_sums[prog] / f1_samples[prog]
        for prog in f1_sums
    }

    # Sort programs by total reference note count (descending)
    sorted_progs = sorted(counts, key=lambda p: counts[p], reverse=True)
    top_progs = sorted_progs[:top_n]
    other_progs = sorted_progs[top_n:]

    top_list = [
        {
            "program": prog,
            "avg_f1": averages.get(prog, 0.0),
            "total_count": counts.get(prog, 0),
            "inst_class": inst_classes.get(prog, "unknown"),
        }
        for prog in top_progs
    ]

    others_count = sum(counts.get(p, 0) for p in other_progs)
    others_avg_f1 = (
        sum(averages.get(p, 0.0) * counts.get(p, 0) for p in other_progs)
        / others_count
        if others_count > 0
        else 0.0
    )

    return {
        "counts": counts,
        "averages": averages,
        "inst_classes": inst_classes,
        "top_10": top_list,
        "others": {
            "avg_f1": others_avg_f1,
            "total_count": others_count,
            "num_programs": len(other_progs),
        },
    }
