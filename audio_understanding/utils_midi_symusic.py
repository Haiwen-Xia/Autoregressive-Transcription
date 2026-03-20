from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

from pretty_midi import Instrument, Note as PMNote, PrettyMIDI
from symusic import Note, Score, Track

DRUM_PROGRAM = 128


def prettymidi_to_symusic(midi_data: Any, ttype: str = "second") -> Any:
    assert isinstance(midi_data, PrettyMIDI)

    with NamedTemporaryFile(suffix=".mid", delete=True) as f:
        temp_path = Path(f.name)
        midi_data.write(str(temp_path))
        score = Score.from_midi(temp_path.read_bytes(), ttype=ttype)

    return cast(Any, score)


def symusic_to_prettymidi(score: Any) -> PrettyMIDI:
    midi = PrettyMIDI()

    for track in score.tracks:
        is_drum = bool(getattr(track, "is_drum", False))
        src_program = int(track.program)
        midi_program = 0 if is_drum else max(0, min(127, src_program))

        inst = Instrument(program=midi_program)
        inst.is_drum = is_drum

        for note in track.notes:
            start = float(note.time)
            end = float(note.time + note.duration)
            if end <= start:
                continue

            inst.notes.append(
                PMNote(
                    velocity=int(note.velocity),
                    pitch=int(note.pitch),
                    start=start,
                    end=end,
                )
            )

        if len(inst.notes) > 0:
            midi.instruments.append(inst)

    return midi


def clip_symusic_notes(
    notes: list,
    start_time: float,
    duration: float,
    mode: str = "overlap",
) -> tuple[list, list[dict]]:
    assert mode in ["overlap", "clip"], mode

    end_time = start_time + duration
    out_notes: list = []
    cropped_list: list[dict] = []

    for note in notes:
        note_start = float(note.time)
        note_end = float(note.time + note.duration)

        if note_end <= start_time or note_start >= end_time:
            continue

        lhs_cropped = note_start < start_time
        rhs_cropped = note_end > end_time

        if mode == "overlap":
            out_notes.append(
                Note(
                    note_start,
                    max(0.0, note_end - note_start),
                    int(note.pitch),
                    int(note.velocity),
                    ttype="second",
                )
            )
            cropped_list.append({"lhs_cropped": lhs_cropped, "rhs_cropped": rhs_cropped})
            continue

        clipped_start = max(note_start, start_time)
        clipped_end = min(note_end, end_time)
        clipped_duration = clipped_end - clipped_start
        if clipped_duration <= 0:
            continue

        out_notes.append(
            Note(
                clipped_start,
                clipped_duration,
                int(note.pitch),
                int(note.velocity),
                ttype="second",
            )
        )
        cropped_list.append({"lhs_cropped": lhs_cropped, "rhs_cropped": rhs_cropped})

    return out_notes, cropped_list


def read_midi_clip_symusic(
    midi_path: str,
    start_time: float,
    duration: float,
    mode: str = "clip",
    n_program: int = 129,
    n_pitch: int = 128,
    use_drum_bin: bool = True,
) -> tuple[Any, list[int], list[int], list[dict]]:
    assert mode in ["overlap", "clip"], mode
    assert n_pitch == 128

    if use_drum_bin:
        assert n_program >= 129

    score = cast(Any, Score(midi_path, ttype="second"))

    lhs_flags = [0] * (n_program * n_pitch)
    rhs_flags = [0] * (n_program * n_pitch)

    clipped_score = cast(Any, Score(ttype="second"))
    crop_records: list[dict] = []

    for track_idx, track in enumerate(score.tracks):
        is_drum = bool(getattr(track, "is_drum", False))
        if is_drum:
            if not use_drum_bin:
                continue
            program_for_flags = DRUM_PROGRAM
        else:
            program_for_flags = int(track.program)

        assert 0 <= program_for_flags < n_program

        clipped_notes, cropped_list = clip_symusic_notes(
            notes=list(track.notes),
            start_time=start_time,
            duration=duration,
            mode=mode,
        )

        for note, crop in zip(clipped_notes, cropped_list):
            pitch = int(note.pitch)
            flat_idx = program_for_flags * n_pitch + pitch
            if crop["lhs_cropped"]:
                lhs_flags[flat_idx] = 1
            if crop["rhs_cropped"]:
                rhs_flags[flat_idx] = 1

        if len(clipped_notes) == 0:
            continue

        new_track = Track(
            name=track.name,
            program=int(track.program),
            is_drum=is_drum,
            ttype="second",
        )
        for note in clipped_notes:
            new_track.notes.append(note)
        clipped_score.tracks.append(new_track)

        crop_records.append(
            {
                "track_index": track_idx,
                "track_name": track.name,
                "program": int(track.program),
                "is_drum": is_drum,
                "cropped_list": cropped_list,
            }
        )

    return clipped_score, lhs_flags, rhs_flags, crop_records


def score_to_event_like_tokens(
    score: Any,
    fps: float = 100.0,
    include_program: bool = True,
) -> list[str]:
    events: list[tuple[tuple[int, str, int, int, int], list[str]]] = []

    for track in score.tracks:
        is_drum = bool(getattr(track, "is_drum", False))
        program = DRUM_PROGRAM if is_drum else int(track.program)

        for note in track.notes:
            onset_idx = round(float(note.time) * fps)
            offset_idx = round(float(note.time + note.duration) * fps)
            if offset_idx <= onset_idx:
                offset_idx = onset_idx + 1

            onset = [
                "name=note_onset",
                f"time_index={onset_idx}",
                f"pitch={int(note.pitch)}",
                f"velocity={int(note.velocity)}",
            ]
            offset = [
                "name=note_offset",
                f"time_index={offset_idx}",
                f"pitch={int(note.pitch)}",
            ]

            if include_program:
                onset.append(f"program={program}")
                offset.append(f"program={program}")

            sort_program = program if include_program else -1
            onset_key = (onset_idx, "note_onset", sort_program, int(note.pitch), int(note.velocity))
            offset_key = (offset_idx, "note_offset", sort_program, int(note.pitch), 0)
            events.append((onset_key, onset))
            events.append((offset_key, offset))

    events.sort(key=lambda x: x[0])
    return [tok for _, event in events for tok in event]


def midi_to_token_string(
    midi_obj: Any,
    fps: float = 100.0,
    include_program: bool = True,
) -> list[str]:
    if isinstance(midi_obj, PrettyMIDI):
        score = prettymidi_to_symusic(midi_obj, ttype="second")
    elif isinstance(midi_obj, (str, Path)):
        score = cast(Any, Score(str(midi_obj), ttype="second"))
    else:
        score = midi_obj

    return score_to_event_like_tokens(score=score, fps=fps, include_program=include_program)


def token_string_to_score(
    tokens: list[str],
    fps: float = 100.0,
    include_program: bool = True,
) -> Any:
    open_notes: dict[tuple[int, int], list[dict]] = {}
    finished: list[dict] = []

    n = len(tokens)
    i = 0

    while i < n:
        assert "=" in tokens[i]
        key, value = tokens[i].split("=", 1)
        assert key == "name"

        if value == "note_onset":
            event_len = 5 if include_program else 4
            assert i + event_len <= n

            t_key, t_val = tokens[i + 1].split("=", 1)
            p_key, p_val = tokens[i + 2].split("=", 1)
            v_key, v_val = tokens[i + 3].split("=", 1)
            assert t_key == "time_index"
            assert p_key == "pitch"
            assert v_key == "velocity"

            time_index = int(t_val)
            pitch = int(p_val)
            velocity = int(v_val)
            program = 0

            if include_program:
                g_key, g_val = tokens[i + 4].split("=", 1)
                assert g_key == "program"
                program = int(g_val)

            open_key = (pitch, program)
            open_notes.setdefault(open_key, []).append(
                {
                    "onset_time": time_index / fps,
                    "pitch": pitch,
                    "velocity": velocity,
                    "program": program,
                }
            )
            i += event_len
            continue

        if value == "note_offset":
            event_len = 4 if include_program else 3
            assert i + event_len <= n

            t_key, t_val = tokens[i + 1].split("=", 1)
            p_key, p_val = tokens[i + 2].split("=", 1)
            assert t_key == "time_index"
            assert p_key == "pitch"

            time_index = int(t_val)
            pitch = int(p_val)
            program = 0

            if include_program:
                g_key, g_val = tokens[i + 3].split("=", 1)
                assert g_key == "program"
                program = int(g_val)

            open_key = (pitch, program)
            assert open_key in open_notes and len(open_notes[open_key]) > 0

            note = open_notes[open_key].pop(0)
            offset_time = time_index / fps
            if offset_time <= note["onset_time"]:
                offset_time = note["onset_time"] + 1.0 / fps
            note["offset_time"] = offset_time
            finished.append(note)

            i += event_len
            continue

        raise AssertionError(value)

    assert sum(len(v) for v in open_notes.values()) == 0

    score = cast(Any, Score(ttype="second"))
    tracks: dict[int, Any] = {}

    for note in finished:
        program = int(note["program"])
        is_drum = program == DRUM_PROGRAM
        track_program = 0 if is_drum else max(0, min(127, program))

        if program not in tracks:
            track = Track(
                name=f"program_{program}",
                program=track_program,
                is_drum=is_drum,
                ttype="second",
            )
            tracks[program] = track
            score.tracks.append(track)

        tracks[program].notes.append(
            Note(
                float(note["onset_time"]),
                float(note["offset_time"] - note["onset_time"]),
                int(note["pitch"]),
                int(note["velocity"]),
                ttype="second",
            )
        )

    return score


def token_string_to_midi(
    tokens: list[str],
    fps: float = 100.0,
    include_program: bool = True,
) -> PrettyMIDI:
    score = token_string_to_score(tokens=tokens, fps=fps, include_program=include_program)
    return symusic_to_prettymidi(score)


def evaluate_token_string(
    ref_score: Any,
    est_tokens: list[str],
    fps: float = 100.0,
    include_program: bool = True,
) -> dict:
    from audio_understanding.eval.transcription.metrics import (
        note_onset_f1,
        note_with_offset_f1,
        per_instrument_metrics,
        program_aware_f1,
    )

    ref_notes = []
    for track in ref_score.tracks:
        is_drum = bool(getattr(track, "is_drum", False))
        program = DRUM_PROGRAM if is_drum else int(track.program)

        for note in track.notes:
            onset_time = float(note.time)
            offset_time = float(note.time + note.duration)
            if offset_time <= onset_time:
                offset_time = onset_time + 1.0 / fps

            ref_notes.append(
                {
                    "onset_time": onset_time,
                    "offset_time": offset_time,
                    "pitch": int(note.pitch),
                    "velocity": int(note.velocity),
                    "program": program,
                    "is_drum": is_drum,
                }
            )

    est_score = token_string_to_score(tokens=est_tokens, fps=fps, include_program=include_program)
    est_notes = []
    for track in est_score.tracks:
        is_drum = bool(getattr(track, "is_drum", False))
        program = DRUM_PROGRAM if is_drum else int(track.program)

        for note in track.notes:
            est_notes.append(
                {
                    "onset_time": float(note.time),
                    "offset_time": float(note.time + note.duration),
                    "pitch": int(note.pitch),
                    "velocity": int(note.velocity),
                    "program": program,
                    "is_drum": is_drum,
                }
            )

    result = {
        "note_onset": note_onset_f1(ref_notes, est_notes),
        "note_offset": note_with_offset_f1(ref_notes, est_notes),
    }

    if include_program:
        result["program_aware"] = program_aware_f1(ref_notes, est_notes)
        result["per_instrument"] = per_instrument_metrics(ref_notes, est_notes)

    return result
