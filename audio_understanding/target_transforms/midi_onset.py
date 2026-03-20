from __future__ import annotations


class MIDI2OnsetTokens:
    r"""Convert notes to onset-only tokens.

    Output token format per event:
        ["time_index=<int>", "pitch=<int>" or "drum_pitch=<int>"]

    Only note onsets inside the current cropped clip are emitted.
    No offset / velocity / program tokens are produced.
    """

    def __init__(self, fps: float, drum_pitch: bool = False) -> None:
        self.fps = fps
        self.drum_pitch = drum_pitch

    def __call__(self, data: dict) -> dict:
        start_time = float(data["start_time"])
        duration = float(data["duration"])
        end_time = start_time + duration

        notes = data["note"]
        note_program = data.get("note_program")
        note_is_drum = data.get("note_is_drum")
        if note_program is not None:
            assert len(note_program) == len(notes)
        if note_is_drum is not None:
            assert len(note_is_drum) == len(notes)

        events: list[list[str]] = []

        for idx, note in enumerate(notes):
            onset_time = float(note.start)
            if onset_time < start_time:
                continue
            if onset_time > end_time:
                continue

            onset_idx = round((onset_time - start_time) * self.fps)
            if onset_idx < 0:
                continue

            pitch_token_key = "pitch"
            if self.drum_pitch:
                is_drum = False
                if note_is_drum is not None:
                    is_drum = bool(note_is_drum[idx])
                elif note_program is not None:
                    is_drum = int(note_program[idx]) == 128
                if is_drum:
                    pitch_token_key = "drum_pitch"

            events.append([
                f"time_index={onset_idx}",
                f"{pitch_token_key}={int(note.pitch)}",
            ])

        # Stable sort by frame index.
        events.sort(key=lambda ev: int(ev[0].split("=", 1)[1]))

        tokens: list[str] = []
        for event in events:
            tokens += event

        data.update({"token": tokens})
        return data
