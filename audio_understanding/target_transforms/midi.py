class MIDI2Tokens:
    def __init__(
        self,
        fps: float,
        include_program: bool = True,
        drum_pitch: bool = False,
        event_token_order: str = "time_first",
    ) -> None:
        r"""Convert MIDI events to captions."""
        self.fps = fps
        self.include_program = include_program
        self.drum_pitch = drum_pitch
        assert event_token_order in ["time_first", "name_first"]
        self.event_token_order = event_token_order

    def __call__(self, data: dict) -> dict:
        r"""Convert data of MIDI events to tokens.
        
        Args:
            data: dict

        Outputs:
            tokens: list[str], e.g., ["time_index=15", "name=note_onset", "pitch=36", "velocity=27", ...]
        """

        start_time = data["start_time"]
        duration = data["duration"]
        end_time = start_time + duration

        notes = data["note"]
        pedals = data["pedal"]
        note_program = data.get("note_program")
        note_is_drum = data.get("note_is_drum")
        if self.include_program and note_program is not None:
            assert len(note_program) == len(notes)
        if note_is_drum is not None:
            assert len(note_is_drum) == len(notes)

        events = []

        for idx, note in enumerate(notes):
            pitch_token_key = self.get_pitch_token_key(
                idx=idx,
                note_program=note_program,
                note_is_drum=note_is_drum,
            )

            program_token = []
            if self.include_program and note_program is not None:
                program_token = ["program={}".format(note_program[idx])]

            if note.end < start_time:
                pass

            elif (note.start < start_time) and (start_time <= note.end <= end_time):
                
                events.append(
                    self.build_event_tokens(
                        time_index=round((note.end - start_time) * self.fps),
                        name="note_offset",
                        pitch_token_key=pitch_token_key,
                        pitch=int(note.pitch),
                        velocity=None,
                        program_token=program_token,
                    )
                )

            elif (note.start < start_time) and (end_time < note.end):
                pass

            elif (start_time <= note.start <= end_time) and (start_time <= note.end <= end_time):

                onset_idx = round((note.start - start_time) * self.fps)
                offset_idx = round((note.end - start_time) * self.fps)
                # Ensure offset comes after onset in the token sequence.
                # When they fall on the same frame (zero-duration notes, common
                # for drums), the sort key would place offset before onset,
                # breaking the parser's open-note matching.
                #! this only works on segment handling and should be removed in the future. Theoretically, zero-duration notes CAN BE detected.
                if offset_idx <= onset_idx:
                    offset_idx = onset_idx + 1
                #* time_index applied first
                events.append(
                    self.build_event_tokens(
                        time_index=onset_idx,
                        name="note_onset",
                        pitch_token_key=pitch_token_key,
                        pitch=int(note.pitch),
                        velocity=int(note.velocity),
                        program_token=program_token,
                    )
                )

                events.append(
                    self.build_event_tokens(
                        time_index=offset_idx,
                        name="note_offset",
                        pitch_token_key=pitch_token_key,
                        pitch=int(note.pitch),
                        velocity=None,
                        program_token=program_token,
                    )
                )

            elif (start_time <= note.start <= end_time) and (end_time < note.end):

                events.append(
                    self.build_event_tokens(
                        time_index=round((note.start - start_time) * self.fps),
                        name="note_onset",
                        pitch_token_key=pitch_token_key,
                        pitch=int(note.pitch),
                        velocity=int(note.velocity),
                        program_token=program_token,
                    )
                )

            elif end_time < note.start:
                pass

            else:
                raise NotImplementedError

        # Sort events by time
        events = self.sort_events(events)

        # Flat tokens
        tokens = self.flat_events(events)

        data.update({"token": tokens})

        return data

    def sort_events(self, events: list[list[str]]) -> list[list[str]]:
        r"""Sort events by time.

        Args:
            events: e.g., [
                ["name=note_offset", "time_index=497", "pitch=69"]
                ["name=note_onset", "time_index=480", "pitch=69", "velocity=62"],
                ...]
            
        Returns:
            sorted_events: e.g., [
                ["name=note_onset", "time_index=480", "pitch=69", "velocity=62"],
                ["name=note_offset", "time_index=497", "pitch=69"]
                ...]
        """
        
        pairs = []

        for event in events:
            pair = self.get_key_value_pair(event)
            pairs.append(pair)

        pairs.sort(key=lambda x: x[0])

        sorted_events = [x[1] for x in pairs]

        return sorted_events

    def get_key_value_pair(self, event: list[str]) -> tuple[str, list[str]]:
        r"""Get key and value pair for sorting events.

        Args:
            event: list[str], e.g., ["name=note_offset", "time_index=56", "pitch=44"]

        Returns:
            key: e.g., "time_index=000056,name=note_offset,pitch=000044"
            value: e.g., ["name=note_offset", "time_index=56", "pitch=44"]
        """

        desired_order = ["time_index", "name", "program", "pitch", "drum_pitch", "velocity"]
        
        # Sort tokens by desired order
        sorted_tokens = sorted(event, key=lambda x: desired_order.index(x.split('=')[0]))
        # E.g., ["time_index=56", 'name=note_offset', "pitch=44"]

        # Pad 0 for sort
        extended_tokens = [self.extend_token(token) for token in sorted_tokens]
        # E.g., ["time_index=000056", 'name=note_offset', "pitch=000044"]

        key = ",".join(extended_tokens)
        # E.g., "time_index=000056,name=00_note_offset,pitch=000044"

        return key, event

    def build_event_tokens(
        self,
        time_index: int,
        name: str,
        pitch_token_key: str,
        pitch: int,
        velocity: None | int,
        program_token: list[str],
    ) -> list[str]:
        assert name in ["note_onset", "note_offset"]

        ordered_tokens: list[str] = []
        if self.event_token_order == "time_first":
            ordered_tokens.append("time_index={}".format(int(time_index)))
            ordered_tokens.append("name={}".format(name))
        else:
            ordered_tokens.append("name={}".format(name))
            ordered_tokens.append("time_index={}".format(int(time_index)))

        ordered_tokens.append("{}={}".format(pitch_token_key, int(pitch)))
        if velocity is not None:
            ordered_tokens.append("velocity={}".format(int(velocity)))
        ordered_tokens += program_token
        return ordered_tokens

    def extend_token(self, token: str) -> str:
        r"""Left pad values for sorting."""

        key, value = token.split("=")

        if value == "note_offset":
            return "{}=00_{}".format(key, value)

        elif value == "note_onset":
            return "{}=01_{}".format(key, value)

        elif value.isdigit():
            return "{}={:06d}".format(key, int(value))

        else:
            raise NotImplementedError(token)

    def get_pitch_token_key(
        self,
        idx: int,
        note_program: list[int] | None,
        note_is_drum: list[bool] | None,
    ) -> str:
        if not self.drum_pitch:
            return "pitch"

        is_drum = False
        if note_is_drum is not None:
            is_drum = bool(note_is_drum[idx])
        elif note_program is not None:
            is_drum = int(note_program[idx]) == 128

        if is_drum:
            return "drum_pitch"
        return "pitch"

    def flat_events(self, events: list[list[str]]) -> list[str]:

        tokens = []

        for event in events:
            tokens += event

        return tokens