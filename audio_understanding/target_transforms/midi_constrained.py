"""Constrained decoder for MIDI token generation.

Enforces token generation order per event (matching MIDI2Tokens construction order):
    time_first: time_index -> name -> pitch -> velocity (onset only) -> program (optional)
    name_first: name -> time_index -> pitch -> velocity (onset only) -> program (optional)

Tracks violations: how many top-k candidates fall outside the allowed set at each step.
"""

import torch


class MidiConstrainedDecoder:

    # States
    EXPECT_START_OR_SEP = 0  # start of a new event, or [SEP] to end
    EXPECT_SECOND = 1
    EXPECT_PITCH = 2
    EXPECT_VELOCITY = 3
    EXPECT_PROGRAM = 4
    STATE_NAMES = ["EXPECT_START_OR_SEP", "EXPECT_SECOND", "EXPECT_PITCH", "EXPECT_VELOCITY", "EXPECT_PROGRAM"]

    def __init__(
        self,
        tokenizer,
        vocab_size: int,
        include_program: bool = False,
        token_order: str = "time_first",
        device: str = "cuda",
    ):
        """
        Args:
            tokenizer: BertMIDI tokenizer instance
            vocab_size: total vocabulary size
            include_program: whether program=X tokens are part of each event
            token_order: one of ["time_first", "name_first"]
            device: torch device
        """
        self.include_program = include_program
        self.vocab_size = vocab_size
        self.device = device
        assert token_order in ["time_first", "name_first"]
        self.token_order = token_order

        tok = tokenizer.tok
        self.sep_id = tok.sep_token_id
        self.name_onset_id = tok.convert_tokens_to_ids("name=note_onset")
        self.name_offset_id = tok.convert_tokens_to_ids("name=note_offset")

        # Precompute reusable masks.
        self.mask_time_or_sep = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        self.mask_name_or_sep = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        self.mask_time_only = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        self.mask_name_only = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        self.mask_pitch_only = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        self.mask_velocity_only = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        self.mask_program_only = torch.zeros(vocab_size, dtype=torch.bool, device=device)

        # time_index=0..6000
        time_start = tok.convert_tokens_to_ids("time_index=0")
        self.mask_time_or_sep[time_start : time_start + 6001] = True
        self.mask_time_or_sep[self.sep_id] = True
        self.mask_time_only[time_start : time_start + 6001] = True

        # name=note_onset / name=note_offset
        self.mask_name_or_sep[self.name_onset_id] = True
        self.mask_name_or_sep[self.name_offset_id] = True
        self.mask_name_or_sep[self.sep_id] = True
        self.mask_name_only[self.name_onset_id] = True
        self.mask_name_only[self.name_offset_id] = True

        # pitch=0..127 plus optional drum_pitch=0..127
        pitch_start = tok.convert_tokens_to_ids("pitch=0")
        self.mask_pitch_only[pitch_start : pitch_start + 128] = True
        drum_pitch_start = tok.convert_tokens_to_ids("drum_pitch=0")
        if drum_pitch_start != tok.unk_token_id:
            self.mask_pitch_only[drum_pitch_start : drum_pitch_start + 128] = True

        # velocity=0..127
        vel_start = tok.convert_tokens_to_ids("velocity=0")
        self.mask_velocity_only[vel_start : vel_start + 128] = True

        # program=0..128
        prog_start = tok.convert_tokens_to_ids("program=0")
        self.mask_program_only[prog_start : prog_start + 129] = True

        # Runtime state
        self.state = self.EXPECT_START_OR_SEP
        self.current_name_id = None
        self.violations = 0
        self.total_topk_candidates = 0

    # ------------------------------------------------------------------
    def get_allowed_mask(self) -> torch.Tensor:
        """Boolean mask of shape (vocab_size,) for current state."""
        if self.state == self.EXPECT_START_OR_SEP:
            if self.token_order == "time_first":
                return self.mask_time_or_sep
            return self.mask_name_or_sep
        if self.state == self.EXPECT_SECOND:
            if self.token_order == "time_first":
                return self.mask_name_only
            return self.mask_time_only
        if self.state == self.EXPECT_PITCH:
            return self.mask_pitch_only
        if self.state == self.EXPECT_VELOCITY:
            return self.mask_velocity_only
        if self.state == self.EXPECT_PROGRAM:
            return self.mask_program_only
        raise ValueError(self.state)

    def count_violations(self, topk_ids: torch.Tensor) -> None:
        """Count how many top-k candidates violate the current constraint.

        Args:
            topk_ids: (k,) or (b, k) tensor of token IDs
        """
        allowed = self.get_allowed_mask()  # (vocab_size,)
        flat = topk_ids.reshape(-1)
        n = flat.shape[0]
        self.total_topk_candidates += n
        self.violations += int((~allowed[flat]).sum().item())

    def update(self, token_id: int) -> bool:
        """Transition state after generating *token_id*.

        Returns:
            True  -> keep generating
            False -> stop (SEP was generated)
        """
        if self.state == self.EXPECT_START_OR_SEP:
            if token_id == self.sep_id:
                return False  # end of sequence
            if self.token_order == "time_first":
                self.state = self.EXPECT_SECOND
            else:
                # name_first: first token is name.
                self.current_name_id = token_id
                self.state = self.EXPECT_SECOND

        elif self.state == self.EXPECT_SECOND:
            if self.token_order == "time_first":
                # time_first: second token is name.
                self.current_name_id = token_id
            # name_first: second token is time_index.
            self.state = self.EXPECT_PITCH

        elif self.state == self.EXPECT_PITCH:
            if self.current_name_id == self.name_onset_id:
                self.state = self.EXPECT_VELOCITY
            else:
                # note_offset: no velocity
                if self.include_program:
                    self.state = self.EXPECT_PROGRAM
                else:
                    self.state = self.EXPECT_START_OR_SEP

        elif self.state == self.EXPECT_VELOCITY:
            if self.include_program:
                self.state = self.EXPECT_PROGRAM
            else:
                self.state = self.EXPECT_START_OR_SEP

        elif self.state == self.EXPECT_PROGRAM:
            self.state = self.EXPECT_START_OR_SEP

        return True

    def reset(self) -> None:
        self.state = self.EXPECT_START_OR_SEP
        self.current_name_id = None
        self.violations = 0
        self.total_topk_candidates = 0

    def __repr__(self) -> str:
        return (
            f"MidiConstrainedDecoder(state={self.STATE_NAMES[self.state]}, "
            f"include_program={self.include_program}, "
            f"violations={self.violations}/{self.total_topk_candidates})"
        )
