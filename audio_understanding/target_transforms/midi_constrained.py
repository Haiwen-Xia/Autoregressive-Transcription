"""Constrained decoder for MIDI token generation.

Enforces token generation order per event (matching MIDI2Tokens construction order):
    time_index -> name -> pitch -> velocity (onset only) -> program (optional)

Tracks violations: how many top-k candidates fall outside the allowed set at each step.
"""

import torch


class MidiConstrainedDecoder:

    # States
    EXPECT_TIME_OR_SEP = 0  # start of a new event, or [SEP] to end
    EXPECT_NAME = 1
    EXPECT_PITCH = 2
    EXPECT_VELOCITY = 3
    EXPECT_PROGRAM = 4
    STATE_NAMES = ["EXPECT_TIME_OR_SEP", "EXPECT_NAME", "EXPECT_PITCH", "EXPECT_VELOCITY", "EXPECT_PROGRAM"]

    def __init__(self, tokenizer, vocab_size: int, include_program: bool = False, device: str = "cuda"):
        """
        Args:
            tokenizer: BertMIDI tokenizer instance
            vocab_size: total vocabulary size
            include_program: whether program=X tokens are part of each event
            device: torch device
        """
        self.include_program = include_program
        self.vocab_size = vocab_size
        self.device = device

        tok = tokenizer.tok
        self.sep_id = tok.sep_token_id
        self.name_onset_id = tok.convert_tokens_to_ids("name=note_onset")
        self.name_offset_id = tok.convert_tokens_to_ids("name=note_offset")

        # Precompute allowed-ID boolean masks for each state, shape: (5, vocab_size)
        self.masks = torch.zeros(5, vocab_size, dtype=torch.bool, device=device)

        # EXPECT_TIME_OR_SEP: time_index=0..6000 or [SEP]
        time_start = tok.convert_tokens_to_ids("time_index=0")
        self.masks[self.EXPECT_TIME_OR_SEP, time_start : time_start + 6001] = True
        self.masks[self.EXPECT_TIME_OR_SEP, self.sep_id] = True

        # EXPECT_NAME: note_onset / note_offset
        self.masks[self.EXPECT_NAME, self.name_onset_id] = True
        self.masks[self.EXPECT_NAME, self.name_offset_id] = True

        # EXPECT_PITCH: pitch=0..127 plus optional drum_pitch=0..127
        pitch_start = tok.convert_tokens_to_ids("pitch=0")
        self.masks[self.EXPECT_PITCH, pitch_start : pitch_start + 128] = True
        drum_pitch_start = tok.convert_tokens_to_ids("drum_pitch=0")
        if drum_pitch_start != tok.unk_token_id:
            self.masks[self.EXPECT_PITCH, drum_pitch_start : drum_pitch_start + 128] = True

        # EXPECT_VELOCITY: velocity=0..127
        vel_start = tok.convert_tokens_to_ids("velocity=0")
        self.masks[self.EXPECT_VELOCITY, vel_start : vel_start + 128] = True

        # EXPECT_PROGRAM: program=0..127
        prog_start = tok.convert_tokens_to_ids("program=0")
        self.masks[self.EXPECT_PROGRAM, prog_start : prog_start + 129] = True

        # Runtime state
        self.state = self.EXPECT_TIME_OR_SEP
        self.current_name_id = None
        self.violations = 0
        self.total_topk_candidates = 0

    # ------------------------------------------------------------------
    def get_allowed_mask(self) -> torch.Tensor:
        """Boolean mask of shape (vocab_size,) for current state."""
        return self.masks[self.state]

    def count_violations(self, topk_ids: torch.Tensor) -> None:
        """Count how many top-k candidates violate the current constraint.

        Args:
            topk_ids: (k,) or (b, k) tensor of token IDs
        """
        allowed = self.masks[self.state]  # (vocab_size,)
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
        if self.state == self.EXPECT_TIME_OR_SEP:
            if token_id == self.sep_id:
                return False  # end of sequence
            self.state = self.EXPECT_NAME

        elif self.state == self.EXPECT_NAME:
            # got name=note_onset or name=note_offset
            self.current_name_id = token_id
            self.state = self.EXPECT_PITCH

        elif self.state == self.EXPECT_PITCH:
            if self.current_name_id == self.name_onset_id:
                self.state = self.EXPECT_VELOCITY
            else:
                # note_offset: no velocity
                if self.include_program:
                    self.state = self.EXPECT_PROGRAM
                else:
                    self.state = self.EXPECT_TIME_OR_SEP

        elif self.state == self.EXPECT_VELOCITY:
            if self.include_program:
                self.state = self.EXPECT_PROGRAM
            else:
                self.state = self.EXPECT_TIME_OR_SEP

        elif self.state == self.EXPECT_PROGRAM:
            self.state = self.EXPECT_TIME_OR_SEP

        return True

    def reset(self) -> None:
        self.state = self.EXPECT_TIME_OR_SEP
        self.current_name_id = None
        self.violations = 0
        self.total_topk_candidates = 0

    def __repr__(self) -> str:
        return (
            f"MidiConstrainedDecoder(state={self.STATE_NAMES[self.state]}, "
            f"include_program={self.include_program}, "
            f"violations={self.violations}/{self.total_topk_candidates})"
        )
