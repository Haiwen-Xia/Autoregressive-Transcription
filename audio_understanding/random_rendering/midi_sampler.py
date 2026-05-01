"""MidiSampler: note sampling → symusic Score with CC messages.

Wraps an existing note sampler (MarginalRandomSampler / UniformSampler etc.)
and produces a symusic Score object with proper MIDI tracks and CC events.

Usage:
    from audio_understanding.random_rendering.midi_sampler import MidiSampler
    sampler = MidiSampler(note_sampler, clip_duration=5.0, cc_config={...})
    score = sampler.sample(seed=42)
    score.dump_midi("/tmp/out.mid")
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np
from symusic import ControlChange, Note, Score, Tempo, Track


# ticks per quarter note — high resolution for accurate timing
TPQ = 960
# default tempo
DEFAULT_BPM = 120.0
# tick duration in seconds at default BPM
_TICK_DUR = 60.0 / (DEFAULT_BPM * TPQ)


def seconds_to_ticks(t: float) -> int:
    return round(t / _TICK_DUR)


def note_dicts_to_score(
    notes: List[Dict[str, Any]],
    clip_duration: float,
    cc_events: Optional[List[Dict[str, Any]]] = None,
) -> Score:
    """Convert note dicts + optional CC events to a symusic Score (tick-based).

    Args:
        notes: list of {"pitch", "start", "dur", "velocity", "program"}
        clip_duration: total clip length in seconds (unused in conversion, kept for API)
        cc_events: list of {"time": float, "number": int, "value": int, "program": int}
    """
    score = Score(ttype="tick")
    score.tempos.append(Tempo(time=0, qpm=DEFAULT_BPM))

    # group notes by program
    tracks_dict: Dict[int, Track] = {}
    for n in notes:
        prog = int(n["program"])
        if prog not in tracks_dict:
            tracks_dict[prog] = Track(program=prog, is_drum=False)
        t = tracks_dict[prog]
        t.notes.append(Note(
            time=seconds_to_ticks(n["start"]),
            duration=max(1, seconds_to_ticks(n["dur"])),
            pitch=int(n["pitch"]),
            velocity=int(n["velocity"]),
        ))

    # add CC events
    if cc_events:
        for cc in cc_events:
            prog = int(cc.get("program", 0))
            if prog not in tracks_dict:
                tracks_dict[prog] = Track(program=prog, is_drum=False)
            tracks_dict[prog].controls.append(ControlChange(
                time=seconds_to_ticks(cc["time"]),
                number=int(cc["number"]),
                value=int(cc["value"]),
            ))

    for t in tracks_dict.values():
        t.sort()
        score.tracks.append(t)

    return score


# ---- CC generators ----

def generate_sustain_pedal_cc(
    clip_duration: float,
    rng: np.random.RandomState,
    n_presses_range: tuple = (0, 4),
    min_hold: float = 0.3,
    max_hold: float = 2.0,
) -> List[Dict[str, Any]]:
    """Generate random sustain pedal (CC64) on/off pairs."""
    n = rng.randint(n_presses_range[0], n_presses_range[1] + 1)
    if n == 0:
        return []

    events = []
    # sample non-overlapping pedal intervals
    starts = sorted(rng.uniform(0, clip_duration - min_hold, size=n))
    for s in starts:
        hold = rng.uniform(min_hold, min(max_hold, clip_duration - s))
        events.append({"time": s, "number": 64, "value": 127, "program": 0})
        events.append({"time": s + hold, "number": 64, "value": 0, "program": 0})
    return events


def generate_soft_pedal_cc(
    clip_duration: float,
    rng: np.random.RandomState,
    prob: float = 0.2,
) -> List[Dict[str, Any]]:
    """Optionally turn soft pedal (CC67) on for the whole clip."""
    if rng.random() < prob:
        return [
            {"time": 0.0, "number": 67, "value": 127, "program": 0},
            {"time": clip_duration, "number": 67, "value": 0, "program": 0},
        ]
    return []


CC_GENERATORS = {
    "sustain": generate_sustain_pedal_cc,
    "soft_pedal": generate_soft_pedal_cc,
}


class MidiSampler:
    """Wraps a note sampler and adds CC events → symusic Score.

    Args:
        note_sampler: any BaseSampler subclass with .sample(seed) -> List[Dict]
        clip_duration: clip length in seconds
        cc_config: which CC generators to enable and their kwargs.
            e.g. {"sustain": {"n_presses_range": [0, 3]}, "soft_pedal": {"prob": 0.15}}
            Set to None or {} to disable all CC.
    """
    def __init__(
        self,
        note_sampler,
        clip_duration: float,
        cc_config: Optional[Dict[str, Dict]] = None,
    ):
        self.note_sampler = note_sampler
        self.clip_duration = clip_duration
        self.cc_config = cc_config or {}

    def sample(self, seed: Optional[int] = None) -> tuple[Score, List[Dict[str, Any]]]:
        """Sample notes + CC → (symusic Score, note_dicts).

        Returns both the Score (for MIDI file export) and the raw note_dicts
        (for token conversion / ground truth).
        """
        notes = self.note_sampler.sample(seed=seed)

        rng = np.random.RandomState(seed)
        cc_events = []
        for name, kwargs in self.cc_config.items():
            assert name in CC_GENERATORS, f"Unknown CC generator: {name}"
            cc_events.extend(CC_GENERATORS[name](self.clip_duration, rng, **kwargs))

        score = note_dicts_to_score(notes, self.clip_duration, cc_events)
        return score, notes

    def sample_and_save(self, path: str, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Sample → save MIDI file, return note_dicts."""
        score, notes = self.sample(seed=seed)
        score.dump_midi(path)
        return notes
