"""Online rendering buffer for training.

Workflow (buffer mode — autoregressive tokens):
    buffer.render_epoch(epoch_size)   # sample + render + tokenize → epoch list
    for batch in buffer.iterate_epoch(batch_size, repeat=3):
        audio, question, answering = get_audio_question_answering(batch)
        ...

Workflow (iterator mode — autoregressive tokens):
    for data in buffer:               # auto re-renders each epoch
        ...

Workflow (iterator mode — framewise piano rolls):
    buf = FramewiseRenderingBuffer(render_engine, note_sampler, ...)
    for data in buf:
        audio, frame_roll, onset_roll, offset_roll = ...
"""
from __future__ import annotations

import multiprocessing as mp
import random
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from audio_understanding.random_rendering.rendering import Sampler as RenderEngine
from audio_understanding.random_rendering.sampler import MarginalRandomSampler

TRANSCRIPTION_QUESTIONS = [
    "Music transcription.",
    "Convert audio music into MIDI data format.",
    "Transcribe music recordings into MIDI note sequences.",
    "Automatically generate MIDI file from audio music.",
    "Extract music elements and convert to MIDI notes.",
]


def notes_to_tokens(
    notes: List[Dict],
    fps: float,
    include_program: bool = True,
    event_token_order: str = "time_first",
) -> List[str]:
    """Convert sampler note dicts to flat token list, matching MIDI2Tokens output.

    Each note produces an onset event and an offset event.
    Events are sorted by (time_index, name, pitch) then flattened.
    """
    events: List[List[str]] = []

    for n in notes:
        onset_idx = round(n["start"] * fps)
        offset_idx = round((n["start"] + n["dur"]) * fps)
        if offset_idx <= onset_idx:
            offset_idx = onset_idx + 1

        program_token = [f"program={int(n['program'])}"] if include_program else []

        # onset
        if event_token_order == "time_first":
            onset_ev = [f"time_index={onset_idx}", "name=note_onset"]
        else:
            onset_ev = ["name=note_onset", f"time_index={onset_idx}"]
        onset_ev.append(f"pitch={int(n['pitch'])}")
        onset_ev.append(f"velocity={int(n['velocity'])}")
        onset_ev += program_token
        events.append(onset_ev)

        # offset
        if event_token_order == "time_first":
            offset_ev = [f"time_index={offset_idx}", "name=note_offset"]
        else:
            offset_ev = ["name=note_offset", f"time_index={offset_idx}"]
        offset_ev.append(f"pitch={int(n['pitch'])}")
        offset_ev += program_token
        events.append(offset_ev)

    # Sort by (time_index, name_order, pitch) — same key logic as MIDI2Tokens
    def _sort_key(ev: List[str]) -> str:
        desired_order = ["time_index", "name", "program", "pitch", "drum_pitch", "velocity"]
        sorted_tokens = sorted(ev, key=lambda x: desired_order.index(x.split("=")[0]))
        parts = []
        for tok in sorted_tokens:
            k, v = tok.split("=", 1)
            if v == "note_offset":
                parts.append(f"{k}=00_{v}")
            elif v == "note_onset":
                parts.append(f"{k}=01_{v}")
            elif v.lstrip("-").isdigit():
                parts.append(f"{k}={int(v):06d}")
            else:
                parts.append(tok)
        return ",".join(parts)

    events.sort(key=_sort_key)

    tokens: List[str] = []
    for ev in events:
        tokens += ev
    return tokens


def make_token_fn(clip_duration: float, **midi2tokens_kwargs) -> Callable[[List[Dict]], List[str]]:
    """Create a token_fn that delegates to MIDI2Tokens.

    This avoids maintaining a separate token conversion and guarantees
    output is identical to the MAESTRO training pipeline.
    """
    from audio_understanding.target_transforms.midi import MIDI2Tokens

    m2t = MIDI2Tokens(**midi2tokens_kwargs)

    def token_fn(note_dicts: List[Dict]) -> List[str]:
        notes = [SimpleNamespace(
            start=n["start"],
            end=n["start"] + n["dur"],
            pitch=n["pitch"],
            velocity=n["velocity"],
        ) for n in note_dicts]
        data = {
            "start_time": 0.0,
            "duration": clip_duration,
            "note": notes,
            "pedal": [],
            "note_program": [n["program"] for n in note_dicts],
            "note_is_drum": [False] * len(note_dicts),
        }
        return m2t(data)["token"]

    return token_fn


# ---- multiprocessing worker (module-level for spawn) ----
_mp_ctx: dict = {}


def _render_single(note_dicts): #* use processes
    audio = _mp_ctx["engine"].render_notes(note_dicts, time=_mp_ctx["clip_duration"])
    tokens = _mp_ctx["token_fn"](note_dicts)
    return audio, tokens


def _spawn_worker_init(engine_spec: dict, token_fn_kwargs: dict):
    """Initializer for spawn pool workers.

    Reconstructs the render engine and token_fn from serializable specs.
    spawn workers start with an empty interpreter (no inherited _mp_ctx, no CUDA),
    so everything must be built from scratch here.
    """
    engine_type = engine_spec["type"]
    if engine_type == "dawdreamer":
        from audio_understanding.random_rendering.rendering import DawDreamerSampler
        engine = DawDreamerSampler(
            time=engine_spec["time"],
            sr=engine_spec["sr"],
            vst_path=engine_spec["vst_path"],
            buffer_size=engine_spec["buffer_size"],
        )
    else:
        assert engine_type == "sampler"
        from audio_understanding.random_rendering.rendering import Sampler
        engine = Sampler(
            time=engine_spec["time"],
            sr=engine_spec["sr"],
            data_dir=engine_spec["data_dir"],
        )
    _mp_ctx["engine"] = engine
    _mp_ctx["clip_duration"] = engine_spec["time"]
    _mp_ctx["token_fn"] = make_token_fn(**token_fn_kwargs)


class OnlineRenderingBuffer:
    """Online rendering buffer with shuffle+repeat epoch iteration.

    Two usage modes:

    1. Manual (explicit render_epoch + iterate_epoch):
        buf.render_epoch(epoch_size=256)
        for batch in buf.iterate_epoch(batch_size=4, repeat=3):
            ...

    2. Iterator (auto re-render, tracks epoch/step internally):
        buf = OnlineRenderingBuffer(..., epoch_size=256, repeat_times=3, batch_size=4)
        for data in buf:          # __iter__/__next__ handle render + shuffle
            ...                   # runs forever; stop externally via break
        print(buf.global_step, buf.epoch_counter)
    """

    def __init__(
        self,
        render_engine: RenderEngine,
        note_sampler: MarginalRandomSampler,
        clip_duration: float,
        token_fn: Callable[[List[Dict]], List[str]],
        engine_spec: Optional[dict] = None,
        token_fn_kwargs: Optional[dict] = None,
        # --- iterator-mode params (optional, needed for __iter__) ---
        epoch_size: int = 0,
        repeat_times: int = 1,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        self.render_engine = render_engine
        self.note_sampler = note_sampler
        self.clip_duration = clip_duration
        self.token_fn = token_fn
        # Needed to reconstruct engine/token_fn in spawn workers (no CUDA inheritance)
        self.engine_spec = engine_spec
        self.token_fn_kwargs = token_fn_kwargs
        self.epoch: List[Dict[str, Any]] = []

        # iterator-mode state
        self._epoch_size = epoch_size
        self._repeat_times = repeat_times
        self._batch_size = batch_size
        self._num_workers = num_workers
        self.epoch_counter: int = 0
        self.global_step: int = 0
        # internal generator for __next__
        self._iter_gen: Optional[Any] = None

    def __len__(self) -> int:
        return len(self.epoch)

    def render_epoch(
        self, epoch_size: int, seed: Optional[int] = None, num_workers: int = 0,
    ):
        """Sample + render + tokenize *epoch_size* clips and append to buffer.

        Args:
            seed: if not None, seed RNG once at epoch start.
            num_workers: >0 to parallelize render+tokenize via fork.
        """
        # Seed once per epoch — subsequent samples use flowing RNG state
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Sampling must be sequential (RNG order)
        all_notes = [self.note_sampler.sample(seed=None) for _ in range(epoch_size)]
        questions = [random.choice(TRANSCRIPTION_QUESTIONS) for _ in range(epoch_size)]

        # Render + tokenize (parallelizable)
        if num_workers > 0:
            assert self.engine_spec is not None and self.token_fn_kwargs is not None, (
                "engine_spec and token_fn_kwargs must be provided to use num_workers > 0"
            )
            ctx = mp.get_context("spawn")  # spawn avoids CUDA fork-deadlock in parent
            with ctx.Pool(
                num_workers,
                initializer=_spawn_worker_init,
                initargs=(self.engine_spec, self.token_fn_kwargs),
            ) as pool:
                results = pool.map(_render_single, all_notes)
        else:
            results = []
            for note_dicts in all_notes:
                audio = self.render_engine.render_notes(
                    note_dicts, time=self.clip_duration,
                )
                tokens = self.token_fn(note_dicts)
                results.append((audio, tokens))

        # Replace epoch — discard old content (FIFO: buffer is for slow rendering)
        self.epoch = []
        for (audio, tokens), question in zip(results, questions):
            self.epoch.append({
                "dataset_name": "RandomRendering",
                "audio": audio[np.newaxis, :],   # (1, T)
                "question": question,
                "token": tokens,
            })

    def iterate_epoch(self, batch_size: int, repeat: int = 1):
        """Yield batches by shuffling the epoch and iterating *repeat* times.

        Each repeat shuffles independently. Total steps = repeat * (epoch_size // batch_size).
        """
        assert len(self.epoch) > 0, "call render_epoch first"
        indices = list(range(len(self.epoch)))
        for _ in range(repeat):
            random.shuffle(indices)
            for start in range(0, len(indices) - batch_size + 1, batch_size):
                batch_idx = indices[start:start + batch_size]
                yield self._collate([self.epoch[i] for i in batch_idx])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Map-style access into the current epoch (call render_epoch first)."""
        assert len(self.epoch) > 0, "call render_epoch first"
        return self.epoch[idx]

    # ---- iterator protocol: for data in buffer ----
    def _infinite_gen(self):
        """Internal generator: render → shuffle → yield batches → repeat forever."""
        assert self._epoch_size > 0, "set epoch_size > 0 to use iterator mode"
        while True:
            self.render_epoch(
                self._epoch_size,
                seed=self.epoch_counter * self._epoch_size,
                num_workers=self._num_workers,
            )
            self.epoch_counter += 1
            for batch in self.iterate_epoch(self._batch_size, repeat=self._repeat_times):
                self.global_step += 1
                yield batch

    def __iter__(self):
        self._iter_gen = self._infinite_gen()
        return self

    def __next__(self) -> Dict[str, Any]:
        """Return next collated batch. Auto re-renders when epoch is exhausted."""
        if self._iter_gen is None:
            self._iter_gen = self._infinite_gen()
        return next(self._iter_gen)

    @staticmethod
    def _collate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        audio_np = np.stack([item["audio"] for item in items], axis=0)  # (B, 1, T)
        return {
            "dataset_name": [item["dataset_name"] for item in items],
            "audio": torch.from_numpy(audio_np),
            "question": [item["question"] for item in items],
            "token": [item["token"] for item in items],
        }


def notes_to_piano_roll(
    notes: List[Dict],
    fps: float,
    clip_duration: float,
    pitches_num: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert sampler note dicts to binary piano rolls.

    Args:
        notes: list of {"pitch", "start", "dur", "velocity", ...}
        fps: frames per second
        clip_duration: total clip length in seconds
        pitches_num: number of pitch bins (default 128)

    Returns:
        frame_roll:  (clip_frames, pitches_num) float32
        onset_roll:  (clip_frames, pitches_num) float32
        offset_roll: (clip_frames, pitches_num) float32
    """
    clip_frames = round(fps * clip_duration) + 1
    frame_roll = np.zeros((clip_frames, pitches_num), dtype=np.float32)
    onset_roll = np.zeros((clip_frames, pitches_num), dtype=np.float32)
    offset_roll = np.zeros((clip_frames, pitches_num), dtype=np.float32)

    for n in notes:
        onset_time = float(n["start"])
        offset_time = float(n["start"]) + float(n["dur"])
        pitch = int(n["pitch"])

        if not (0 <= pitch < pitches_num):
            continue
        if offset_time <= 0 or onset_time > clip_duration:
            continue

        onset_idx = max(0, min(round(onset_time * fps), clip_frames - 1))
        offset_idx = max(0, min(round(offset_time * fps), clip_frames - 1))

        if onset_time >= 0:
            onset_roll[onset_idx, pitch] = 1
        if offset_time <= clip_duration:
            offset_roll[offset_idx, pitch] = 1

        lo = onset_idx if onset_time >= 0 else 0
        hi = offset_idx if offset_time <= clip_duration else clip_frames - 1
        frame_roll[lo:hi + 1, pitch] = 1

    return frame_roll, onset_roll, offset_roll


# ---- multiprocessing worker for framewise rendering ----
def _spawn_worker_init_framewise(engine_spec: dict):
    """Initializer for spawn pool workers (framewise mode — no token_fn needed)."""
    engine_type = engine_spec["type"]
    if engine_type == "dawdreamer":
        from audio_understanding.random_rendering.rendering import DawDreamerSampler
        engine = DawDreamerSampler(
            time=engine_spec["time"],
            sr=engine_spec["sr"],
            vst_path=engine_spec["vst_path"],
            buffer_size=engine_spec["buffer_size"],
        )
    else:
        assert engine_type == "sampler"
        from audio_understanding.random_rendering.rendering import Sampler
        engine = Sampler(
            time=engine_spec["time"],
            sr=engine_spec["sr"],
            data_dir=engine_spec["data_dir"],
        )
    _mp_ctx["engine"] = engine
    _mp_ctx["clip_duration"] = engine_spec["time"]


def _render_single_framewise(args):
    note_dicts, fps, clip_duration, pitches_num = args
    audio = _mp_ctx["engine"].render_notes(note_dicts, time=clip_duration)
    frame_roll, onset_roll, offset_roll = notes_to_piano_roll(
        note_dicts, fps, clip_duration, pitches_num
    )
    return audio, frame_roll, onset_roll, offset_roll


class FramewiseRenderingBuffer:
    """Buffer-based online rendering for framewise (piano-roll) prediction.

    Same render→shuffle→repeat pattern as OnlineRenderingBuffer, but stores
    piano-roll targets (frame/onset/offset) instead of token sequences.

    Usage:
        buf = FramewiseRenderingBuffer(render_engine, note_sampler, ...)
        for data in buf:   # infinite iterator, auto re-renders each epoch
            audio = data["audio"]           # (B, 1, T)
            frame_roll = data["frame_roll"] # (B, frames, 128)
            ...
        print(buf.global_step, buf.epoch_counter)
    """

    def __init__(
        self,
        render_engine,
        note_sampler,
        clip_duration: float,
        fps: float,
        pitches_num: int = 128,
        epoch_size: int = 256,
        repeat_times: int = 2,
        batch_size: int = 4,
        num_workers: int = 0,
        engine_spec: Optional[dict] = None,
    ):
        self.render_engine = render_engine
        self.note_sampler = note_sampler
        self.clip_duration = float(clip_duration)
        self.fps = float(fps)
        self.pitches_num = int(pitches_num)
        self.epoch_size = epoch_size
        self.repeat_times = repeat_times
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.engine_spec = engine_spec

        self.epoch: List[Dict[str, Any]] = []
        self.epoch_counter: int = 0
        self.global_step: int = 0
        self._iter_gen = None

    def __len__(self) -> int:
        return len(self.epoch)

    def render_epoch(self, epoch_size: int, seed: Optional[int] = None, num_workers: int = 0):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        all_notes = [self.note_sampler.sample(seed=None) for _ in range(epoch_size)]

        if num_workers > 0:
            assert self.engine_spec is not None
            ctx = mp.get_context("spawn")
            args_list = [
                (notes, self.fps, self.clip_duration, self.pitches_num)
                for notes in all_notes
            ]
            with ctx.Pool(
                num_workers,
                initializer=_spawn_worker_init_framewise,
                initargs=(self.engine_spec,),
            ) as pool:
                results = pool.map(_render_single_framewise, args_list)
        else:
            results = []
            for note_dicts in all_notes:
                audio = self.render_engine.render_notes(note_dicts, time=self.clip_duration)
                fr, on, off = notes_to_piano_roll(
                    note_dicts, self.fps, self.clip_duration, self.pitches_num
                )
                results.append((audio, fr, on, off))

        self.epoch = []
        for audio, frame_roll, onset_roll, offset_roll in results:
            self.epoch.append({
                "audio": audio[np.newaxis, :],       # (1, T)
                "frame_roll": frame_roll,             # (frames, pitches_num)
                "onset_roll": onset_roll,
                "offset_roll": offset_roll,
            })

    def iterate_epoch(self, batch_size: int, repeat: int = 1):
        assert len(self.epoch) > 0, "call render_epoch first"
        indices = list(range(len(self.epoch)))
        for _ in range(repeat):
            random.shuffle(indices)
            for start in range(0, len(indices) - batch_size + 1, batch_size):
                batch_idx = indices[start:start + batch_size]
                yield self._collate([self.epoch[i] for i in batch_idx])

    @staticmethod
    def _collate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "audio":       torch.from_numpy(np.stack([it["audio"] for it in items])),       # (B, 1, T)
            "frame_roll":  torch.from_numpy(np.stack([it["frame_roll"] for it in items])),   # (B, frames, 128)
            "onset_roll":  torch.from_numpy(np.stack([it["onset_roll"] for it in items])),
            "offset_roll": torch.from_numpy(np.stack([it["offset_roll"] for it in items])),
        }

    # ---- iterator protocol ----
    def _infinite_gen(self):
        while True:
            self.render_epoch(
                self.epoch_size,
                seed=self.epoch_counter * self.epoch_size,
                num_workers=self.num_workers,
            )
            self.epoch_counter += 1
            for batch in self.iterate_epoch(self.batch_size, repeat=self.repeat_times):
                self.global_step += 1
                yield batch

    def __iter__(self):
        self._iter_gen = self._infinite_gen()
        return self

    def __next__(self) -> Dict[str, Any]:
        if self._iter_gen is None:
            self._iter_gen = self._infinite_gen()
        return next(self._iter_gen)
