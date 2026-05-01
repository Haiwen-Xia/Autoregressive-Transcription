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
from torch.utils.data import Dataset

from audio_understanding.random_rendering.rendering import Sampler as RenderEngine
from audio_understanding.random_rendering.sampler import MarginalRandomSampler
from audio_understanding.random_rendering.dsp_render import pianoish_fast, pianoish_physics, pianoish_fixed, midi_to_hz
from audio_understanding.random_rendering.dsp_gp import generate_music_note, random_sample_params, precompute_gp_cholesky

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


def collate_random_token_batch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for token rendering datasets.

    Keeps question/token as Python lists because each token sequence has variable length.
    """
    audio_np = np.stack([item["audio"] for item in items], axis=0)  # (B, 1, T)
    return {
        "dataset_name": [item["dataset_name"] for item in items],
        "audio": torch.from_numpy(audio_np),
        "question": [item["question"] for item in items],
        "token": [item["token"] for item in items],
    }


def collate_framewise_batch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for framewise rendering datasets."""
    batch = {
        "audio": torch.from_numpy(np.stack([it["audio"] for it in items])),
        "frame_roll": torch.from_numpy(np.stack([it["frame_roll"] for it in items])),
        "onset_roll": torch.from_numpy(np.stack([it["onset_roll"] for it in items])),
        "offset_roll": torch.from_numpy(np.stack([it["offset_roll"] for it in items])),
    }
    if "note_dicts" in items[0]:
        batch["note_dicts"] = [it["note_dicts"] for it in items]
    return batch


class BaseRenderingDataset(Dataset):
    """Unified base class for on-the-fly rendering datasets.

    Subclasses should implement __getitem__ and put all rendering logic there.
    This makes different render backends (sample-based, VST, external program)
    share the same dataset access interface.
    """

    def __init__(self, clip_duration: float, epoch_size: int):
        self.clip_duration = float(clip_duration)
        self.epoch_size = int(epoch_size)

    def __len__(self) -> int:
        return self.epoch_size


class RandomTokenRenderingDataset(BaseRenderingDataset):
    """On-the-fly random rendering dataset for autoregressive token training."""

    def __init__(
        self,
        render_engine,
        note_sampler,
        clip_duration: float,
        token_fn: Callable[[List[Dict]], List[str]],
        epoch_size: int,
    ):
        super().__init__(clip_duration=clip_duration, epoch_size=epoch_size)
        self.render_engine = render_engine
        self.note_sampler = note_sampler
        self.token_fn = token_fn

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        del idx
        note_dicts = self.note_sampler.sample(seed=None)
        audio = self.render_engine.render_notes(note_dicts, time=self.clip_duration)
        tokens = self.token_fn(note_dicts)
        return {
            "dataset_name": "RandomRendering",
            "audio": audio[np.newaxis, :],
            "question": random.choice(TRANSCRIPTION_QUESTIONS),
            "token": tokens,
        }


class RandomFramewiseRenderingDataset(BaseRenderingDataset):
    """On-the-fly random rendering dataset for frame/onset/offset prediction."""

    def __init__(
        self,
        render_engine,
        note_sampler,
        clip_duration: float,
        fps: float,
        epoch_size: int,
        pitches_num: int = 128,
    ):
        super().__init__(clip_duration=clip_duration, epoch_size=epoch_size)
        self.render_engine = render_engine
        self.note_sampler = note_sampler
        self.fps = float(fps)
        self.pitches_num = int(pitches_num)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        del idx
        note_dicts = self.note_sampler.sample(seed=None)
        audio = self.render_engine.render_notes(note_dicts, time=self.clip_duration)
        frame_roll, onset_roll, offset_roll = notes_to_piano_roll(
            note_dicts,
            fps=self.fps,
            clip_duration=self.clip_duration,
            pitches_num=self.pitches_num,
        )
        return {
            "audio": audio[np.newaxis, :],
            "frame_roll": frame_roll,
            "onset_roll": onset_roll,
            "offset_roll": offset_roll,
            "note_dicts": note_dicts,
        }


class DSPFramewiseRenderingDataset(BaseRenderingDataset):
    """On-the-fly framewise dataset using procedural DSP piano-like rendering."""

    def __init__(
        self,
        note_sampler,
        clip_duration: float,
        fps: float,
        sr: int,
        epoch_size: int,
        pitches_num: int = 128,
        dsp_variant: str = "fast",  # "fast" | "physics" | "fixed"
    ):
        super().__init__(clip_duration=clip_duration, epoch_size=epoch_size)
        self.note_sampler = note_sampler
        self.fps = float(fps)
        self.sr = int(sr)
        self.pitches_num = int(pitches_num)
        self.lowpass_max = 8000.0  # max lowpass cutoff for pianoish_fast
        assert dsp_variant in ("fast", "physics", "fixed"), f"Unknown dsp_variant: {dsp_variant}"
        self.dsp_variant = dsp_variant

    def _render_notes_dsp(self, note_dicts: List[Dict[str, Any]]) -> np.ndarray:
        total_samples = int(round(self.clip_duration * self.sr))
        out = np.zeros(total_samples, dtype=np.float32)
        rng = np.random.default_rng()

        for n in note_dicts:
            start = float(n["start"])
            dur = float(n["dur"])
            pitch = float(n["pitch"])
            velocity = float(n["velocity"]) / 127.0

            if dur <= 0.0:
                continue
            if start >= self.clip_duration or start + dur <= 0.0:
                continue

            # Keep synthesis stable for very short sampled notes.
            dur = max(dur, 0.01)

            f0 = midi_to_hz(pitch)
            lowpass_min = max(1.25 * f0, 1200.0)
            lowpass_max = self.lowpass_max

            note_seed = int(rng.integers(0, 2**31 - 1))
            if self.dsp_variant == "fast":
                note_audio = pianoish_fast(
                    duration=dur,
                    pitch=pitch,
                    velocity=velocity,
                    sr=self.sr,
                    seed=note_seed,
                    use_lowpass=True,
                    lowpass_range=(lowpass_min, lowpass_max),
                    core_mode="harmonic",
                )
            elif self.dsp_variant == "physics":
                note_audio = pianoish_physics(
                    duration=dur,
                    pitch=pitch,
                    velocity=velocity,
                    sr=self.sr,
                    seed=note_seed,
                    use_lowpass=True,
                    lowpass_range=(lowpass_min, lowpass_max),
                )
            else:  # fixed
                note_audio = pianoish_fixed(
                    duration=dur,
                    pitch=pitch,
                    velocity=velocity,
                    sr=self.sr,
                    seed=note_seed,
                )

            write_start = int(round(start * self.sr))
            src_start = 0
            if write_start < 0:
                src_start = -write_start
                write_start = 0

            if write_start >= total_samples or src_start >= len(note_audio):
                continue

            write_end = min(total_samples, write_start + (len(note_audio) - src_start))
            if write_end <= write_start:
                continue

            seg_len = write_end - write_start
            out[write_start:write_end] += note_audio[src_start:src_start + seg_len]

        peak = float(np.max(np.abs(out))) + 1e-9
        if peak > 1.0:
            out = 0.95 * out / peak

        return out.astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        del idx
        note_dicts = self.note_sampler.sample(seed=None)
        audio = self._render_notes_dsp(note_dicts)
        frame_roll, onset_roll, offset_roll = notes_to_piano_roll(
            note_dicts,
            fps=self.fps,
            clip_duration=self.clip_duration,
            pitches_num=self.pitches_num,
        )
        return {
            "audio": audio[np.newaxis, :],
            "frame_roll": frame_roll,
            "onset_roll": onset_roll,
            "offset_roll": offset_roll,
            "note_dicts": note_dicts,
        }


class GPRenderDataset(BaseRenderingDataset):
    """On-the-fly framewise dataset using GP-based additive synthesis (dsp_gp).

    Each note keeps independent timbre sampling for diversity.
    All notes in a clip share the same mix buffer.
    """

    def __init__(
        self,
        note_sampler,
        clip_duration: float,
        fps: float,
        sr: int,
        epoch_size: int,
        pitches_num: int = 128,
    ):
        super().__init__(clip_duration=clip_duration, epoch_size=epoch_size)
        self.note_sampler = note_sampler
        self.fps = float(fps)
        self.sr = int(sr)
        self.pitches_num = int(pitches_num)

    def _render_notes_gp(self, note_dicts: List[Dict[str, Any]]) -> np.ndarray:
        """Render a list of note dicts using generate_music_note, mix into clip.

        GP 核协方差（Exponential + StdPeriodic）每个 clip 预先计算一次 (O(n^3))，
        同一 clip 内所有 note 共用 Cholesky 因子，采样时只需矩阵向量乘 (O(n^2))。
        其他 timbre 参数（n_overtones、rolloff、decay 等）每个 note 仍独立随机。
        """
        total_samples = int(round(self.clip_duration * self.sr))
        out = np.zeros(total_samples, dtype=np.float32)
        global_rng = np.random.default_rng()

        # 每个 clip 采样一组共享的 GP 核参数，预计算 Cholesky
        n_frames = 100
        gp_variance       = float(global_rng.uniform(0.5, 2.5))
        gp_lengthscale    = float(global_rng.uniform(0.15, 0.6))
        vibrato_variance  = float(global_rng.uniform(0.5, 2.0))
        vibrato_period    = float(global_rng.uniform(0.08, 0.3))
        vibrato_lengthscale = float(global_rng.uniform(0.05, 0.2))
        L_gp_exp, L_gp_vibrato = precompute_gp_cholesky(
            n_frames, gp_variance, gp_lengthscale,
            vibrato_variance, vibrato_period, vibrato_lengthscale,
        )

        for n in note_dicts:
            start = float(n["start"])
            dur = float(n["dur"])
            pitch = float(n["pitch"])
            # sampler velocity is 0-127 int; normalize to 0-1
            velocity = float(n["velocity"]) / 127.0

            if dur <= 0.0 or start >= self.clip_duration or start + dur <= 0.0:
                continue

            note_seed = int(global_rng.integers(0, 2**31 - 1))
            # 每个 note 独立采样 timbre 参数，但覆盖 GP 核参数以匹配预计算的 L
            note_timbre_params = random_sample_params(np.random.default_rng(note_seed))
            note_timbre_params["n_frames"]           = n_frames
            note_timbre_params["gp_variance"]        = gp_variance
            note_timbre_params["gp_lengthscale"]     = gp_lengthscale
            note_timbre_params["vibrato_variance"]   = vibrato_variance
            note_timbre_params["vibrato_period"]     = vibrato_period
            note_timbre_params["vibrato_lengthscale"] = vibrato_lengthscale

            note_audio, _ = generate_music_note(
                pitch=pitch,
                duration=max(dur, 0.02),
                velocity=velocity,
                sr=self.sr,
                seed=note_seed,
                L_gp_exp=L_gp_exp,
                L_gp_vibrato=L_gp_vibrato,
                **note_timbre_params,
            )

            write_start = int(round(start * self.sr))
            src_start = 0
            if write_start < 0:
                src_start = -write_start
                write_start = 0

            if write_start >= total_samples or src_start >= len(note_audio):
                continue

            write_end = min(total_samples, write_start + (len(note_audio) - src_start))
            seg_len = write_end - write_start
            out[write_start:write_end] += note_audio[src_start:src_start + seg_len].astype(np.float32)

        peak = float(np.max(np.abs(out))) + 1e-9
        if peak > 1.0:
            out = 0.95 * out / peak
        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        del idx
        note_dicts = self.note_sampler.sample(seed=None)
        audio = self._render_notes_gp(note_dicts)
        frame_roll, onset_roll, offset_roll = notes_to_piano_roll(
            note_dicts,
            fps=self.fps,
            clip_duration=self.clip_duration,
            pitches_num=self.pitches_num,
        )
        return {
            "audio": audio[np.newaxis, :],
            "frame_roll": frame_roll,
            "onset_roll": onset_roll,
            "offset_roll": offset_roll,
            "note_dicts": note_dicts,
        }


DSPGPFramewiseRenderingDataset = GPRenderDataset  # backward compat alias


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


# ---- helpers for MaestroRenderingBuffer ----

def _load_maestro_metadata(maestro_root: str, split: str = "train") -> List[Dict]:
    """Load MAESTRO CSV and return list of {midi_path, duration} for given split."""
    import csv
    from pathlib import Path
    root = Path(maestro_root)
    csv_path = root / "maestro-v3.0.0.csv"
    assert csv_path.exists(), f"MAESTRO CSV not found: {csv_path}"
    pieces = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != split:
                continue
            pieces.append({
                "midi_path": str(root / row["midi_filename"]),
                "duration": float(row["duration"]),
            })
    return pieces


def _clip_midi_to_note_dicts(
    midi_path: str,
    start_time: float,
    clip_duration: float,
) -> List[Dict]:
    """Read MIDI with symusic, clip to [start_time, start_time+clip_duration], return note dicts."""
    from symusic import Score
    score: Any = Score(midi_path, ttype="second")
    from audio_understanding.utils_midi_symusic import clip_symusic_notes

    note_dicts = []
    for track in score.tracks:
        is_drum = bool(getattr(track, "is_drum", False))
        program = 128 if is_drum else int(track.program)

        clipped_notes, _ = clip_symusic_notes(
            notes=list(track.notes),
            start_time=start_time,
            duration=clip_duration,
            mode="clip",
        )
        for note in clipped_notes:
            note_dicts.append({
                "start": float(note.time) - start_time,
                "dur": float(note.duration),
                "pitch": int(note.pitch),
                "velocity": int(note.velocity),
                "program": program,
            })
    return note_dicts


# module-level worker for MaestroRenderingBuffer spawn pool
def _render_single_maestro(args):
    """Worker: clip MIDI → render → tokenize. Runs in spawn pool."""
    midi_path, start_time, clip_duration = args
    note_dicts = _clip_midi_to_note_dicts(midi_path, start_time, clip_duration)
    audio = _mp_ctx["engine"].render_notes(note_dicts, time=clip_duration)
    tokens = _mp_ctx["token_fn"](note_dicts)
    return audio, tokens


class MaestroRenderingBuffer:
    """Online rendering buffer that clips real MIDI from MAESTRO and renders with VST.

    Same interface as OnlineRenderingBuffer (render_epoch + iterate_epoch),
    so training loops can be swapped with zero code changes.
    """

    def __init__(
        self,
        maestro_root: str,
        render_engine: "RenderEngine",
        clip_duration: float,
        token_fn: Callable[[List[Dict]], List[str]],
        engine_spec: Optional[dict] = None,
        token_fn_kwargs: Optional[dict] = None,
        split: str = "train",
    ):
        self.render_engine = render_engine
        self.clip_duration = clip_duration
        self.token_fn = token_fn
        self.engine_spec = engine_spec
        self.token_fn_kwargs = token_fn_kwargs
        self.epoch: List[Dict[str, Any]] = []

        self.pieces = _load_maestro_metadata(maestro_root, split=split)
        # filter out pieces shorter than clip_duration
        orig_len = len(self.pieces)
        self.pieces = [p for p in self.pieces if p["duration"] >= clip_duration]
        if len(self.pieces) < orig_len:
            import logging
            logging.getLogger(__name__).info(
                "MaestroRenderingBuffer: dropped %d pieces shorter than %.1fs, %d remain",
                orig_len - len(self.pieces), clip_duration, len(self.pieces),
            )
        assert len(self.pieces) > 0, "No MAESTRO pieces long enough for clip_duration"

    def __len__(self) -> int:
        return len(self.epoch)

    def render_epoch(
        self, epoch_size: int, seed: Optional[int] = None, num_workers: int = 0,
    ):
        """Sample piece indices → random clip → render → tokenize."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # sample piece indices (with replacement)
        piece_indices = [random.randrange(len(self.pieces)) for _ in range(epoch_size)]
        questions = [random.choice(TRANSCRIPTION_QUESTIONS) for _ in range(epoch_size)]

        # compute (midi_path, start_time) for each sample
        render_args = []
        for idx in piece_indices:
            p = self.pieces[idx]
            max_start = p["duration"] - self.clip_duration
            start_time = random.uniform(0, max_start)
            start_time = round(start_time / 0.01) * 0.01  # snap to 10ms grid
            render_args.append((p["midi_path"], start_time, self.clip_duration))

        # render + tokenize
        if num_workers > 0:
            assert self.engine_spec is not None and self.token_fn_kwargs is not None
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                num_workers,
                initializer=_spawn_worker_init,
                initargs=(self.engine_spec, self.token_fn_kwargs),
            ) as pool:
                results = pool.map(_render_single_maestro, render_args)
        else:
            results = []
            for midi_path, start_time, dur in render_args:
                note_dicts = _clip_midi_to_note_dicts(midi_path, start_time, dur)
                audio = self.render_engine.render_notes(note_dicts, time=dur)
                tokens = self.token_fn(note_dicts)
                results.append((audio, tokens))

        self.epoch = []
        for (audio, tokens), question in zip(results, questions):
            self.epoch.append({
                "dataset_name": "MaestroRendering",
                "audio": audio[np.newaxis, :],   # (1, T)
                "question": question,
                "token": tokens,
            })

    def iterate_epoch(self, batch_size: int, repeat: int = 1):
        """Yield batches — identical to OnlineRenderingBuffer.iterate_epoch."""
        assert len(self.epoch) > 0, "call render_epoch first"
        indices = list(range(len(self.epoch)))
        for _ in range(repeat):
            random.shuffle(indices)
            for start in range(0, len(indices) - batch_size + 1, batch_size):
                batch_idx = indices[start:start + batch_size]
                yield OnlineRenderingBuffer._collate([self.epoch[i] for i in batch_idx])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        assert len(self.epoch) > 0, "call render_epoch first"
        return self.epoch[idx]


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
