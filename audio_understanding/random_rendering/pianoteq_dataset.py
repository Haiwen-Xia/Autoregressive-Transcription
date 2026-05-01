"""PyTorch Dataset that renders audio on-the-fly via Pianoteq CLI.

Flow per __getitem__:
    1. MidiSampler.sample() → note_dicts + symusic Score
    2. Score → temp .mid file
    3. pianoteq --midi X.mid --wav X.wav [--set-param ...] → temp .wav
    4. Load .wav → torch tensor
    5. notes → token list (via notes_to_tokens or MIDI2Tokens)

Usage:
    dataset = PianoteqDataset(note_sampler, clip_duration=5.0, sr=16000, ...)
    audio, tokens = dataset[0]
"""
from __future__ import annotations

import os
import gc
import signal
import subprocess
import tempfile
import time
import ctypes
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


def _default_token_fn(notes):
    """Default token function: picklable for DataLoader spawn workers."""
    from audio_understanding.random_rendering.dataset import notes_to_tokens
    return notes_to_tokens(notes, fps=100.0, include_program=True)

from audio_understanding.random_rendering.midi_sampler import MidiSampler, note_dicts_to_score

# ---- Pianoteq randomizable parameters ----
# Priority: non-pedal, non-mic, non-locked, continuous-valued
# Each entry: (param_name, min_val, max_val)
PIANOTEQ_RANDOM_PARAMS = [
    ("Condition", 0.0, 10.0),
    ("Dynamics", 1.0, 100.0),
    ("Velocity Offset", -1.0, 1.0),
    ("Post Effect Gain", -12.0, 12.0),
    ("Stereo Width", 0.0, 5.0),
    ("Sound Speed", 200.0, 500.0),
    ("Damper Noise", 0.0, 24.0),
    ("Key Release Noise", 0.0, 25.0),
    ("Attack Envelope", 0.0, 10.0),
    ("Virtuosity", 0.0, 1.0),
    ("Reverb Duration", 0.0, 5.0),
    ("Reverb Mix", 0.0, 50.0),
    ("Room Dimensions", 5.0, 50.0),
    ("Reverb Pre-delay", 0.0, 0.2),
    ("Reverb Early Reflections", -20.0, 20.0),
    ("Reverb Tone", -1.0, 1.0),
]


def sample_pianoteq_params(
    rng: np.random.RandomState,
    param_specs: List[Tuple[str, float, float]] = PIANOTEQ_RANDOM_PARAMS,
    n_params: int = 4,
) -> Dict[str, float]:
    """Randomly select n_params and sample values uniformly within range."""
    indices = rng.choice(len(param_specs), size=min(n_params, len(param_specs)), replace=False)
    params = {}
    for i in indices:
        name, lo, hi = param_specs[i]
        params[name] = round(float(rng.uniform(lo, hi)), 4)
    return params


def render_pianoteq(
    midi_path: str,
    wav_path: str,
    params: Optional[Dict[str, float]] = None,
    preset: Optional[str] = None,
    pianoteq_bin: str = "pianoteq",
    sr: int = 16000,
    timeout_sec: int = 120,
    retry: int = 2,
    retry_wait_sec: float = 1.0,
    hard_reset_on_timeout: bool = False,
) -> None:
    """Call pianoteq CLI to render a MIDI file to WAV with retry and timeout recovery."""
    cmd = [pianoteq_bin, "--midi", midi_path, "--wav", wav_path]
    if sr != 44100:
        cmd.extend(["--rate", str(sr)])
    if preset:
        cmd.extend(["--preset", preset])
    if params:
        for k, v in params.items():
            cmd.extend(["--set-param", f"{k}={v}"])

    assert retry >= 0, f"retry must be >= 0, got {retry}"
    assert timeout_sec > 0, f"timeout_sec must be > 0, got {timeout_sec}"

    def _hard_reset() -> None:
        gc.collect()
        # Best-effort allocator trim on glibc; harmless no-op elsewhere.
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    max_attempts = retry + 1
    last_err = ""
    for attempt in range(1, max_attempts + 1):
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            _stdout, _stderr = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _stdout, _stderr = proc.communicate()
            last_err = f"pianoteq timeout after {timeout_sec}s (attempt {attempt}/{max_attempts}): {_stderr}"
            if hard_reset_on_timeout:
                _hard_reset()
            if attempt < max_attempts:
                time.sleep(retry_wait_sec * attempt)
                continue
            assert False, last_err

        if proc.returncode == 0:
            return

        last_err = f"pianoteq failed with code {proc.returncode} (attempt {attempt}/{max_attempts}): {_stderr}"
        if hard_reset_on_timeout:
            _hard_reset()
        if attempt < max_attempts:
            time.sleep(retry_wait_sec * attempt)
            continue

    assert False, last_err


class PianoteqDataset(Dataset):
    """On-the-fly rendering dataset using Pianoteq CLI.

    Args:
        note_sampler: BaseSampler instance (MarginalRandomSampler, UniformSampler, etc.)
        clip_duration: seconds per clip
        sr: target sample rate
        output_mode: "tokens" → returns audio + token list;
                     "piano_roll" → returns audio + frame/onset/offset roll tensors (for framewise training)
        fps: frames-per-second for piano roll (only used when output_mode="piano_roll")
        token_fn: callable(note_dicts) -> List[str]; if None, uses notes_to_tokens
        epoch_size: virtual dataset length
        cc_config: CC generator config for MidiSampler (e.g. {"sustain": {}})
        random_params: whether to randomize pianoteq rendering params
        n_random_params: how many params to randomize per sample
        param_specs: list of (name, min, max) tuples; defaults to PIANOTEQ_RANDOM_PARAMS
        preset: fixed preset name; if None, randomly selects per group
        preset_programs: list of MIDI programs to sample presets from (default: [0] = piano only)
        preset_per_sample: number of preset groups inside one sample.
            1 => one unified MIDI render (fastest).
            >1 => split notes into that many groups and render groups in parallel, then mix.
        pianoteq_bin: path to pianoteq binary
        tmp_dir: directory for temp files (defaults to system temp)
        seed_offset: base seed offset for reproducibility
    """
    def __init__(
        self,
        note_sampler,
        clip_duration: float,
        sr: int = 16000,
        output_mode: str = "tokens",
        fps: float = 100.0,
        token_fn: Optional[Callable] = None,
        epoch_size: int = 256,
        cc_config: Optional[Dict] = None,
        random_params: bool = True,
        n_random_params: int = 4,
        param_specs: Optional[List[Tuple[str, float, float]]] = None,
        preset: Optional[str] = None,
        preset_programs: Optional[List[int]] = None,
        preset_per_sample: int = 1,
        pianoteq_bin: str = "pianoteq",
        render_timeout_sec: int = 120,
        render_retry: int = 2,
        render_retry_wait_sec: float = 1.0,
        render_hard_reset_on_timeout: bool = False,
        tmp_dir: Optional[str] = None,
        seed_offset: int = 0,
    ):
        assert output_mode in ("tokens", "piano_roll"), f"Unknown output_mode: {output_mode}"
        self.midi_sampler = MidiSampler(note_sampler, clip_duration, cc_config)
        self.clip_duration = clip_duration
        self.sr = sr
        self.epoch_size = epoch_size
        self.random_params = random_params
        self.n_random_params = n_random_params
        self.param_specs = param_specs or PIANOTEQ_RANDOM_PARAMS
        self.preset = preset
        self.preset_programs = preset_programs if preset_programs is not None else [0]
        assert len(self.preset_programs) > 0, "preset_programs must not be empty"
        assert preset_per_sample >= 1, f"preset_per_sample must be >= 1, got {preset_per_sample}"
        self.preset_per_sample = int(preset_per_sample)
        self.pianoteq_bin = pianoteq_bin
        self.render_timeout_sec = int(render_timeout_sec)
        self.render_retry = int(render_retry)
        self.render_retry_wait_sec = float(render_retry_wait_sec)
        self.render_hard_reset_on_timeout = bool(render_hard_reset_on_timeout)
        self.tmp_dir = tmp_dir
        self.seed_offset = seed_offset
        self.output_mode = output_mode
        self.fps = fps

        self.token_fn = token_fn if token_fn is not None else _default_token_fn

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # no fixed seed: each call generates fresh random data across epochs
        rng = np.random.RandomState()

        # 1. sample notes + CC
        score, note_dicts = self.midi_sampler.sample(seed=None)

        try:

            # 2. sample rendering params
            params = None
            if self.random_params:
                params = sample_pianoteq_params(rng, self.param_specs, self.n_random_params)

            # 2b + 3 + 4. grouped preset rendering
            target_len = int(self.clip_duration * self.sr)
            group_count = min(self.preset_per_sample, max(1, len(note_dicts)))
            grouped_notes: List[List[Dict[str, Any]]] = [[] for _ in range(group_count)]
            for i, n in enumerate(note_dicts):
                grouped_notes[i % group_count].append(n)

            from audio_understanding.random_rendering.pianoteq_presets import get_random_preset #* preset 很多，实际解决方式是rule-based+迭代

            group_specs: List[Tuple[List[Dict[str, Any]], str, int]] = []
            for notes_group in grouped_notes:
                if self.preset:
                    group_preset = self.preset
                    group_program = int(rng.choice(self.preset_programs))
                else:
                    group_preset, group_program = get_random_preset(rng, self.preset_programs)
                for n in notes_group:
                    n["program"] = group_program
                group_specs.append((notes_group, group_preset, group_program))

            def _render_group(notes_group: List[Dict[str, Any]], group_program: int, group_preset: str) -> np.ndarray:
                with tempfile.NamedTemporaryFile(suffix=".mid", dir=self.tmp_dir, delete=False) as f_mid:
                    group_mid_path = f_mid.name
                with tempfile.NamedTemporaryFile(suffix=".wav", dir=self.tmp_dir, delete=False) as f_wav:
                    group_wav_path = f_wav.name
                try:
                    if group_count == 1:
                        # Keep sampled CC events in unified mode.
                        for track in score.tracks:
                            track.program = group_program
                        score.dump_midi(group_mid_path)
                    else:
                        group_score = note_dicts_to_score(notes_group, self.clip_duration, cc_events=None)
                        group_score.dump_midi(group_mid_path)

                    render_pianoteq(
                        group_mid_path,
                        group_wav_path,
                        params,
                        group_preset,
                        self.pianoteq_bin,
                        self.sr,
                        timeout_sec=self.render_timeout_sec,
                        retry=self.render_retry,
                        retry_wait_sec=self.render_retry_wait_sec,
                        hard_reset_on_timeout=self.render_hard_reset_on_timeout,
                    )
                    group_audio, group_sr = sf.read(group_wav_path, dtype="float32")
                    assert group_sr == self.sr, f"Expected sr={self.sr}, got {group_sr}"
                    if group_audio.ndim == 2:
                        group_audio = group_audio.mean(axis=1)
                    if len(group_audio) < target_len:
                        group_audio = np.pad(group_audio, (0, target_len - len(group_audio)))
                    else:
                        group_audio = group_audio[:target_len]
                    return group_audio.astype(np.float32)
                finally:
                    os.unlink(group_mid_path)
                    if os.path.exists(group_wav_path):
                        os.unlink(group_wav_path)

            if group_count == 1:
                spec = group_specs[0]
                audio = _render_group(spec[0], spec[2], spec[1])
                preset = spec[1]
            else:
                max_workers = min(group_count, os.cpu_count() or group_count)
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    rendered_groups = list(ex.map(lambda s: _render_group(s[0], s[2], s[1]), group_specs))
                mixed = np.zeros(target_len, dtype=np.float32)
                for group_audio in rendered_groups:
                    mixed += group_audio
                peak = float(np.max(np.abs(mixed))) + 1e-9
                if peak > 1.0:
                    mixed = 0.95 * mixed / peak
                audio = mixed
                preset = [s[1] for s in group_specs]

            # pad or trim to clip_duration
            target_len = int(self.clip_duration * self.sr)
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]

            audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, T)

            # 5. output
            if self.output_mode == "piano_roll":
                from audio_understanding.random_rendering.dataset import notes_to_piano_roll
                frame_roll, onset_roll, offset_roll = notes_to_piano_roll(
                    note_dicts, fps=self.fps, clip_duration=self.clip_duration
                )
                return {
                    "audio": audio_tensor,                          # (1, T)
                    "frame_roll": torch.from_numpy(frame_roll),    # (frames, 128)
                    "onset_roll": torch.from_numpy(onset_roll),    # (frames, 128)
                    "offset_roll": torch.from_numpy(offset_roll),  # (frames, 128)
                    # preset/params intentionally excluded: vary per sample, can't default_collate
                }
            else:  # tokens
                tokens = self.token_fn(note_dicts)
                return {
                    "audio": audio_tensor,
                    "tokens": tokens,
                    "note_dicts": note_dicts,
                    "params": params or {},
                    "preset": preset,
                }
        finally:
            pass


def main_func(
    note_sampler,
    clip_duration: float = 5.0,
    sr: int = 16000,
    epoch_size: int = 256,
    cc_config: Optional[Dict] = None,
    random_params: bool = True,
    n_random_params: int = 4,
    preset_per_sample: int = 1,
    pianoteq_bin: str = "pianoteq",
    **kwargs,
) -> PianoteqDataset:
    """Convenience entry point for creating a PianoteqDataset."""
    return PianoteqDataset(
        note_sampler=note_sampler,
        clip_duration=clip_duration,
        sr=sr,
        epoch_size=epoch_size,
        cc_config=cc_config,
        random_params=random_params,
        n_random_params=n_random_params,
        preset_per_sample=preset_per_sample,
        pianoteq_bin=pianoteq_bin,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse
    from audio_understanding.random_rendering.sampler import MarginalRandomSampler

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="/data/yrb/musicarena/Haiwen/Autoregressive-Transcription/Rendering_assets/piano_stats_5s.json")
    parser.add_argument("--clip_duration", type=float, default=5.0)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--epoch_size", type=int, default=4)
    parser.add_argument("--dedup", type=str, default="FIFO")
    parser.add_argument("--pianoteq_bin", type=str, default="pianoteq")
    args = parser.parse_args()

    note_sampler = MarginalRandomSampler(
        time=args.clip_duration,
        config_json=args.config,
        deduplication=args.dedup,
    )

    dataset = main_func(
        note_sampler=note_sampler,
        clip_duration=args.clip_duration,
        sr=args.sr,
        epoch_size=args.epoch_size,
        cc_config={"sustain": {"n_presses_range": (0, 3)}},
        random_params=True,
        n_random_params=4,
        pianoteq_bin=args.pianoteq_bin,
    )

    print(f"Dataset size: {len(dataset)}")
    import soundfile
    output_dir = '/data/yrb/musicarena/Haiwen/Autoregressive-Transcription/debug_rendering/pianoteq'
    os.makedirs(output_dir, exist_ok=True)
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\n--- Sample {i} ---")
        print(f"  audio shape: {sample['audio'].shape}")
        print(f"  tokens ({len(sample['tokens'])}): {sample['tokens'][:10]}...")
        print(f"  notes: {len(sample['note_dicts'])}")
        print(f"  preset: {sample['preset']}")
        print(f"  params: {sample['params']}")
        soundfile.write(f"{output_dir}/sample_{i}.wav", sample["audio"].squeeze(0).numpy(), args.sr)
