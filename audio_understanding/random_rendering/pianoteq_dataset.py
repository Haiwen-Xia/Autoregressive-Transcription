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
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


def _default_token_fn(notes):
    """Default token function: picklable for DataLoader spawn workers."""
    from audio_understanding.random_rendering.dataset import notes_to_tokens
    return notes_to_tokens(notes, fps=100.0, include_program=True)

from audio_understanding.random_rendering.midi_sampler import MidiSampler

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
) -> None:
    """Call pianoteq CLI to render a MIDI file to WAV."""
    cmd = [pianoteq_bin, "--midi", midi_path, "--wav", wav_path]
    if sr != 44100:
        cmd.extend(["--rate", str(sr)])
    if preset:
        cmd.extend(["--preset", preset])
    if params:
        for k, v in params.items():
            cmd.extend(["--set-param", f"{k}={v}"])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"pianoteq failed: {result.stderr}"


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
        preset: fixed preset name; if None, randomly selects per sample
        preset_programs: list of MIDI programs to sample presets from (default: [0] = piano only)
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
        pianoteq_bin: str = "pianoteq",
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
        self.pianoteq_bin = pianoteq_bin
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

        # 1. sample notes + CC → MIDI
        with tempfile.NamedTemporaryFile(suffix=".mid", dir=self.tmp_dir, delete=False) as f_mid:
            mid_path = f_mid.name
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=self.tmp_dir, delete=False) as f_wav:
            wav_path = f_wav.name

        try:
            note_dicts = self.midi_sampler.sample_and_save(mid_path, seed=None)

            # 2. sample rendering params
            params = None
            if self.random_params:
                params = sample_pianoteq_params(rng, self.param_specs, self.n_random_params)

            # 2b. select preset
            if self.preset:
                preset = self.preset
            else:
                from audio_understanding.random_rendering.pianoteq_presets import get_random_preset #* preset 很多，实际解决方式是rule-based+迭代
                preset, _ = get_random_preset(rng, self.preset_programs)

            # 3. render via pianoteq
            render_pianoteq(mid_path, wav_path, params, preset, self.pianoteq_bin, self.sr)

            # 4. load audio
            audio, file_sr = sf.read(wav_path, dtype="float32")
            assert file_sr == self.sr, f"Expected sr={self.sr}, got {file_sr}"
            # stereo → mono
            if audio.ndim == 2:
                audio = audio.mean(axis=1)

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
            os.unlink(mid_path)
            if os.path.exists(wav_path):
                os.unlink(wav_path)


def main_func(
    note_sampler,
    clip_duration: float = 5.0,
    sr: int = 16000,
    epoch_size: int = 256,
    cc_config: Optional[Dict] = None,
    random_params: bool = True,
    n_random_params: int = 4,
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
