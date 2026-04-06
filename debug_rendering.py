"""Debug script: render N samples and save audio+MIDI under a debug_rendering/ folder.

Usage:
    python debug_rendering.py [overrides]   # uses configs/random_rendering/random_rendering.yaml
    python debug_rendering.py rendering.rendering_engine=sampler n_samples=5
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import pretty_midi
import soundfile as sf
from omegaconf import DictConfig, OmegaConf

from audio_understanding.random_rendering.rendering import DawDreamerSampler, Sampler as RenderEngine
from audio_understanding.random_rendering.sampler import MarginalRandomSampler, UniformSampler


def notes_to_midi(notes: list[dict], clip_duration: float) -> pretty_midi.PrettyMIDI:
    """Convert sampler note dicts to a PrettyMIDI object."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    # group by program
    by_program: dict[int, pretty_midi.Instrument] = {}
    for n in notes:
        prog = int(n["program"])
        if prog not in by_program:
            by_program[prog] = pretty_midi.Instrument(program=prog)
        note = pretty_midi.Note(
            velocity=int(n["velocity"]),
            pitch=int(n["pitch"]),
            start=float(n["start"]),
            end=min(float(n["start"]) + float(n["dur"]), clip_duration),
        )
        by_program[prog].notes.append(note)
    for inst in by_program.values():
        pm.instruments.append(inst)
    return pm


def main_func(cfg: DictConfig, n_samples: int = 5) -> None:
    configs = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))

    out_dir = Path('debug_rendering') / configs.get("out_dir", "debug_rendering")
    out_dir.mkdir(exist_ok=True)

    rendering_cfg = configs["rendering"]
    sr = configs["sample_rate"]
    clip_duration = configs["clip_duration"]
    rendering_engine_type = rendering_cfg.get("rendering_engine", "sampler")

    print(f"rendering_engine={rendering_engine_type}  sr={sr}  clip_duration={clip_duration}s")
    print(f"Saving {n_samples} samples to {out_dir.resolve()}/")

    # ---- build render engine ----
    if rendering_engine_type == "dawdreamer":
        render_engine = DawDreamerSampler(
            time=clip_duration,
            sr=sr,
            vst_path=rendering_cfg["vst_name"],
            buffer_size=rendering_cfg.get("buffer_size", 512),
        )
    else:
        assert rendering_engine_type == "sampler"
        render_engine = RenderEngine(
            time=clip_duration,
            sr=sr,
            data_dir=rendering_cfg["data_dir"],
        )

    # ---- build note sampler ----
    sampler_type = rendering_cfg.get("sampler_type", "marginal")
    deduplication = rendering_cfg.get("deduplication", "disabled")
    if sampler_type == "marginal":
        note_sampler = MarginalRandomSampler(
            time=clip_duration,
            config_json=rendering_cfg["sampler_config"],
            deduplication=deduplication,
        )
    elif sampler_type == "uniform":
        from audio_understanding.random_rendering.sampler import UniformSampler
        note_sampler = UniformSampler(
            time=clip_duration,
            config_json=rendering_cfg["sampler_config"],
            keep_marginal_keys=rendering_cfg.get("keep_marginal_keys", []),
            deduplication=deduplication,
        )
    else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}")

    # ---- render ----
    for i in range(n_samples):
        notes = note_sampler.sample(seed=i)
        audio = render_engine.render_notes(notes, time=clip_duration)

        wav_path = out_dir / f"sample_{i:02d}.wav"
        mid_path = out_dir / f"sample_{i:02d}.mid"

        sf.write(str(wav_path), audio, sr)

        pm = notes_to_midi(notes, clip_duration)
        pm.write(str(mid_path))

        print(f"  [{i}] {len(notes):3d} notes  audio={wav_path}  midi={mid_path}")

    print("Done.")


@hydra.main(version_base=None, config_path="configs/random_rendering", config_name="random_rendering")
def main(cfg: DictConfig) -> None:
    n_samples = int(OmegaConf.select(cfg, "n_samples", default=5))
    main_func(cfg, n_samples=n_samples)


if __name__ == "__main__":
    main()
