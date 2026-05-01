"""
Statistics for MAESTRO 5-second excerpts.

Samples 50k 5-second excerpts from the MAESTRO dataset and computes:
- note number distribution (how many notes per excerpt)
- note duration distribution
- note pitch distribution (same as note number, but as pitch values)
- note velocity distribution

Results are saved to JSON.

Usage:
    python scripts/stats_maestro_5s.py --root /path/to/maestro --split train --n_samples 50000
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import sys 
sys.path.append(str(Path(__file__).parent.parent))  # Add project root to path
from audio_understanding.utils_midi_symusic import read_midi_clip_symusic


def load_maestro_meta(root: Path, split: str) -> tuple[list[str], list[float]]:
    """Load MAESTRO metadata and return (midi_paths, durations) for the given split."""
    csv_path = root / "maestro-v3.0.0.csv"
    assert csv_path.exists(), f"Missing metadata csv: {csv_path}"
    df = pd.read_csv(csv_path)
    if split != "all":
        df = df[df["split"] == split]
    
    midi_paths = [str(root / midi_name) for midi_name in df["midi_filename"].tolist()]
    durations = df["duration"].values.tolist()
    
    return midi_paths, durations


def sample_segments(
    midi_paths: list[str],
    durations: list[float],
    segment_duration: float,
    n_samples: int,
    seed: int = 42
) -> list[tuple[int, float]]:
    """
    Sample (midi_idx, start_time) pairs for segments.
    
    Returns list of (midi_idx, start_time) tuples.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    segments = []
    attempts = 0
    max_attempts = n_samples * 50
    
    while len(segments) < n_samples and attempts < max_attempts:
        idx = random.randint(0, len(midi_paths) - 1)
        dur = durations[idx]
        
        if dur < segment_duration:
            attempts += 1
            continue
        
        t = random.uniform(0, dur - segment_duration)
        segments.append((idx, t))
        attempts += 1
    
    if len(segments) < n_samples:
        print(f"Warning: Only got {len(segments)} segments (requested {n_samples})")
    
    return segments


def extract_note_features(
    midi_path: str,
    start_time: float,
    segment_duration: float
) -> dict:
    """
    Extract note features from a MIDI file segment.
    
    Returns dict with:
        - note_numbers: list of note counts per excerpt (actually just the count for this excerpt)
        - durations: list of note durations
        - pitches: list of note pitches (MIDI note numbers)
        - velocities: list of note velocities
    """
    clipped_score, _, _, _ = read_midi_clip_symusic(
        midi_path=midi_path,
        start_time=start_time,
        duration=segment_duration,
        mode="clip",
    )

    note_durations = []
    pitches = []
    velocities = []
    programs = []

    for track in clipped_score.tracks:
        program = 128 if bool(getattr(track, "is_drum", False)) else int(track.program)
        for note in track.notes:
            note_durations.append(float(note.duration))
            pitches.append(int(note.pitch))
            velocities.append(int(note.velocity))
            programs.append(program)
    
    return {
        "note_count": len(note_durations),
        "durations": note_durations,
        "pitches": pitches,
        "velocities": velocities,
        "programs": programs,
    }


def counter_to_distribution(counter: Counter) -> dict:
    total = sum(counter.values())
    if total == 0:
        return {"values": [], "probs": []}

    values = sorted(counter.keys())
    probs = [counter[v] for v in values]
    return {"values": values, "probs": probs}


def compute_statistics(
    root: str,
    split: str,
    segment_duration: float,
    n_samples: int,
    seed: int = 42,
    dur_round_ndigits: int = 3,
) -> dict:
    """
    Compute statistics for MAESTRO 5-second excerpts.
    """
    root_path = Path(root)
    
    # Load metadata
    print(f"Loading MAESTRO metadata (split={split})...")
    midi_paths, durations = load_maestro_meta(root_path, split)
    print(f"Found {len(midi_paths)} MIDI files")
    
    # Sample segments
    print(f"Sampling {n_samples} segments...")
    segments = sample_segments(midi_paths, durations, segment_duration, n_samples, seed)
    print(f"Sampled {len(segments)} segments")
    
    # Extract features
    all_note_counts = []
    all_durations = []
    all_pitches = []
    all_velocities = []
    all_programs = []
    
    t_global_start = time.time()
    t_last_log = t_global_start
    for i, (midi_idx, start_time) in enumerate(segments):
        features = extract_note_features(
            midi_path=midi_paths[midi_idx],
            start_time=start_time,
            segment_duration=segment_duration
        )
        
        all_note_counts.append(features["note_count"])
        all_durations.extend(features["durations"])
        all_pitches.extend(features["pitches"])
        all_velocities.extend(features["velocities"])
        all_programs.extend(features["programs"])

        processed = i + 1
        if processed % 1000 == 0:
            now = time.time()
            batch_sec = now - t_last_log
            elapsed_sec = now - t_global_start
            speed = processed / elapsed_sec
            remaining = len(segments) - processed
            eta_sec = remaining / speed
            print(
                f"Processing segment {processed}/{len(segments)} | "
                f"last1000: {batch_sec:.2f}s | elapsed: {elapsed_sec:.2f}s | "
                f"speed: {speed:.2f} seg/s | eta: {eta_sec:.2f}s"
            )
            t_last_log = now
    
    # Compute distributions
    print("Computing distributions...")
    rounded_durations = [round(d, dur_round_ndigits) for d in all_durations]

    dist = {
        "program": counter_to_distribution(Counter(all_programs)),
        "pitch": counter_to_distribution(Counter(all_pitches)),
        "dur": counter_to_distribution(Counter(rounded_durations)),
        "velocity": counter_to_distribution(Counter(all_velocities)),
        "num_notes": counter_to_distribution(Counter(all_note_counts)),
    }
    
    return dist


def print_report(stats: dict) -> None:
    """Print a brief summary report."""
    print("=" * 80)
    s = stats
    print("\n[Saved distributions]")
    print(f"  program values: {len(s['program']['values'])}")
    print(f"  pitch values: {len(s['pitch']['values'])}")
    print(f"  dur values: {len(s['dur']['values'])}")
    print(f"  velocity values: {len(s['velocity']['values'])}")
    print(f"  num_notes values: {len(s['num_notes']['values'])}")
    
    print("=" * 80)


def main_func(args: argparse.Namespace) -> None:
    stats = compute_statistics(
        root=args.root,
        split=args.split,
        segment_duration=args.segment_duration,
        n_samples=args.n_samples,
        seed=args.seed,
        dur_round_ndigits=args.dur_round_ndigits,
    )

    print_report(stats)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nSaved JSON to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute statistics for MAESTRO 5s excerpts")
    parser.add_argument("--root", type=str, required=True, help="MAESTRO dataset root")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test", "all"])
    parser.add_argument("--segment_duration", type=float, default=5.0, help="Segment duration in seconds")
    parser.add_argument("--n_samples", type=int, default=50000, help="Number of samples to extract")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dur_round_ndigits", type=int, default=3, help="Rounding digits for duration distribution")
    parser.add_argument("--output_json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()
    if args.output_json is None:
        args.output_json = f"stats/maestro_{args.segment_duration}.json"
    main_func(args)


if __name__ == "__main__":
    main()
