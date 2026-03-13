from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi


def resolve_maestro_midis(root: Path, split: str) -> list[Path]:
    csv_path = root / "maestro-v3.0.0.csv"
    assert csv_path.exists(), f"Missing metadata csv: {csv_path}"
    df = pd.read_csv(csv_path)
    if split != "all":
        df = df[df["split"] == split]
    return [root / midi_name for midi_name in df["midi_filename"].tolist()]


def resolve_slakh_midis(root: Path, split: str) -> list[Path]:
    split_names = ["train", "validation", "test"] if split == "all" else [split]
    midi_paths: list[Path] = []
    for split_name in split_names:
        split_dir = root / split_name
        assert split_dir.exists(), f"Missing split dir: {split_dir}"
        midi_paths.extend(sorted(split_dir.glob("Track*/MIDI/*.mid")))
    return midi_paths


def resolve_generic_midis(root: Path, pattern: str) -> list[Path]:
    return sorted(root.glob(pattern))


def summarize_velocity(velocity_counter: Counter[int], total_notes: int) -> dict:
    if total_notes == 0:
        return {
            "total_notes": 0,
            "top_values": [],
            "is_near_single_default": False,
        }

    velocities: list[int] = []
    for v, c in velocity_counter.items():
        velocities.extend([v] * c)
    vel = np.asarray(velocities, dtype=np.int32)

    top_values = [
        {"velocity": int(v), "count": int(c), "ratio": float(c / total_notes)}
        for v, c in velocity_counter.most_common(10)
    ]
    dominant_ratio = velocity_counter.most_common(1)[0][1] / total_notes

    return {
        "total_notes": int(total_notes),
        "min": int(vel.min()),
        "max": int(vel.max()),
        "mean": float(vel.mean()),
        "std": float(vel.std()),
        "q05": float(np.quantile(vel, 0.05)),
        "q25": float(np.quantile(vel, 0.25)),
        "q50": float(np.quantile(vel, 0.50)),
        "q75": float(np.quantile(vel, 0.75)),
        "q95": float(np.quantile(vel, 0.95)),
        "unique_values": int(len(velocity_counter)),
        "dominant_value_ratio": float(dominant_ratio),
        "is_near_single_default": bool(dominant_ratio >= 0.8),
        "top_values": top_values,
    }


def summarize_cc(cc_counter: Counter[int], cc_value_counter: dict[int, Counter[int]]) -> dict:
    all_cc = []
    for cc_num in range(103):
        cnt = int(cc_counter.get(cc_num, 0))
        value_hist = cc_value_counter.get(cc_num, Counter())
        top_values = [
            {"value": int(v), "count": int(c)}
            for v, c in value_hist.most_common(5)
        ]
        all_cc.append(
            {
                "cc": cc_num,
                "count": cnt,
                "unique_values": int(len(value_hist)),
                "top_values": top_values,
            }
        )

    top_cc = sorted(all_cc, key=lambda x: x["count"], reverse=True)[:15]
    focused = [x for x in all_cc if x["cc"] in (1, 64)]
    return {
        "cc_0_102": all_cc,
        "top_15_cc": top_cc,
        "focused": focused,
    }


def probe_midis(midi_paths: list[Path], max_files: int) -> dict:
    if max_files > 0:
        midi_paths = midi_paths[:max_files]

    file_count = len(midi_paths)
    assert file_count > 0, "No MIDI files found"

    program_track_counter: Counter[int] = Counter()
    program_note_counter: Counter[int] = Counter()
    velocity_counter: Counter[int] = Counter()
    cc_counter: Counter[int] = Counter()
    cc_value_counter: dict[int, Counter[int]] = {}

    parsed_files = 0
    failed_files: list[str] = []
    total_instruments = 0
    total_notes = 0
    total_cc_events = 0

    for midi_path in midi_paths:
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as e:
            failed_files.append(f"{midi_path}: {e}")
            continue

        parsed_files += 1

        for inst in pm.instruments:
            total_instruments += 1
            program = 128 if inst.is_drum else int(inst.program)
            program_track_counter[program] += 1

            for note in inst.notes:
                total_notes += 1
                program_note_counter[program] += 1
                velocity_counter[int(note.velocity)] += 1

            for cc in inst.control_changes:
                cc_num = int(cc.number)
                if 0 <= cc_num <= 102:
                    total_cc_events += 1
                    cc_counter[cc_num] += 1
                    if cc_num not in cc_value_counter:
                        cc_value_counter[cc_num] = Counter()
                    cc_value_counter[cc_num][int(cc.value)] += 1

    program_track_top = [
        {"program": int(p), "track_count": int(c)}
        for p, c in program_track_counter.most_common(20)
    ]
    program_note_top = [
        {"program": int(p), "note_count": int(c)}
        for p, c in program_note_counter.most_common(20)
    ]

    return {
        "files_requested": int(file_count),
        "files_parsed": int(parsed_files),
        "files_failed": int(len(failed_files)),
        "failed_examples": failed_files[:10],
        "total_instruments": int(total_instruments),
        "total_notes": int(total_notes),
        "total_cc_events_0_102": int(total_cc_events),
        "program": {
            "unique_programs": int(len(program_track_counter)),
            "track_count_top20": program_track_top,
            "note_count_top20": program_note_top,
            "has_drums_128": bool(128 in program_track_counter),
            "drums_128_track_count": int(program_track_counter.get(128, 0)),
            "drums_128_note_count": int(program_note_counter.get(128, 0)),
        },
        "velocity": summarize_velocity(velocity_counter, total_notes),
        "cc": summarize_cc(cc_counter, cc_value_counter),
    }


def print_brief_report(summary: dict) -> None:
    print("=" * 80)
    print("MIDI Probe Summary")
    print("=" * 80)
    print(
        "files: requested={files_requested} parsed={files_parsed} failed={files_failed}".format(
            **summary
        )
    )
    print(
        "totals: instruments={total_instruments} notes={total_notes} cc(0-102)={total_cc_events_0_102}".format(
            **summary
        )
    )

    prog = summary["program"]
    print("\n[Program]")
    print(
        f"unique={prog['unique_programs']} drums_128_track_count={prog['drums_128_track_count']} drums_128_note_count={prog['drums_128_note_count']}"
    )
    print("top track_count:")
    for item in prog["track_count_top20"][:10]:
        print(f"  program={item['program']:>3}  tracks={item['track_count']}")

    vel = summary["velocity"]
    print("\n[Velocity]")
    if vel["total_notes"] == 0:
        print("  no notes")
    else:
        print(
            f"min={vel['min']} max={vel['max']} mean={vel['mean']:.2f} std={vel['std']:.2f} unique={vel['unique_values']}"
        )
        print(
            f"q05={vel['q05']:.1f} q25={vel['q25']:.1f} q50={vel['q50']:.1f} q75={vel['q75']:.1f} q95={vel['q95']:.1f}"
        )
        print(
            f"dominant_value_ratio={vel['dominant_value_ratio']:.3f} near_single_default={vel['is_near_single_default']}"
        )
        print("top values:")
        for item in vel["top_values"][:10]:
            print(
                f"  velocity={item['velocity']:>3} count={item['count']:>8} ratio={item['ratio']:.3f}"
            )

    cc = summary["cc"]
    print("\n[CC 0-102]")
    print("focused (CC1, CC64):")
    for item in cc["focused"]:
        top_values_text = ", ".join(
            [f"{v['value']}:{v['count']}" for v in item["top_values"]]
        )
        print(
            f"  cc={item['cc']:>3} count={item['count']:>8} unique_values={item['unique_values']:>3} top_values=[{top_values_text}]"
        )
    print("top cc by count:")
    for item in cc["top_15_cc"]:
        print(f"  cc={item['cc']:>3} count={item['count']}")


def main_func(args: argparse.Namespace) -> None:
    root = Path(args.root)
    assert root.exists(), f"Root does not exist: {root}"

    if args.dataset == "maestro":
        midi_paths = resolve_maestro_midis(root=root, split=args.split)
    elif args.dataset == "slakh":
        midi_paths = resolve_slakh_midis(root=root, split=args.split)
    else:
        midi_paths = resolve_generic_midis(root=root, pattern=args.pattern)

    summary = probe_midis(midi_paths=midi_paths, max_files=args.max_files)

    summary["dataset"] = args.dataset
    summary["root"] = str(root)
    summary["split"] = args.split
    summary["pattern"] = args.pattern
    summary["max_files"] = args.max_files

    print_brief_report(summary)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSaved json report to: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe MIDI datasets (CC/program/velocity)")
    parser.add_argument("--dataset", type=str, default="generic", choices=["generic", "maestro", "slakh"])
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "validation", "test"])
    parser.add_argument("--pattern", type=str, default="**/*.mid")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--output-json", type=str, default="")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    main_func(parser.parse_args())