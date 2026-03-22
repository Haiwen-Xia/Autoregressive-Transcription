from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_step_from_ckpt(ckpt_path: Path) -> int:
    m = re.fullmatch(r"step=(\d+)\.pth", ckpt_path.name)
    assert m is not None, f"Invalid checkpoint filename: {ckpt_path.name}"
    return int(m.group(1))


def list_checkpoints(run_dir: Path) -> list[Path]:
    ckpt_dir = run_dir / "ckpt"
    assert ckpt_dir.exists(), f"Checkpoint directory not found: {ckpt_dir}"
    ckpts = sorted(ckpt_dir.glob("step=*.pth"), key=parse_step_from_ckpt)
    assert len(ckpts) > 0, f"No checkpoint found under: {ckpt_dir}"
    return ckpts


def flatten_numeric(d: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_numeric(v, key))
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[key] = float(v)
    return out


def safe_metric_filename(metric_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", metric_key)


def run_single_evaluate(
    evaluate_py: Path,
    run_dir: Path,
    ckpt_path: Path,
    summary_path: Path,
    device: str,
    max_samples: int,
    eval_mode: str,
    use_train: bool,
    teacher_forced_vocab_stats: bool,
    python_bin: str,
) -> None:
    cmd = [
        python_bin,
        str(evaluate_py),
        str(run_dir),
        "--path",
        str(ckpt_path),
        "--summary_path",
        str(summary_path),
        "--device",
        device,
        "--max_samples",
        str(max_samples),
        "--eval_mode",
        eval_mode,
    ]
    if use_train:
        cmd.append("--use_train")
    if teacher_forced_vocab_stats:
        cmd.append("--teacher_forced_vocab_stats")

    print(f"\n[Run] step={parse_step_from_ckpt(ckpt_path)}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def make_plots(records: list[dict], output_dir: Path) -> list[str]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_metric_keys: set[str] = set()
    for rec in records:
        all_metric_keys.update(rec["metrics_flat"].keys())

    steps = [int(rec["step"]) for rec in records]
    plot_paths: list[str] = []

    for metric_key in sorted(all_metric_keys):
        ys = [rec["metrics_flat"].get(metric_key) for rec in records]
        if sum(v is not None for v in ys) < 2:
            continue

        x_plot = [x for x, y in zip(steps, ys) if y is not None]
        y_plot = [float(y) for y in ys if y is not None]

        plt.figure(figsize=(8, 4.5))
        plt.plot(x_plot, y_plot, marker="o", linewidth=1.8)
        plt.title(metric_key)
        plt.xlabel("step")
        plt.ylabel(metric_key)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        fig_path = plots_dir / f"{safe_metric_filename(metric_key)}.png"
        plt.savefig(fig_path, dpi=160)
        plt.close()
        plot_paths.append(str(fig_path))

    priority_keys = [
        "teacher_forced_ce.mean_ce",
        "note_onset.f1",
        "note_offset.f1",
        "drum.f1",
        "empty_audio_pred_acc.acc",
    ]
    available_priority = [
        key for key in priority_keys if any(key in rec["metrics_flat"] for rec in records)
    ]

    if len(available_priority) > 0:
        plt.figure(figsize=(10, 5.5))
        for metric_key in available_priority:
            ys = [rec["metrics_flat"].get(metric_key) for rec in records]
            x_plot = [x for x, y in zip(steps, ys) if y is not None]
            y_plot = [float(y) for y in ys if y is not None]
            if len(x_plot) < 2:
                continue
            plt.plot(x_plot, y_plot, marker="o", linewidth=1.8, label=metric_key)

        plt.title("Selected Metrics vs Checkpoint Step")
        plt.xlabel("step")
        plt.ylabel("metric value")
        plt.grid(alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()

        combined_path = plots_dir / "selected_metrics.png"
        plt.savefig(combined_path, dpi=180)
        plt.close()
        plot_paths.append(str(combined_path))

    return plot_paths


def save_table(records: list[dict], output_csv: Path) -> None:
    all_metric_keys: set[str] = set()
    for rec in records:
        all_metric_keys.update(rec["metrics_flat"].keys())

    fieldnames = ["step", "checkpoint", "dataset", "eval_mode"] + sorted(all_metric_keys)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as fw:
        writer = csv.DictWriter(fw, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = {
                "step": rec["step"],
                "checkpoint": rec["checkpoint"],
                "dataset": rec["dataset"],
                "eval_mode": rec["eval_mode"],
            }
            row.update(rec["metrics_flat"])
            writer.writerow(row)


def main_func(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    assert run_dir.exists(), f"run_dir not found: {run_dir}"

    evaluate_py = (Path(__file__).resolve().parent.parent / "evaluate.py").resolve()
    assert evaluate_py.exists(), f"evaluate.py not found: {evaluate_py}"

    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "eval_all_ckpts"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.summary_path).resolve() if args.summary_path else (output_dir / "full_checkpoints.local.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    ckpts = list_checkpoints(run_dir)
    if args.max_ckpts > 0:
        ckpts = ckpts[: args.max_ckpts]

    print(f"Found {len(ckpts)} checkpoints under: {run_dir / 'ckpt'}")
    print(f"Isolated summary_path: {summary_path}")

    records: list[dict] = []

    for ckpt_path in ckpts:
        run_single_evaluate(
            evaluate_py=evaluate_py,
            run_dir=run_dir,
            ckpt_path=ckpt_path,
            summary_path=summary_path,
            device=args.device,
            max_samples=args.max_samples,
            eval_mode=args.eval_mode,
            use_train=args.use_train,
            teacher_forced_vocab_stats=args.teacher_forced_vocab_stats,
            python_bin=args.python_bin,
        )

        summary_json = run_dir / "eval" / "summary.json"
        assert summary_json.exists(), f"Missing summary after evaluate: {summary_json}"
        with open(summary_json, "r", encoding="utf-8") as fr:
            summary = json.load(fr)

        step = parse_step_from_ckpt(ckpt_path)
        dataset = str(summary.get("dataset", ""))
        eval_mode = str(summary.get("eval_mode", ""))
        metrics = summary.get("metrics", {})
        assert isinstance(metrics, dict), f"Expected dict metrics, got {type(metrics)}"

        per_ckpt_summary = output_dir / "per_checkpoint" / f"step={step}.summary.json"
        per_ckpt_summary.parent.mkdir(parents=True, exist_ok=True)
        with open(per_ckpt_summary, "w", encoding="utf-8") as fw:
            json.dump(summary, fw, ensure_ascii=False, indent=2)

        rec = {
            "step": step,
            "checkpoint": str(ckpt_path),
            "dataset": dataset,
            "eval_mode": eval_mode,
            "metrics_flat": flatten_numeric(metrics),
            "summary_path": str(per_ckpt_summary),
        }
        records.append(rec)

    records = sorted(records, key=lambda x: int(x["step"]))

    output_json = output_dir / "aggregate_metrics.json"
    with open(output_json, "w", encoding="utf-8") as fw:
        json.dump(records, fw, ensure_ascii=False, indent=2)

    output_csv = output_dir / "aggregate_metrics.csv"
    save_table(records=records, output_csv=output_csv)

    plot_paths = make_plots(records=records, output_dir=output_dir)

    report = {
        "run_dir": str(run_dir),
        "checkpoint_count": int(len(records)),
        "summary_path": str(summary_path),
        "aggregate_json": str(output_json),
        "aggregate_csv": str(output_csv),
        "plots": plot_paths,
    }
    report_path = output_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as fw:
        json.dump(report, fw, ensure_ascii=False, indent=2)

    print("\n=== Done ===")
    print(f"Aggregate JSON: {output_json}")
    print(f"Aggregate CSV : {output_csv}")
    print(f"Report JSON   : {report_path}")
    print(f"Plots count   : {len(plot_paths)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoints in a run dir and plot metric curves."
    )
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory containing ckpt/ and config.yaml")
    parser.add_argument("--summary_path", type=str, default="", help="Isolated registry path passed to evaluate.py")
    parser.add_argument("--output_dir", type=str, default="", help="Output dir for aggregate JSON/CSV/plots")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--eval_mode", type=str, default="segment", choices=["segment", "song"])
    parser.add_argument("--max_ckpts", type=int, default=0, help="0 means evaluate all checkpoints")
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--use_train", action="store_true")
    parser.add_argument("--teacher_forced_vocab_stats", action="store_true")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    main_func(parser.parse_args())
