from __future__ import annotations

import argparse
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path


def _normalize_gpu_ids(raw: str) -> tuple[str, int]:
    value = (raw or "").strip().replace(" ", "")
    if value.lower() in {"cpu", "none", ""}:
        return "", 0

    parts = [part for part in value.split(",") if part]
    if not parts:
        return "", 0

    for part in parts:
        if not part.isdigit():
            raise ValueError(f"Invalid gpu id: {part!r}")

    normalized = ",".join(parts)
    return normalized, len(parts)


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def build_launch_command(
    python_exe: str,
    train_script: Path,
    train_accelerate_script: Path,
    gpu_ids: str,
    nproc: int,
    main_process_port: int,
    passthrough: list[str],
) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()

    if nproc <= 1:
        cmd = [python_exe, str(train_script), f"train.device=cuda:{gpu_ids}"] + passthrough
        return cmd, env

    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    cmd = [
        "accelerate",
        "launch",
        "--gpu_ids",
        gpu_ids,
        "--num_processes",
        str(nproc),
        "--main_process_port",
        str(main_process_port),
        str(train_accelerate_script),
        *passthrough,
    ]
    return cmd, env


def main_func(args: argparse.Namespace) -> int:
    try:
        gpu_ids, nproc = _normalize_gpu_ids(args.gpu_ids)
    except ValueError as exc:
        print(f"[auto_launch] {exc}", file=sys.stderr)
        return 2

    base_dir = Path(__file__).resolve().parent
    train_script = base_dir / "train.py"
    train_accelerate_script = base_dir / "train_accelerate.py"

    launch_port = args.main_process_port
    if nproc > 1 and launch_port == 0:
        launch_port = _pick_free_port()

    cmd, env = build_launch_command(
        python_exe=sys.executable,
        train_script=train_script,
        train_accelerate_script=train_accelerate_script,
        gpu_ids=gpu_ids,
        nproc=nproc,
        main_process_port=launch_port,
        passthrough=args.train_args,
    )

    if nproc <= 1:
        print("[auto_launch] CPU launch:")
    else:
        print(
            f"[auto_launch] Multi-GPU launch: "
            f"CUDA_VISIBLE_DEVICES={gpu_ids}, num_processes={nproc}, main_process_port={launch_port}"
        )
    print("  " + shlex.join(cmd))

    proc = subprocess.Popen(cmd, env=env)
    if nproc <= 1:
        print(f"[auto_launch] train PID: {proc.pid}")
    else:
        print(f"[auto_launch] accelerate PID: {proc.pid}")
    return proc.wait()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified launcher for train.py / train_accelerate.py.\n"
            "Examples:\n"
            "  python auto_launch.py 0 -- --config-name asr_librispeech run_name=asr_exp\n"
            "  python auto_launch.py 0,1 -- --config-name piano_transcription_maestro train.batch_size_per_device=2\n"
            "  python auto_launch.py cpu -- --config-name audio_caption_clotho"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("gpu_ids", type=str, help="GPU ids like '0' or '0,1'; use 'cpu' for CPU run")
    parser.add_argument(
        "--main_process_port",
        type=int,
        default=0,
        help="Port for distributed init in multi-GPU mode. Use 0 to auto-select a free port.",
    )
    parsed, train_args = parser.parse_known_args(argv[1:])
    parsed.train_args = train_args

    if parsed.train_args and parsed.train_args[0] == "--":
        parsed.train_args = parsed.train_args[1:]

    return parsed


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    return main_func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
