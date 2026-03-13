from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

STEP_CKPT_RE = re.compile(r"step=(\d+)\.pth$")


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit_idx = 0
    while value >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    return f"{value:.2f} {units[unit_idx]}"


def path_size(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        return path.stat().st_size

    total = 0
    for walk_root, _dir_names, file_names in iter_walk(path):
        for file_name in file_names:
            file_path = walk_root / file_name
            total += file_path.stat().st_size
    return total


def iter_walk(root: Path):
    walk_fn = getattr(root, "walk", None)
    if walk_fn is not None:
        yield from walk_fn(top_down=True)
        return

    for walk_root, dir_names, file_names in os.walk(root, topdown=True):
        yield Path(walk_root), dir_names, file_names


def is_empty_dir(path: Path) -> bool:
    return len(list(path.iterdir())) == 0


def collect_cleanup_actions(root: Path, keep_last_n: int | None) -> list[dict]:
    actions: list[dict] = []

    for walk_root, dir_names, _file_names in iter_walk(root):
        if "ckpt" not in dir_names:
            continue

        run_dir = walk_root
        ckpt_dir = run_dir / "ckpt"

        if is_empty_dir(ckpt_dir):
            targets = [run_dir]
            bytes_to_delete = path_size(run_dir)
            actions.append(
                {
                    "type": "wipe_run_contents",
                    "run_dir": run_dir,
                    "targets": targets,
                    "bytes": bytes_to_delete,
                }
            )
            continue

        if keep_last_n is None:
            continue

        ckpt_step_files: list[tuple[int, Path]] = []
        for ckpt_file in ckpt_dir.iterdir():
            if not ckpt_file.is_file():
                continue
            match = STEP_CKPT_RE.fullmatch(ckpt_file.name)
            if match is None:
                continue
            step = int(match.group(1))
            ckpt_step_files.append((step, ckpt_file))

        ckpt_step_files.sort(key=lambda item: item[0], reverse=True)
        to_delete = [item[1] for item in ckpt_step_files[keep_last_n:]]
        if len(to_delete) == 0:
            continue

        bytes_to_delete = sum(path_size(target) for target in to_delete)
        actions.append(
            {
                "type": "prune_ckpt",
                "run_dir": run_dir,
                "ckpt_dir": ckpt_dir,
                "keep_last_n": keep_last_n,
                "delete_ckpts": to_delete,
                "bytes": bytes_to_delete,
            }
        )

    return actions


def print_actions(actions: list[dict]) -> None:
    print("\nPlanned cleanup actions:")
    if len(actions) == 0:
        print("  (none)")
        return

    for idx, action in enumerate(actions, start=1):
        action_type = action["type"]
        if action_type == "wipe_run_contents":
            print(f"\n[{idx}] remove run dir: {action['run_dir']}")
            for target in action["targets"]:
                print(f"  - delete: {target}")
            print(f"  estimated freed: {format_bytes(action['bytes'])}")
            continue

        if action_type == "prune_ckpt":
            print(f"\n[{idx}] prune ckpt: {action['ckpt_dir']} (keep last {action['keep_last_n']})")
            for ckpt_file in action["delete_ckpts"]:
                print(f"  - delete: {ckpt_file}")
            print(f"  estimated freed: {format_bytes(action['bytes'])}")
            continue

        raise AssertionError(f"Unknown action type: {action_type}")


def execute_actions(actions: list[dict]) -> int:
    deleted_bytes = 0

    for action in actions:
        if action["type"] == "wipe_run_contents":
            for target in action["targets"]:
                deleted_bytes += path_size(target)
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            continue

        if action["type"] == "prune_ckpt":
            for ckpt_file in action["delete_ckpts"]:
                deleted_bytes += path_size(ckpt_file)
                ckpt_file.unlink()
            continue

        raise AssertionError(f"Unknown action type: {action['type']}")

    return deleted_bytes


def confirm_execute(non_interactive_yes: bool) -> bool:
    if non_interactive_yes:
        return True

    print("\nType YES to execute deletion, anything else to cancel:")
    user_input = input().strip()
    return user_input == "YES"


def main_func(args: argparse.Namespace) -> None:
    root = Path(args.root_dir).resolve()
    assert root.exists(), f"Root does not exist: {root}"
    assert root.is_dir(), f"Root is not a directory: {root}"

    keep_last_n = args.keep_last_n
    if keep_last_n is not None:
        assert keep_last_n >= 0

    actions = collect_cleanup_actions(root=root, keep_last_n=keep_last_n)
    total_estimated_bytes = sum(action["bytes"] for action in actions)

    print_actions(actions)
    print(f"\nTotal estimated to free: {format_bytes(total_estimated_bytes)}")

    if len(actions) == 0:
        print("No action needed.")
        return

    if args.dry_run:
        print("\nDry-run mode: no files were deleted.")
        return

    should_execute = confirm_execute(non_interactive_yes=args.yes)
    if not should_execute:
        print("Cancelled. No files were deleted.")
        return

    free_before = shutil.disk_usage(root).free
    deleted_bytes = execute_actions(actions)
    free_after = shutil.disk_usage(root).free
    delta_free = free_after - free_before

    print("\nDeletion finished.")
    print(f"Deleted bytes (sum of deleted files): {format_bytes(deleted_bytes)}")
    print(f"Disk free delta: {format_bytes(delta_free)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cleanup run directories under a root path. "
            "Rules: if p/ckpt exists and is empty, delete all contents under p; "
            "if p/ckpt has step checkpoints, optionally keep only last n."
        )
    )
    parser.add_argument("root_dir", type=str, help="Root directory to scan")
    parser.add_argument(
        "--keep-last-n",
        type=int,
        default=None,
        help="For non-empty ckpt dirs, keep latest n step=*.pth checkpoints and delete older ones",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned deletions; do not delete anything",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    main_func(args)


if __name__ == "__main__":
    main()
