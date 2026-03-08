"""Evaluation entry script.

Loads the latest checkpoint from a run directory and evaluates the model on
the test split of whichever dataset is configured.

Usage::

    python evaluate.py <dir> [--device cuda] [--max_samples N]

The script expects:
  - ``<dir>/config.yaml``    – the training configuration (saved by train.py)
  - ``<dir>/ckpt/step=N.pth`` – checkpoints; the one with the largest N is used
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from audio_understanding.utils import parse_yaml
from train import get_audio_encoder, get_llm, get_tokenizer


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(ckpt_dir: Path) -> Path:
    """Return the checkpoint file with the largest step number.

    Checkpoint files are expected to be named ``step=<N>.pth``.
    """
    ckpt_files = list(ckpt_dir.glob("step=*.pth"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    def _step(p: Path) -> int:
        m = re.fullmatch(r"step=(\d+)\.pth", p.name)
        return int(m.group(1)) if m else -1

    return max(ckpt_files, key=_step)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def get_eval_dataset(configs: dict) -> tuple[Any, str]:
    """Build the test dataset from config and return ``(dataset, dataset_name)``.

    Uses ``if/else`` to dispatch on each supported dataset name.
    The dataset is always built with ``target_transform=None`` so that raw
    Note objects are available for transcription evaluation.
    """
    from audidata.io.crops import StartCrop
    from audidata.transforms import Mono

    sr = configs["sample_rate"]
    clip_duration = configs["clip_duration"]
    test_datasets_cfg = configs["test_datasets"]

    for name, ds_cfg in test_datasets_cfg.items():

        if name == "MAESTRO":
            from audio_understanding.datasets.maestro import MAESTRO

            dataset = MAESTRO(
                root=ds_cfg["root"],
                split=ds_cfg["split"],
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration),
                transform=Mono(),
                load_target=True,
                extend_pedal=True,
                target_transform=None,
            )
            return dataset, name

        elif name == "Slakh2100":
            from audio_understanding.datasets.slakh2100 import Slakh2100

            mode = ds_cfg.get("mode", "all")
            sample_num = ds_cfg.get("sample_num", 1)
            keep_track_info = ds_cfg.get("keep_track_info", False)
            question_config = ds_cfg.get("question", None)

            dataset = Slakh2100(
                root=ds_cfg["root"],
                split=ds_cfg["split"],
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration),
                transform=Mono(),
                target=True,
                extend_pedal=True,
                target_transform=None,
                sample_num=sample_num,
                mode=mode,
                keep_track_info=keep_track_info,
                question_config=question_config,
            )
            return dataset, name

        elif name == "LibriSpeech":
            from audio_understanding.datasets.librispeech import LibriSpeech

            dataset = LibriSpeech(
                root=ds_cfg["root"],
                split=ds_cfg["split"],
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration),
                transform=Mono(),
            )
            return dataset, name

        elif name == "GTZAN":
            from audio_understanding.datasets.gtzan import GTZAN

            dataset = GTZAN(
                root=ds_cfg["root"],
                split=ds_cfg["split"],
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration),
                transform=Mono(),
            )
            return dataset, name

        elif name == "Clotho":
            from audio_understanding.datasets.clotho import Clotho

            dataset = Clotho(
                root=ds_cfg["root"],
                split=ds_cfg["split"],
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration),
                transform=Mono(),
            )
            return dataset, name

        elif name == "AudioCaps":
            from audio_understanding.datasets.audiocaps import AudioCaps

            dataset = AudioCaps(
                root=ds_cfg["root"],
                split=ds_cfg["split"],
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration),
                transform=Mono(),
            )
            return dataset, name

        elif name == "WavCaps":
            from audio_understanding.datasets.wavcaps import WavCaps

            dataset = WavCaps(
                root=ds_cfg["root"],
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration),
                transform=Mono(),
            )
            return dataset, name

        else:
            raise ValueError(f"Unknown dataset: {name}")

    raise ValueError("No test_datasets found in config.")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def make_transcription_inference_fn(
    audio_encoder: nn.Module,
    llm: nn.Module,
    tokenizer: Any,
    configs: dict,
    device: str,
):
    """Return an inference callable suitable for :func:`batch_evaluate`.

    The returned function accepts one dataset ``__getitem__`` output dict and
    returns a flat list of MIDI token strings using constrained decoding.
    """
    from inference_transcription import transcribe_audio

    def inference_fn(data: dict) -> list[str]:
        audio = data["audio"]
        if not isinstance(audio, torch.Tensor):
            audio = torch.Tensor(audio)
        audio_tensor = audio.unsqueeze(0).to(device)  # (1, c, samples)
        question = data.get("question", "Music transcription.")
        result = transcribe_audio(
            audio_encoder=audio_encoder,
            llm=llm,
            tokenizer=tokenizer,
            audio_tensor=audio_tensor,
            configs=configs,
            question=question,
            temperature=1.0,
            top_k=1,
        )
        return result["tokens"]

    return inference_fn


def run_text_evaluation(
    dataset: Any,
    dataset_name: str,
    audio_encoder: nn.Module,
    llm: nn.Module,
    tokenizer: Any,
    configs: dict,
    device: str,
    max_samples: int | None = None,
) -> list[dict]:
    """Run greedy inference on text-output datasets and print sample results.

    Returns a list of ``{"prediction": str, "target": str}`` dicts.
    """
    n_total = len(dataset)
    if max_samples is not None:
        n_total = min(n_total, max_samples)

    try:
        from tqdm import tqdm
        iterator = tqdm(range(n_total), desc=f"Evaluating {dataset_name}")
    except ImportError:
        iterator = range(n_total)

    audio_encoder.eval()
    llm.eval()

    results = []
    for i in iterator:
        data = dataset[i]

        audio = data["audio"]
        if not isinstance(audio, torch.Tensor):
            audio = torch.Tensor(audio)
        audio_tensor = audio.unsqueeze(0).to(device)  # (1, c, samples)

        question = data.get("question", "")

        with torch.no_grad():
            audio_latent = audio_encoder.encode(audio=audio_tensor, train_mode=False)

            question_ids = tokenizer.texts_to_ids(
                texts=[question],
                fix_length=configs["max_question_len"],
            ).to(device)

            answering_ids_in = torch.LongTensor([[tokenizer.cls_token_id]]).to(device)

            seqs = [audio_latent, question_ids, answering_ids_in]
            seq_types = ["audio", "id", "id"]

            output_seqs = llm.generate(
                seqs=seqs,
                seq_types=seq_types,
                max_new_ids=configs["max_answering_len"],
                temperature=1.0,
                top_k=1,
            )

        output_text = tokenizer.tok.decode(output_seqs[2][0], skip_special_tokens=True)

        # Collect the ground-truth field (varies by dataset)
        target: Any = (
            data.get("target")
            or data.get("text")
            or data.get("caption")
            or data.get("label")
            or ""
        )
        results.append({"prediction": output_text, "target": target})

        if i < 5:
            print(f"  [Sample {i}] Q: {question!r}")
            print(f"             Target: {target!r}")
            print(f"             Pred:   {output_text!r}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model from a run directory."
    )
    parser.add_argument(
        "dir",
        type=str,
        help="Run directory containing config.yaml and ckpt/step=N.pth files.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap the number of evaluation samples (useful for quick checks).",
    )
    args = parser.parse_args()

    run_dir = Path(args.dir)
    config_path = run_dir / "config.yaml"
    ckpt_dir = run_dir / "ckpt"

    # Load config
    configs = parse_yaml(str(config_path))
    print(f"Loaded config: {config_path}")

    # Find latest checkpoint
    ckpt_path = find_latest_checkpoint(ckpt_dir)
    print(f"Using checkpoint: {ckpt_path}")

    device = args.device

    # Build models
    audio_encoder = get_audio_encoder(configs=configs, ckpt_path=str(ckpt_path)).to(device)
    tokenizer = get_tokenizer(configs=configs)
    llm = get_llm(
        configs=configs,
        audio_latent_dim=audio_encoder.latent_dim,
        vocab_size=len(tokenizer),
        ckpt_path=str(ckpt_path),
    ).to(device)

    audio_encoder.eval()
    llm.eval()

    # Build test dataset
    dataset, dataset_name = get_eval_dataset(configs)
    print(f"Dataset: {dataset_name}, test samples: {len(dataset)}")

    # --- Evaluate ---
    if dataset_name in ("MAESTRO", "Slakh2100"):
        from audio_understanding.eval.transcription.batch_eval import batch_evaluate

        include_program = configs.get("midi_include_program", False)
        fps = configs["fps"]

        inference_fn = make_transcription_inference_fn(
            audio_encoder=audio_encoder,
            llm=llm,
            tokenizer=tokenizer,
            configs=configs,
            device=device,
        )

        results = batch_evaluate(
            dataset=dataset,
            inference_fn=inference_fn,
            fps=fps,
            include_program=include_program,
            max_samples=args.max_samples,
            verbose=True,
        )

        print("\n=== Evaluation Results ===")
        print(json.dumps(results, indent=2, default=str))

    elif dataset_name in ("LibriSpeech", "GTZAN", "Clotho", "AudioCaps", "WavCaps"):
        results = run_text_evaluation(
            dataset=dataset,
            dataset_name=dataset_name,
            audio_encoder=audio_encoder,
            llm=llm,
            tokenizer=tokenizer,
            configs=configs,
            device=device,
            max_samples=args.max_samples,
        )

        print(f"\n=== Evaluation Results ({dataset_name}) ===")
        print(f"Evaluated {len(results)} samples.")

    else:
        raise ValueError(f"Unsupported dataset for evaluation: {dataset_name}")


if __name__ == "__main__":
    main()
