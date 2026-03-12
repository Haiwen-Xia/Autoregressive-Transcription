"""Evaluation entry script.

Loads the latest checkpoint from a run directory and evaluates the model on
the test split of whichever dataset is configured.

Two transcription eval modes:
  - segment: random clips from the dataset, single-chunk inference (fast)
  - song:    full songs, chunked inference with timestamp shift (thorough)

Usage::

    python evaluate.py <dir> [--device cuda] [--max_samples N] [--eval_mode segment|song]

The script expects:
  - ``<dir>/config.yaml``    – the training configuration (saved by train.py)
  - ``<dir>/ckpt/step=N.pth`` – checkpoints; the one with the largest N is used
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, cast
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

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

def get_eval_dataset(configs: dict, use_train: bool, segment: bool = False) -> tuple[Any, str]:
    """Build the test dataset from config and return ``(dataset, dataset_name)``.

    Args:
        segment: if True, build with RandomCrop (for segment-wise eval).
                 if False, build with crop=None (for song-wise eval).
    """
    from audidata.transforms import Mono
    from audidata.io.crops import RandomCrop

    sr = configs["sample_rate"]
    clip_duration = configs["clip_duration"]
    test_datasets_cfg = configs["train_datasets"] if use_train else configs["test_datasets"]
    crop = RandomCrop(clip_duration=clip_duration, end_pad=clip_duration - 0.1) if segment else None

    for name, ds_cfg in test_datasets_cfg.items():

        if name == "MAESTRO":
            from audio_understanding.datasets.maestro import MAESTRO

            dataset = MAESTRO(
                root=ds_cfg["root"],
                split=ds_cfg["split"],
                sr=sr,
                crop=crop,
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
                crop=crop,
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

        else:
            raise ValueError(f"Unknown dataset: {name}")

    raise ValueError("No test_datasets found in config.")




def _decode_generated_token_ids(tokenizer: Any, token_ids: list[int]) -> list[str]:
    tok = tokenizer.tok
    ids = token_ids
    if len(ids) > 0 and ids[0] == tok.cls_token_id:
        ids = ids[1:]
    if tok.sep_token_id in ids:
        ids = ids[: ids.index(tok.sep_token_id)]
    return cast(list[str], tok.convert_ids_to_tokens(ids))


def _shift_time_index_tokens(tokens: list[str], frame_offset: int) -> list[str]:
    if frame_offset == 0:
        return list(tokens)

    shifted = []
    for tok in tokens:
        if tok.startswith("time_index="):
            value = int(tok.split("=", 1)[1]) + frame_offset
            shifted.append(f"time_index={value}")
        else:
            shifted.append(tok)
    return shifted


def _split_contiguous_chunks(audio_mono: torch.Tensor, chunk_samples: int) -> list[tuple[int, torch.Tensor]]:
    assert audio_mono.ndim == 1
    assert chunk_samples > 0

    total_samples = int(audio_mono.shape[0])
    chunks = []
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = audio_mono[start:end]
        if int(chunk.shape[0]) == 0:
            continue
        chunks.append((start, chunk))

    assert len(chunks) > 0
    return chunks


def _transcribe_song_chunked(
    data: dict,
    audio_encoder: nn.Module,
    llm: nn.Module,
    tokenizer: Any,
    configs: dict,
    device: str,
    chunk_batch_size: int = 8,
) -> list[str]:
    """Transcribe a full song by splitting into chunks, batching chunks, and merging tokens.

    Args:
        chunk_batch_size: how many chunks to batch through the model at once.

    Returns:
        Merged token list with timestamps shifted to absolute song time.
    """
    from audio_understanding.target_transforms.midi_constrained import MidiConstrainedDecoder

    include_program = bool(configs.get("midi_include_program", False))
    sr = int(configs["sample_rate"])
    chunk_seconds = float(configs.get("chunk_seconds", 5.0))
    chunk_samples = max(1, int(round(chunk_seconds * sr)))
    fps = float(configs["fps"])

    audio = data["audio"]
    if not isinstance(audio, torch.Tensor):
        audio = torch.as_tensor(audio)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    assert audio.ndim == 2, f"Expected audio shape (c, t), got {tuple(audio.shape)}"

    question = data.get("question", "Music transcription.")
    mono = audio[0].to(dtype=torch.float32)
    chunks = _split_contiguous_chunks(audio_mono=mono, chunk_samples=chunk_samples)

    merged_tokens: list[str] = []

    # Process chunks in batches
    for batch_start in range(0, len(chunks), chunk_batch_size):
        batch_chunks = chunks[batch_start : batch_start + chunk_batch_size]
        b = len(batch_chunks)

        # Pad all chunks to same length for batching
        max_len = max(c.shape[0] for _, c in batch_chunks)
        padded = torch.zeros(b, 1, max_len, dtype=torch.float32)
        for j, (_, chunk) in enumerate(batch_chunks):
            padded[j, 0, :chunk.shape[0]] = chunk
        padded = padded.to(device)  # (b, 1, max_len)

        with torch.no_grad():
            audio_latent = cast(Any, audio_encoder).encode(
                audio=padded, train_mode=False,
            )  # (b, t, d)

        question_ids = cast(Any, tokenizer).texts_to_ids(
            texts=[question] * b,
            fix_length=configs["max_question_len"],
        ).to(device)  # (b, q_len)

        answering_ids = torch.full(
            (b, 1),
            fill_value=cast(Any, tokenizer).cls_token_id,
            dtype=torch.long,
            device=device,
        )  # (b, 1)

        constraints = [
            MidiConstrainedDecoder(
                tokenizer=tokenizer,
                vocab_size=len(tokenizer),
                include_program=include_program,
                device=device,
            )
            for _ in range(b)
        ]

        seqs = [audio_latent, question_ids, answering_ids]
        seq_types = ["audio", "id", "id"]

        with torch.no_grad():
            output_seqs = cast(Any, llm).generate_constrained_batch(
                seqs=seqs,
                seq_types=seq_types,
                max_new_ids=configs["max_answering_len"],
                constraints=constraints,
                sep_token_id=cast(Any, tokenizer).tok.sep_token_id,
                temperature=1.0,
                top_k=1,
            )

        # Decode each chunk's output and shift timestamps
        for j in range(b):
            token_ids = output_seqs[2][j].detach().cpu().tolist()
            chunk_tokens = _decode_generated_token_ids(tokenizer=tokenizer, token_ids=token_ids)
            start_sample = batch_chunks[j][0]
            frame_offset = int(round((start_sample / sr) * fps))
            shifted = _shift_time_index_tokens(chunk_tokens, frame_offset=frame_offset)
            merged_tokens.extend(shifted)

    return merged_tokens


# ---------------------------------------------------------------------------
# Mode 0: segment eval (fast, random clips)
# ---------------------------------------------------------------------------

def _make_inference_fn(audio_encoder, llm, tokenizer, configs, device):
    """Build an inference_fn compatible with batch_evaluate."""
    from inference_transcription import transcribe_audio

    def inference_fn(data: dict) -> list[str]:
        audio = data["audio"]
        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio)
        audio_tensor = audio.unsqueeze(0).to(device)
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
        return cast(list[str], result["tokens"])

    return inference_fn


def evaluate_segment(
    dataset: Any,
    audio_encoder: nn.Module,
    llm: nn.Module,
    tokenizer: Any,
    configs: dict,
    device: str,
    max_samples: int | None,
) -> tuple[dict, list[dict]]:
    """Segment-wise eval: delegate to batch_evaluate from batch_eval.py."""
    from audio_understanding.eval.transcription.batch_eval import batch_evaluate

    inference_fn = _make_inference_fn(audio_encoder, llm, tokenizer, configs, device)
    include_program = bool(configs.get("midi_include_program", False))
    fps = float(configs["fps"])

    summary = batch_evaluate(
        dataset=dataset,
        inference_fn=inference_fn,
        fps=fps,
        include_program=include_program,
        max_samples=max_samples,
    )
    summary["eval_mode"] = "segment"
    return summary, []  # batch_evaluate already prints per-sample, no separate details needed


# ---------------------------------------------------------------------------
# Mode 1: song eval (full songs, chunked inference)
# ---------------------------------------------------------------------------

def evaluate_song(
    dataset: Any,
    audio_encoder: nn.Module,
    llm: nn.Module,
    tokenizer: Any,
    configs: dict,
    device: str,
    max_samples: int | None,
    chunk_batch_size: int = 8,
) -> tuple[dict, list[dict]]:
    """Song-wise eval: delegate to batch_evaluate with chunked inference."""
    from audio_understanding.eval.transcription.batch_eval import batch_evaluate

    include_program = bool(configs.get("midi_include_program", False))
    fps = float(configs["fps"])

    def inference_fn(data: dict) -> list[str]:
        return _transcribe_song_chunked(
            data=data,
            audio_encoder=audio_encoder,
            llm=llm,
            tokenizer=tokenizer,
            configs=configs,
            device=device,
            chunk_batch_size=chunk_batch_size,
        )

    summary = batch_evaluate(
        dataset=dataset,
        inference_fn=inference_fn,
        fps=fps,
        include_program=include_program,
        max_samples=max_samples,
    )
    summary["eval_mode"] = "song"
    return summary, []


def _get_target_text(data: dict) -> str:
    target = (
        data.get("target")
        or data.get("text")
        or data.get("caption")
        or data.get("label")
        or ""
    )
    assert isinstance(target, str), f"Expected text target to be str, but got {type(target)}"
    return target


def _compute_teacher_forced_ce_and_logits(
    data: dict,
    answering: str | list[str],
    audio_encoder: nn.Module,
    llm: nn.Module,
    tokenizer: Any,
    configs: dict,
    device: str,
    top_k: int = 5,
    max_trace_steps: int = 16,
) -> dict:
    audio = data["audio"]
    if not isinstance(audio, torch.Tensor):
        audio = torch.as_tensor(audio)
    audio_tensor = audio.unsqueeze(0).to(device)
    enc = cast(Any, audio_encoder)
    model = cast(Any, llm)
    tok = cast(Any, tokenizer)

    question = data.get("question", "")
    assert isinstance(question, str)

    with torch.no_grad():
        audio_latent = enc.encode(audio=audio_tensor, train_mode=False)
        question_ids = tok.texts_to_ids(
            texts=[question],
            fix_length=configs["max_question_len"],
        ).to(device)
        answering_ids = tok.texts_to_ids(
            texts=[answering],
            fix_length=configs["max_answering_len"],
        ).to(device)

        seqs = [audio_latent, question_ids, answering_ids]
        seq_types = ["audio", "id", "id"]
        output_seqs = model(
            seqs=seqs,
            seq_types=seq_types,
            mask=None,
        )

    # For next-token CE: predict answering_ids[:, 1:] from logits[:, :-1, :].
    logits = output_seqs[2][:, 0:-1, :]
    targets = answering_ids[:, 1:]
    valid_mask = targets != tok.pad_token_id

    token_losses = F.cross_entropy(
        input=logits.flatten(0, 1),
        target=targets.flatten(0, 1),
        ignore_index=tok.pad_token_id,
        reduction="none",
    ).view_as(targets)

    valid_token_count = int(valid_mask.sum().item())
    loss_sum = (token_losses * valid_mask).sum().item()
    mean_ce = float(loss_sum / max(valid_token_count, 1))

    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    valid_target_logits = target_logits[valid_mask].detach().cpu().tolist()

    trace = []
    steps = min(max_trace_steps, logits.shape[1])
    vocab_top_k = min(top_k, logits.shape[-1])
    for step in range(steps):
        if not bool(valid_mask[0, step]):
            continue
        step_logits = logits[0, step]
        top_values, top_ids = torch.topk(step_logits, k=vocab_top_k)
        trace.append(
            {
                "step": int(step),
                "target_id": int(targets[0, step].item()),
                "target_logit": float(step_logits[targets[0, step]].item()),
                "topk_ids": [int(x) for x in top_ids.detach().cpu().tolist()],
                "topk_logits": [float(x) for x in top_values.detach().cpu().tolist()],
            }
        )

    return {
        "mean_ce": mean_ce,
        "valid_token_count": valid_token_count,
        "target_logits": [float(x) for x in valid_target_logits],
        "logit_trace": trace,
    }


def _build_transcription_target_tokens(data: dict, fps: float, include_program: bool) -> list[str]:
    from audio_understanding.target_transforms.midi import MIDI2Tokens

    midi_to_tokens = MIDI2Tokens(fps=fps, include_program=include_program)
    target_data = {
        "start_time": data["start_time"],
        "duration": data["duration"],
        "note": data.get("note", []),
        "pedal": data.get("pedal", []),
        "note_program": data.get("note_program"),
    }
    token_data = midi_to_tokens(copy.deepcopy(target_data))
    tokens = token_data.get("token", [])
    assert isinstance(tokens, list)
    return tokens


def _format_token_preview_lines(tokens: list[str], include_program: bool, max_lines: int = 10) -> list[str]:
    from inference_transcription import format_tokens_by_event

    formatted = format_tokens_by_event(tokens=tokens, include_program=include_program)
    lines = formatted.splitlines()
    return lines[:max_lines]


def _collect_transcription_sample_previews(
    dataset: Any,
    audio_encoder: nn.Module,
    llm: nn.Module,
    tokenizer: Any,
    configs: dict,
    device: str,
    include_program: bool,
    max_samples: int | None,
    n_samples: int = 3,
) -> list[dict]:
    from inference_transcription import transcribe_audio

    n_total = len(dataset)
    if max_samples is not None:
        n_total = min(n_total, max_samples)

    if n_total <= 0:
        return []

    step = max(1, n_total // n_samples)
    sample_indices = list(range(0, n_total, step))[:n_samples]

    previews = []
    for idx in sample_indices:
        data = dataset[idx]
        audio = data["audio"]
        if not isinstance(audio, torch.Tensor):
            audio = torch.Tensor(audio)
        audio_tensor = audio.unsqueeze(0).to(device)

        question = data.get("question", "Music transcription.")
        target_tokens = _build_transcription_target_tokens(
            data=data,
            fps=configs["fps"],
            include_program=include_program,
        )

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

        gt_preview = _format_token_preview_lines(
            tokens=target_tokens,
            include_program=include_program,
            max_lines=10,
        )
        pred_preview = _format_token_preview_lines(
            tokens=result["tokens"],
            include_program=include_program,
            max_lines=10,
        )

        print(f"\n=== Preview sample idx={idx} ===")
        print(f"audio_path: {data.get('audio_path', '')}")
        print(f"midi_path: {data.get('midi_path', '')}")
        print(f"question: {question}")
        print(f"GT token count: {len(target_tokens)}")
        print("GT first 10 lines:")
        for line in gt_preview:
            print(line)
        print(f"Pred token count: {len(result['tokens'])}")
        print("Pred first 10 lines:")
        for line in pred_preview:
            print(line)

        previews.append(
            {
                "sample_index": int(idx),
                "audio_path": data.get("audio_path", ""),
                "midi_path": data.get("midi_path", ""),
                "question": question,
                "gt_token_count": len(target_tokens),
                "pred_token_count": len(result["tokens"]),
                "gt_preview_lines": gt_preview,
                "pred_preview_lines": pred_preview,
            }
        )

    return previews


def _select_global_metrics_only(metrics: Any) -> Any:
    if isinstance(metrics, dict):
        selected = {}
        for key, value in metrics.items():
            key_l = str(key).lower()
            if "instrument" in key_l or "instru" in key_l or "track" in key_l:
                continue
            if isinstance(value, (dict, list)):
                selected_value = _select_global_metrics_only(value)
                if selected_value in ({}, []):
                    continue
                selected[key] = selected_value
                continue
            if isinstance(value, (int, float, bool, str)):
                selected[key] = value
        return selected

    if isinstance(metrics, list):
        # Global registry should not store per-sample or per-class arrays.
        return []

    return metrics


def collect_transcription_teacher_forced_stats(
    dataset: Any,
    audio_encoder: nn.Module,
    llm: nn.Module,
    tokenizer: Any,
    configs: dict,
    device: str,
    include_program: bool,
    max_samples: int | None,
) -> tuple[dict, list[dict]]:
    n_total = len(dataset)
    if max_samples is not None:
        n_total = min(n_total, max_samples)

    ce_values = []
    token_counts = []
    per_sample = []

    try:
        from tqdm import tqdm

        iterator = tqdm(range(n_total), desc="Teacher-forced CE (transcription)")
    except ImportError:
        iterator = range(n_total)

    for i in iterator:
        data = dataset[i]
        target_tokens = _build_transcription_target_tokens(
            data=data,
            fps=configs["fps"],
            include_program=include_program,
        )
        ce_info = _compute_teacher_forced_ce_and_logits(
            data=data,
            answering=target_tokens,
            audio_encoder=audio_encoder,
            llm=llm,
            tokenizer=tokenizer,
            configs=configs,
            device=device,
        )

        ce_values.append(ce_info["mean_ce"])
        token_counts.append(ce_info["valid_token_count"])
        per_sample.append(
            {
                "sample_index": int(i),
                "audio_path": data.get("audio_path", ""),
                "midi_path": data.get("midi_path", ""),
                "question": data.get("question", ""),
                "target_token_num": len(target_tokens),
                "valid_token_count": ce_info["valid_token_count"],
                "mean_ce": ce_info["mean_ce"],
                "target_logit_mean": float(sum(ce_info["target_logits"]) / max(len(ce_info["target_logits"]), 1)),
                "logit_trace": ce_info["logit_trace"],
            }
        )

    summary = {
        "samples": int(len(ce_values)),
        "mean_ce": float(sum(ce_values) / max(len(ce_values), 1)),
        "total_valid_tokens": int(sum(token_counts)),
    }
    return summary, per_sample


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fw:
        json.dump(data, fw, indent=2, ensure_ascii=False, default=str)


def update_full_checkpoints(full_ckpt_path: Path, run_key: str, record: dict) -> None:
    if full_ckpt_path.exists():
        with open(full_ckpt_path, "r", encoding="utf-8") as fr:
            all_records = json.load(fr)
    else:
        all_records = {}

    all_records[run_key] = record
    save_json(full_ckpt_path, all_records)


def run_text_evaluation(
    dataset: Any,
    dataset_name: str,
    audio_encoder: nn.Module,
    llm: nn.Module,
    tokenizer: Any,
    configs: dict,
    device: str,
    max_samples: int | None = None,
) -> tuple[list[dict], dict, list[dict]]:
    """Run greedy inference on text-output datasets and print sample results.

    Returns:
        - sample results list
        - CE summary dict
        - per-sample CE/logit details
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
    tok = cast(Any, tokenizer)
    model = cast(Any, llm)

    results = []
    ce_values = []
    ce_details = []
    for i in iterator:
        data = dataset[i]

        audio = data["audio"]
        if not isinstance(audio, torch.Tensor):
            audio = torch.Tensor(audio)
        audio_tensor = audio.unsqueeze(0).to(device)  # (1, c, samples)

        question = data.get("question", "")
        assert isinstance(question, str)

        with torch.no_grad():
            audio_latent = audio_encoder.encode(audio=audio_tensor, train_mode=False)

            question_ids = tok.texts_to_ids(
                texts=[question],
                fix_length=configs["max_question_len"],
            ).to(device)

            answering_ids_in = torch.LongTensor([[tok.cls_token_id]]).to(device)

            seqs = [audio_latent, question_ids, answering_ids_in]
            seq_types = ["audio", "id", "id"]

            output_seqs = cast(Any, model).generate(
                seqs=seqs,
                seq_types=seq_types,
                max_new_ids=configs["max_answering_len"],
                temperature=1.0,
                top_k=1,
            )

        output_text = tok.tok.decode(output_seqs[2][0], skip_special_tokens=True)

        target = _get_target_text(data)
        ce_info = _compute_teacher_forced_ce_and_logits(
            data=data,
            answering=target,
            audio_encoder=audio_encoder,
            llm=llm,
            tokenizer=tokenizer,
            configs=configs,
            device=device,
        )
        ce_values.append(ce_info["mean_ce"])

        results.append(
            {
                "sample_index": int(i),
                "question": question,
                "prediction": output_text,
                "target": target,
                "mean_ce": ce_info["mean_ce"],
                "valid_token_count": ce_info["valid_token_count"],
            }
        )

        ce_details.append(
            {
                "sample_index": int(i),
                "audio_path": data.get("audio_path", ""),
                "question": question,
                "target": target,
                "valid_token_count": ce_info["valid_token_count"],
                "mean_ce": ce_info["mean_ce"],
                "target_logits": ce_info["target_logits"],
                "logit_trace": ce_info["logit_trace"],
            }
        )

        if i < 5:
            print(f"  [Sample {i}] Q: {question!r}")
            print(f"             Target: {target!r}")
            print(f"             Pred:   {output_text!r}")
            print(f"             CE:     {ce_info['mean_ce']:.6f}")

    ce_summary = {
        "samples": int(len(ce_values)),
        "mean_ce": float(sum(ce_values) / max(len(ce_values), 1)),
    }

    return results, ce_summary, ce_details


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main_func(args: argparse.Namespace) -> None:
    evaluate_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    run_dir = Path(args.dir).resolve()
    eval_dir = run_dir / "eval"
    full_ckpt_path = Path(__file__).resolve().parent / "full_checkpoints.json"

    config_path = run_dir / "config.yaml"
    ckpt_dir = run_dir / "ckpt"

    # Load config
    configs = parse_yaml(str(config_path))
    print(f"Loaded config: {config_path}")

    # Find latest checkpoint
    ckpt_path = find_latest_checkpoint(ckpt_dir)
    print(f"Using checkpoint: {ckpt_path}")

    device = args.device
    eval_mode = args.eval_mode

    # Build models
    audio_encoder = get_audio_encoder(configs=configs, ckpt_path=str(ckpt_path)).to(device)
    tokenizer = get_tokenizer(configs=configs)
    audio_latent_dim = int(getattr(cast(Any, audio_encoder), "latent_dim"))
    llm = get_llm(
        configs=configs,
        audio_latent_dim=audio_latent_dim,
        vocab_size=len(tokenizer),
        ckpt_path=str(ckpt_path),
    ).to(device)

    audio_encoder.eval()
    llm.eval()

    # Build test dataset
    is_segment = (eval_mode == "segment")
    dataset, dataset_name = get_eval_dataset(configs, args.use_train, segment=is_segment)
    print(f"Dataset: {dataset_name}, eval_mode: {eval_mode}, samples: {len(dataset)}")

    eval_metrics = {}
    additional_info = {
        "evaluate_time": evaluate_time,
        "dataset": dataset_name,
        "eval_mode": eval_mode,
        "max_samples": args.max_samples,
    }

    # --- Evaluate ---
    if dataset_name in ("MAESTRO", "Slakh2100"):
        if eval_mode == "segment":
            print("\n=== Segment-wise Evaluation (fast) ===")
            results, inference_details = evaluate_segment(
                dataset=dataset,
                audio_encoder=audio_encoder,
                llm=llm,
                tokenizer=tokenizer,
                configs=configs,
                device=device,
                max_samples=args.max_samples,
            )
        else:
            print("\n=== Song-wise Evaluation (full audio) ===")
            results, inference_details = evaluate_song(
                dataset=dataset,
                audio_encoder=audio_encoder,
                llm=llm,
                tokenizer=tokenizer,
                configs=configs,
                device=device,
                max_samples=args.max_samples // 4,
            )

        eval_metrics = results
        additional_info["inference_details"] = inference_details

        print("\n=== Evaluation Results ===")
        print(json.dumps(results, indent=2, default=str))

    elif dataset_name in ("LibriSpeech", "GTZAN", "Clotho", "AudioCaps", "WavCaps"):
        text_results, ce_summary, ce_details = run_text_evaluation(
            dataset=dataset,
            dataset_name=dataset_name,
            audio_encoder=audio_encoder,
            llm=llm,
            tokenizer=tokenizer,
            configs=configs,
            device=device,
            max_samples=args.max_samples,
        )

        eval_metrics = {
            "n_samples": len(text_results),
            "teacher_forced_ce": ce_summary,
        }
        additional_info["text_results"] = text_results
        additional_info["teacher_forced_details"] = ce_details

        print(f"\n=== Evaluation Results ({dataset_name}) ===")
        print(f"Evaluated {len(text_results)} samples.")
        print(f"Mean CE: {ce_summary['mean_ce']:.6f}")

    else:
        raise ValueError(f"Unsupported dataset for evaluation: {dataset_name}")

    summary_record = {
        "evaluate_time": evaluate_time,
        "checkpoint": str(ckpt_path),
        "config": configs,
        "dataset": dataset_name,
        "eval_mode": eval_mode,
        "metrics": eval_metrics,
    }

    global_summary_record = {
        "evaluate_time": evaluate_time,
        "checkpoint": str(ckpt_path),
        "dataset": dataset_name,
        "eval_mode": eval_mode,
        "metrics": _select_global_metrics_only(eval_metrics),
    }

    update_full_checkpoints(
        full_ckpt_path=full_ckpt_path,
        run_key=str(run_dir),
        record=global_summary_record,
    )
    save_json(eval_dir / "summary.json", summary_record)
    save_json(eval_dir / "additional_info.json", additional_info)

    print(f"Saved summary to: {eval_dir / 'summary.json'}")
    print(f"Saved additional details to: {eval_dir / 'additional_info.json'}")
    print(f"Updated metrics registry: {full_ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model from a run directory."
    )
    parser.add_argument(
        "dir",
        type=str,
        help="Run directory containing config.yaml and ckpt/step=N.pth files.",
    )
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=4,
        help="Cap the number of evaluation samples (useful for quick checks).",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="song",
        choices=["segment", "song"],
        help="segment: random clips, fast; song: full audio, chunked inference.",
    )
    parser.add_argument(
        "--use_train",
        action="store_true",
        help="Use the training split instead of the test split.",
    )
    args = parser.parse_args()
    main_func(args)


if __name__ == "__main__":
    main()
