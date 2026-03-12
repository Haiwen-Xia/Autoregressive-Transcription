from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import hydra
import soundfile as sf
import torch
from audidata.collate.default import collate_fn
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from audio_understanding.eval.transcription.batch_eval import batch_evaluate, _evaluate_cropped_item
from audio_understanding.eval.transcription.metrics import instrument_summary
from audio_understanding.utils import remove_padded_columns
from inference_transcription import tokens_to_midi, transcribe_audio
from train import (
    ce_loss,
    get_audio_encoder,
    get_audio_question_answering,
    get_dataset,
    get_learnable_params,
    get_llm,
    get_optimizer_and_scheduler,
    get_tokenizer,
)


class DebugFixedDataset(Dataset):
    def __init__(self, base_dataset: Any, fixed_items: list[dict]) -> None:
        self.base_dataset = base_dataset
        self.fixed_items = fixed_items

    def __len__(self) -> int:
        return len(self.fixed_items)

    def __getitem__(self, idx: int) -> dict:
        return self.fixed_items[idx]

    def evaluate(self, *args, **kwargs):
        return self.base_dataset.evaluate(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.base_dataset, name)


def _setup_output_and_logger(configs: dict, script_name: str) -> tuple[Path, Path, logging.Logger]:
    root_dir = Path(get_original_cwd())

    output_root_value = configs.get("output_root", "./checkpoints/train")
    output_root = Path(output_root_value)
    if not output_root.is_absolute():
        output_root = (root_dir / output_root).resolve()

    run_name = configs.get("run_name")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    processed_run_name = f"{timestamp}_{run_name}" if run_name else timestamp

    output_dir = output_root / "debug_train_eval" / processed_run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / f"{script_name}.log"
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return output_dir, log_path, logger


def _build_debug_batch(
    fixed_items: list[dict],
    indices: list[int],
) -> dict:
    items = [dict(fixed_items[idx]) for idx in indices]
    return cast(dict, collate_fn(items))


def _train_one_step(
    configs: dict,
    batch: dict,
    audio_encoder,
    llm,
    tokenizer,
    device: str,
    gradient_accumulation: int,
) -> float:
    audio_encoder.train()
    llm.train()

    audio, question, answering = get_audio_question_answering(batch)
    audio = audio.to(device)

    audio_latent = cast(Any, audio_encoder).encode(
        audio=audio,
        train_mode=configs["audio_encoder"]["trainable"],
    )

    question_ids = cast(Any, tokenizer).texts_to_ids(
        texts=question,
        fix_length=configs["max_question_len"],
    ).to(device)

    answering_ids = cast(Any, tokenizer).texts_to_ids(
        texts=answering,
        fix_length=configs["max_answering_len"],
    ).to(device)

    if configs["train"]["remove_padded_columns"]:
        answering_ids = remove_padded_columns(
            ids=answering_ids,
            pad_token_id=cast(Any, tokenizer).pad_token_id,
        )

    seqs = [audio_latent, question_ids, answering_ids]
    seq_types = ["audio", "id", "id"]
    loss_types = [None, None, "ce"]

    output_seqs = llm(seqs=seqs, seq_types=seq_types, mask=None)
    output_seqs = [seq[:, 0:-1] for seq in output_seqs]
    target_seqs = [seq[:, 1:] for seq in seqs]

    loss = ce_loss(
        output_seqs=output_seqs,
        target_seqs=target_seqs,
        loss_types=loss_types,
        ignore_index=cast(Any, tokenizer).pad_token_id,
    )

    (loss / gradient_accumulation).backward()

    return float(loss.item())


def _make_inference_fn(audio_encoder, llm, tokenizer, configs: dict, device: str):
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


def _is_metric_close_to_one(value: Any, atol: float = 1e-3) -> bool:
    if isinstance(value, dict):
        if {"precision", "recall", "f1"}.issubset(value.keys()):
            return (
                abs(float(value["precision"]) - 1.0) <= atol
                and abs(float(value["recall"]) - 1.0) <= atol
                and abs(float(value["f1"]) - 1.0) <= atol
            )
        return all(_is_metric_close_to_one(v, atol=atol) for v in value.values())
    if isinstance(value, list):
        return all(_is_metric_close_to_one(v, atol=atol) for v in value)
    return True


def _avg_metric_dict(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    if len(metrics_list) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": float(sum(m["precision"] for m in metrics_list) / len(metrics_list)),
        "recall": float(sum(m["recall"] for m in metrics_list) / len(metrics_list)),
        "f1": float(sum(m["f1"] for m in metrics_list) / len(metrics_list)),
    }


def _evaluate_debug_dataset_cropped(
    fixed_items: list[dict],
    fps: float,
    include_program: bool,
) -> dict[str, Any]:
    onset_list = []
    offset_list = []
    program_list = []
    per_inst_list = []

    empty_count = 0
    for sample_i, data in enumerate(fixed_items):
        gt_tokens = cast(list[str], data["token"])
        result = _evaluate_cropped_item(
            data=data,
            output_tokens=gt_tokens,
            fps=fps,
            include_program=include_program,
        )
        diag = result.get("_diag", {})
        ref_n = diag.get("ref_note_count", 0)
        est_n = diag.get("est_note_count", 0)
        onset_m = result["note_onset"]
        offset_m = result["note_offset"]
        # Skip empty samples (both ref and est empty → correct prediction, not F1=0)
        is_empty = ref_n == 0 and est_n == 0
        tag = " [EMPTY->skip]" if is_empty else ""
        print(
            f"[GT roundtrip] sample={sample_i} | "
            f"raw_notes={diag.get('raw_note_count')} -> "
            f"ref_notes={ref_n} | "
            f"est_notes={est_n} | "
            f"gt_tokens={diag.get('est_token_count')} | "
            f"start={diag.get('start_time', 0):.3f} dur={diag.get('duration', 0):.3f} | "
            f"onset_f1={onset_m['f1']:.4f} offset_f1={offset_m['f1']:.4f}{tag}"
        )
        if is_empty:
            empty_count += 1
            continue
        onset_list.append(cast(dict[str, float], onset_m))
        offset_list.append(cast(dict[str, float], offset_m))
        if include_program and "program_aware" in result:
            program_list.append(cast(dict[str, float], result["program_aware"]))
        if include_program and "per_instrument" in result:
            per_inst_list.append(cast(dict, result["per_instrument"]))

    summary: dict[str, Any] = {
        "n_samples": len(fixed_items),
        "empty_ref_skipped": empty_count,
        "note_onset": _avg_metric_dict(onset_list),
        "note_offset": _avg_metric_dict(offset_list),
    }
    if include_program:
        summary["program_aware"] = _avg_metric_dict(program_list)
        if per_inst_list:
            summary["instrument_summary"] = instrument_summary(per_inst_list)
    return summary


def _run_full_debug_on_items(
    full_items: list[dict],
    fps: float,
    include_program: bool,
) -> dict[str, Any]:
    return _evaluate_debug_dataset_cropped(
        fixed_items=full_items,
        fps=fps,
        include_program=include_program,
    )


def _dump_target_and_output_midis(
    fixed_items: list[dict],
    dataset_name: str,
    audio_encoder,
    llm,
    tokenizer,
    configs: dict,
    device: str,
    output_dir: Path,
    stage_name: str,
    max_dump_samples: int,
    logger: logging.Logger,
) -> None:
    include_program = bool(configs.get("midi_include_program", False))
    write_program_tracks = bool(configs.get("midi_write_program_tracks", include_program))
    stage_dir = output_dir / "midi_dump" / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    sr = int(configs["sample_rate"])

    sample_metrics: list[dict[str, Any]] = []

    n_dump = min(max_dump_samples, len(fixed_items))
    logger.info(
        "Dump MIDI pairs (%s): n_samples=%d, dataset=%s, include_program=%s",
        stage_name,
        n_dump,
        dataset_name,
        include_program,
    )

    for i in range(n_dump):
        data = fixed_items[i]

        target_tokens = data["token"]
        target_midi_path = stage_dir / f"sample_{i:02d}_target.mid"
        tokens_to_midi(
            tokens=target_tokens,
            fps=configs["fps"],
            output_path=str(target_midi_path),
            include_program=include_program,
            write_program_tracks=write_program_tracks,
        )

        audio = data["audio"]
        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio)
        audio_tensor = audio.unsqueeze(0).to(device)
        question = data.get("question", "Music transcription.")

        audio_np = audio.squeeze().cpu().numpy()
        sample_audio_path = stage_dir / f"sample_{i:02d}_audio.wav"
        sf.write(str(sample_audio_path), audio_np, sr)

        output_midi_path = stage_dir / f"sample_{i:02d}_output.mid"
        output = transcribe_audio(
            audio_encoder=audio_encoder,
            llm=llm,
            tokenizer=tokenizer,
            audio_tensor=audio_tensor,
            configs=configs,
            output_midi_path=str(output_midi_path),
            question=question,
            temperature=1.0,
            top_k=1,
        )
        output_tokens = cast(list[str], output["tokens"])

        metrics = _evaluate_cropped_item(
            data=data,
            output_tokens=output_tokens,
            fps=float(configs["fps"]),
            include_program=include_program,
        )

        sample_metric = {
            "sample_index": i,
            "target_midi_path": str(target_midi_path),
            "output_midi_path": str(output_midi_path),
            "audio_segment_path": str(sample_audio_path),
            "metrics": _keep_only_f1_metrics(metrics),
        }
        sample_metrics.append(sample_metric)

        metadata = {
            "sample_index": i,
            "question": question,
            "start_time": float(data["start_time"]),
            "duration": float(data["duration"]),
            "audio_path": str(data.get("audio_path", "")),
            "audio_paths": [str(x) for x in data.get("audio_paths", [])],
            "midi_path": str(data.get("midi_path", "")),
            "midi_paths": [str(x) for x in data.get("midi_paths", [])],
            "input_track_ids": [str(x) for x in data.get("input_track_ids", [])],
            "target_track_ids": [str(x) for x in data.get("target_track_ids", [])],
            "target_tokens": data.get("token", []),
            "output_tokens": output_tokens,
        }
        metadata_path = stage_dir / f"sample_{i:02d}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as fw:
            json.dump(metadata, fw, indent=2, ensure_ascii=False)

        logger.info(
            "MIDI+metrics saved (%s, sample=%d): target=%s output=%s metrics=%s metadata=%s",
            stage_name,
            i,
            target_midi_path,
            output_midi_path,
            json.dumps(sample_metric["metrics"], ensure_ascii=False),
            metadata_path,
        )

    sample_metrics_path = stage_dir / "sample_metrics.json"
    with open(sample_metrics_path, "w", encoding="utf-8") as fw:
        json.dump(sample_metrics, fw, indent=2, ensure_ascii=False)
    logger.info("Saved sample metrics (%s): %s", stage_name, sample_metrics_path)


def _keep_only_f1_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return metrics #* no need to be not verbose


def main_func(cfg: DictConfig) -> None:
    script_name = Path(__file__).stem
    configs = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(configs, dict)

    output_dir, log_path, logger = _setup_output_and_logger(configs, script_name)
    cfg_save_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, cfg_save_path)

    device = configs["train"]["device"]

    train_dataset = cast(Any, get_dataset(configs, split="train", use_crop=True))
    train_dataset_name = type(train_dataset).__name__
    if train_dataset_name == "MAESTRO":
        dataset_name = "MAESTRO"
    elif train_dataset_name == "Slakh2100":
        dataset_name = "Slakh2100"
    else:
        dataset_name = train_dataset_name

    dataset = train_dataset
    assert dataset_name in ("MAESTRO", "Slakh2100"), f"Only transcription datasets are supported, got {dataset_name}"
    assert len(dataset) >= 10, f"Dataset has only {len(dataset)} samples, need at least 10"

    subset_indices = list(range(10))
    fixed_items = [cast(dict, dataset[i]) for i in subset_indices]
    fixed_dataset = DebugFixedDataset(base_dataset=dataset, fixed_items=fixed_items)
    logger.info("Dataset=%s total=%d debug_subset=%d", dataset_name, len(dataset), len(fixed_dataset))


    include_program = bool(configs.get("midi_include_program", False))
    logger.info("Eval setup: dataset=%s include_program=%s", dataset_name, include_program)



    logger.info("Run debug_dataset eval with GT tokens under cropped preprocess semantics")
    debug_dataset_eval_small = _evaluate_debug_dataset_cropped(
        fixed_items=fixed_items,
        fps=float(configs["fps"]),
        include_program=include_program,
    )
    logger.info("Eval(debug_dataset, gt->pred): %s", json.dumps(debug_dataset_eval_small, ensure_ascii=False))
    if not _is_metric_close_to_one(debug_dataset_eval_small):
        logger.warning(
            "debug_dataset eval is not perfect under cropped semantics."
        )

    full_debug_summary = None
    if bool(configs.get("debug", {}).get("full_debug", True)):
        logger.info("Run full_debug on non-cropped dataset (separate from crop dataset)")
        full_dataset = cast(Any, get_dataset(configs, split="train", use_crop=False))
        full_debug_samples = int(configs.get("debug", {}).get("full_debug_samples", 1))
        full_items = [cast(dict, full_dataset[i]) for i in range(min(full_debug_samples, len(full_dataset)))]

        full_debug_summary = _run_full_debug_on_items(
            full_items=full_items,
            fps=float(configs["fps"]),
            include_program=include_program,
        )
        logger.info("Eval(full_debug): %s", json.dumps(full_debug_summary, ensure_ascii=False))

    if bool(configs.get("debug", {}).get("only_dataset_checks", False)):
        summary = {
            "dataset": dataset_name,
            "subset_indices": subset_indices,
            "eval_debug_dataset": debug_dataset_eval_small,
            "eval_full_debug": full_debug_summary,
        }
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as fw:
            json.dump(summary, fw, indent=2, ensure_ascii=False)
        logger.info("Dataset-only debug finished.")
        logger.info("Saved config: %s", cfg_save_path)
        logger.info("Saved log: %s", log_path)
        logger.info("Saved summary: %s", summary_path)
        return

    resume_ckpt_path = configs["train"].get("resume_ckpt_path", "")
    audio_encoder = get_audio_encoder(configs=configs, ckpt_path=resume_ckpt_path).to(device)
    tokenizer = get_tokenizer(configs=configs)
    llm = get_llm(
        configs=configs,
        audio_latent_dim=int(getattr(cast(Any, audio_encoder), "latent_dim")),
        vocab_size=len(cast(Any, tokenizer)),
        ckpt_path=resume_ckpt_path,
    ).to(device)

    inference_fn = _make_inference_fn(
        audio_encoder=audio_encoder,
        llm=llm,
        tokenizer=tokenizer,
        configs=configs,
        device=device,
    )
    logger.info("Run eval on subset before train")
    params = get_learnable_params(configs=configs, audio_encoder=audio_encoder, llm=llm)
    assert len(params) > 0, "No trainable parameters. Check audio_encoder.trainable / llm.trainable"
    optimizer, scheduler = get_optimizer_and_scheduler(configs=configs, params=params)

    
    audio_encoder.eval()
    llm.eval()
    pre_eval = batch_evaluate(
        dataset=fixed_dataset,
        inference_fn=inference_fn,
        fps=configs["fps"],
        include_program=include_program,
        max_samples=len(fixed_dataset),
        epochs=1,
        skip_empty_ref_for_averaging=(getattr(fixed_dataset, "mode", None) == "single"),
        verbose=True,
    )
    pre_eval_small = _keep_only_f1_metrics(pre_eval)
    logger.info("Eval(before train): %s", json.dumps(pre_eval_small, ensure_ascii=False))

    batch_size = int(configs["train"]["batch_size_per_device"])
    assert batch_size > 0
    gradient_accumulation = int(configs["train"].get("gradient_accumulation", 1))
    assert gradient_accumulation > 0
    train_steps = int(configs.get("debug", {}).get("train_steps", 3000))
    assert train_steps > 0

    optimizer.zero_grad()
    logger.info("Run train on same subset: steps=%d", train_steps)
    step = 0
    micro_step = 0
    while step < train_steps:
        order = torch.randperm(len(fixed_items)).tolist()
        for begin in range(0, len(order), batch_size):
            idx_list = order[begin : begin + batch_size]
            train_batch = _build_debug_batch(fixed_items=fixed_items, indices=idx_list)
            loss = _train_one_step(
                configs=configs,
                batch=train_batch,
                audio_encoder=audio_encoder,
                llm=llm,
                tokenizer=tokenizer,
                device=device,
                gradient_accumulation=gradient_accumulation,
            )

            micro_step += 1
            if micro_step % gradient_accumulation != 0:
                continue

            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

            step += 1
            if step % 20 == 0:
                logger.info("Train step=%d/%d loss=%.6f", step, train_steps, loss)
            if step >= train_steps:
                break

    logger.info("Run eval on same subset after train")
    audio_encoder.eval()
    llm.eval()
    post_eval = batch_evaluate(
        dataset=fixed_dataset,
        inference_fn=inference_fn,
        fps=configs["fps"],
        include_program=include_program,
        max_samples=len(fixed_dataset),
        epochs=1,
        skip_empty_ref_for_averaging=(getattr(fixed_dataset, "mode", None) == "single"),
        verbose=True,
    )
    post_eval_small = _keep_only_f1_metrics(post_eval)
    logger.info("Eval(after train): %s", json.dumps(post_eval_small, ensure_ascii=False))

    _dump_target_and_output_midis(
        fixed_items=fixed_items,
        dataset_name=dataset_name,
        audio_encoder=audio_encoder,
        llm=llm,
        tokenizer=tokenizer,
        configs=configs,
        device=device,
        output_dir=output_dir,
        stage_name="after_train",
        max_dump_samples=5,
        logger=logger,
    )

    summary = {
        "dataset": dataset_name,
        "subset_indices": subset_indices,
        "train_steps": train_steps,
        "eval_debug_dataset": debug_dataset_eval_small,
        "eval_full_debug": full_debug_summary,
        "eval_before_train": pre_eval_small,
        "eval_after_train": post_eval_small,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fw:
        json.dump(summary, fw, indent=2, ensure_ascii=False)

    logger.info("Saved config: %s", cfg_save_path)
    logger.info("Saved log: %s", log_path)
    logger.info("Saved summary: %s", summary_path)


@hydra.main(version_base=None, config_path="configs", config_name="maestro")
def main(cfg: DictConfig) -> None:
    main_func(cfg)


if __name__ == "__main__":
    main()
