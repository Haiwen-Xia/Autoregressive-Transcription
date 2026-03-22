from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from audidata.collate.default import collate_fn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from audio_understanding.data.samplers import InfiniteSampler
from audio_understanding.eval.transcription.onset_only_eval import batch_evaluate_onset
from audio_understanding.utils import remove_padded_columns
from train import (
    _count_model_params,
    _setup_output_and_logger,
    ce_loss,
    get_audio_encoder,
    get_learnable_params,
    get_llm,
    get_optimizer_and_scheduler,
)


def get_dataset(
    configs: dict,
    split: str,
    use_crop: bool = True,
) -> Dataset:
    from audidata.io.crops import RandomCrop
    from audidata.transforms import Mono

    from audio_understanding.target_transforms.midi_onset import MIDI2OnsetTokens

    sr = configs["sample_rate"]
    clip_duration = configs["clip_duration"]
    tokenizer_cfg = configs["tokenizer"] if "tokenizer" in configs else {}
    drum_pitch = bool(tokenizer_cfg.get("drum_pitch", False))
    datasets_split = f"{split}_datasets"

    datasets = []

    for name in configs[datasets_split].keys():
        if name == "MAESTRO":
            from audio_understanding.datasets.maestro import MAESTRO

            assert configs["midi_to_tokens"] == "MIDI2OnsetTokens"
            midi_transform = MIDI2OnsetTokens(
                fps=configs["fps"],
                drum_pitch=drum_pitch,
            )

            dataset = MAESTRO(
                root=configs[datasets_split][name]["root"],
                split=configs[datasets_split][name]["split"],
                sr=sr,
                crop=RandomCrop(clip_duration=clip_duration, end_pad=clip_duration - 0.1) if use_crop else None,
                transform=Mono(),
                load_target=True,
                extend_pedal=True,
                include_program=False,
                target_transform=midi_transform,
            )
            datasets.append(dataset)
            continue

        if name == "Slakh2100":
            from audio_understanding.datasets.slakh2100 import Slakh2100

            assert configs["midi_to_tokens"] == "MIDI2OnsetTokens"
            midi_transform = MIDI2OnsetTokens(
                fps=configs["fps"],
                drum_pitch=drum_pitch,
            )

            dataset_config = configs[datasets_split][name]
            mode = dataset_config["mode"] if "mode" in dataset_config else "all"
            sample_num = dataset_config["sample_num"] if "sample_num" in dataset_config else 1
            keep_track_info = dataset_config["keep_track_info"] if "keep_track_info" in dataset_config else False
            question_config = dataset_config["question"] if "question" in dataset_config else None

            dataset = Slakh2100(
                root=dataset_config["root"],
                split=dataset_config["split"],
                sr=sr,
                crop=RandomCrop(clip_duration=clip_duration, end_pad=clip_duration - 0.1) if use_crop else None,
                transform=Mono(),
                target=True,
                extend_pedal=True,
                target_transform=midi_transform,
                sample_num=sample_num,
                mode=mode,
                include_drum=True,
                keep_track_info=keep_track_info,
                question_config=question_config,
            )
            datasets.append(dataset)
            continue

        raise ValueError(name)

    if len(datasets) == 1:
        return datasets[0]

    raise ValueError("Do not support multiple datasets in this file.")


def get_tokenizer(configs: dict) -> Any:
    name = configs["tokenizer"]["name"]
    drum_pitch = bool(configs["tokenizer"].get("drum_pitch", False))
    if name == "BertOnset":
        from audio_understanding.tokenizers.bert_onset import BertOnset

        return BertOnset(drum_pitch=drum_pitch)
    raise ValueError(name)


def get_audio_question_answering(data: dict) -> tuple[torch.Tensor, list[str], list[str]]:
    name = data["dataset_name"][0]
    if name in ["MAESTRO", "Slakh2100"]:
        return data["audio"], data["question"], data["token"]
    raise ValueError(name)


def validate(
    configs: dict,
    dataset: Any,
    audio_encoder: nn.Module,
    tokenizer: Any,
    llm: nn.Module,
    valid_steps: int = 50,
) -> float:
    device = next(audio_encoder.parameters()).device
    losses = []

    batch_size = configs["train"]["batch_size_per_device"]
    skip_n = max(1, len(dataset) // valid_steps)

    for idx in range(0, len(dataset), skip_n):
        data = [dataset[i] for i in range(idx, min(idx + batch_size, len(dataset)))]
        data = cast(dict, collate_fn(data))

        audio, question, answering = get_audio_question_answering(data)

        audio = audio.to(device)
        audio_latent = cast(Any, audio_encoder).encode(audio=audio, train_mode=False)

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

        with torch.no_grad():
            llm.eval()
            output_seqs = llm(
                seqs=seqs,
                seq_types=seq_types,
                mask=None,
            )

        output_seqs = [seq[:, 0:-1] for seq in output_seqs]
        target_seqs = [seq[:, 1:] for seq in seqs]

        loss = ce_loss(
            output_seqs=output_seqs,
            target_seqs=target_seqs,
            loss_types=loss_types,
            ignore_index=cast(Any, tokenizer).pad_token_id,
        )

        losses.append(loss.item())

    return float(np.mean(losses))


def _decode_generated_token_ids(tokenizer: Any, token_ids: list[int]) -> list[str]:
    tok = tokenizer.tok
    ids = token_ids
    if len(ids) > 0 and ids[0] == tok.cls_token_id:
        ids = ids[1:]
    if tok.sep_token_id in ids:
        ids = ids[: ids.index(tok.sep_token_id)]
    return cast(list[str], tok.convert_ids_to_tokens(ids))


def _generate_onset_tokens(
    item: dict,
    audio_encoder: nn.Module,
    llm: nn.Module,
    tokenizer: Any,
    configs: dict,
    device: str,
) -> list[str]:
    audio = item["audio"]
    if not isinstance(audio, torch.Tensor):
        audio = torch.as_tensor(audio)

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    audio_tensor = audio.unsqueeze(0).to(device)
    question = item.get("question", "Music onset transcription.")

    with torch.no_grad():
        audio_latent = cast(Any, audio_encoder).encode(audio=audio_tensor, train_mode=False)

    question_ids = cast(Any, tokenizer).texts_to_ids(
        texts=[question],
        fix_length=configs["max_question_len"],
    ).to(device)

    answering_ids = torch.LongTensor([[cast(Any, tokenizer).cls_token_id]]).to(device)

    seqs = [audio_latent, question_ids, answering_ids]
    seq_types = ["audio", "id", "id"]

    was_training = llm.training
    llm.eval()
    generate_fn = getattr(llm, "generate")
    with torch.no_grad():
        output_seqs = generate_fn(  # type: ignore[reportCallIssue]
            seqs=seqs,
            seq_types=seq_types,
            max_new_ids=configs["max_answering_len"],
            temperature=1.0,
            top_k=1,
        )
    if was_training:
        llm.train()

    token_ids = cast(list[int], output_seqs[2][0].detach().cpu().tolist())
    return _decode_generated_token_ids(tokenizer=tokenizer, token_ids=token_ids)


def _log_onset_samples_and_eval(
    configs: dict,
    dataset,
    audio_encoder: nn.Module,
    tokenizer: Any,
    llm: nn.Module,
    logger: logging.Logger,
    device: str,
    split_name: str,
    run_batch_eval: bool,
) -> dict[str, float]:
    n_samples = int(configs["train"].get("onset_preview_samples", 2))
    skip_n = max(1, len(dataset) // n_samples)
    sample_indices = list(range(0, len(dataset), skip_n))[:n_samples]

    for idx in sample_indices:
        item = dataset[idx]
        gt_tokens = cast(list[str], item.get("token", []))
        pred_tokens = _generate_onset_tokens(
            item=item,
            audio_encoder=audio_encoder,
            llm=llm,
            tokenizer=tokenizer,
            configs=configs,
            device=device,
        )
        logger.info(
            "Onset sample split=%s idx=%d | gt_tokens=%d pred_tokens=%d",
            split_name,
            idx,
            len(gt_tokens),
            len(pred_tokens),
        )
        logger.info("  GT preview: %s", " ".join(gt_tokens[:20]))
        logger.info("  PR preview: %s", " ".join(pred_tokens[:20]))

    if not run_batch_eval:
        return {}

    eval_max_samples = None
    if "transcription_eval_max_samples" in configs["train"]:
        eval_max_samples = int(configs["train"]["transcription_eval_max_samples"])

    def inference_fn(item: dict) -> list[str]:
        return _generate_onset_tokens(
            item=item,
            audio_encoder=audio_encoder,
            llm=llm,
            tokenizer=tokenizer,
            configs=configs,
            device=device,
        )

    summary = batch_evaluate_onset(
        dataset=dataset,
        inference_fn=inference_fn,
        fps=float(configs["fps"]),
        max_samples=eval_max_samples,
        verbose=False,
    )

    onset_f1 = float(summary["onset_time"]["f1"])
    onset_p = float(summary["onset_time"]["precision"])
    onset_r = float(summary["onset_time"]["recall"])

    logger.info(
        "OnsetEval split=%s | onset_time_f1=%.6f precision=%.6f recall=%.6f",
        split_name,
        onset_f1,
        onset_p,
        onset_r,
    )

    return {
        "onset_eval/f1": onset_f1,
        "onset_eval/precision": onset_p,
        "onset_eval/recall": onset_r,
    }


def main_func(cfg: DictConfig) -> None:
    script_name = Path(__file__).stem
    configs = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(configs, dict)

    output_dir, ckpt_dir, _log_path, logger = _setup_output_and_logger(configs, script_name)

    cfg_save_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, cfg_save_path)

    wandb_log = not bool(configs.get("no_log", False))
    if wandb_log:
        wandb.init(
            project="audio_understanding",
            group=output_dir.parent.name,
            name=str(output_dir.name),
            config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
        )
        wandb.save(str(cfg_save_path))

    device = configs["train"]["device"]

    train_dataset = cast(Any, get_dataset(configs, split="train"))
    test_dataset = cast(Any, get_dataset(configs, split="test"))

    train_sampler = InfiniteSampler(train_dataset)
    num_workers = configs["train"]["num_workers"]
    dataloader_kwargs = {
        "dataset": train_dataset,
        "batch_size": configs["train"]["batch_size_per_device"],
        "sampler": train_sampler,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": True,
    }

    if num_workers > 0:
        dataloader_kwargs.update(
            {
                "multiprocessing_context": "spawn",
                "persistent_workers": True,
                "prefetch_factor": 2,
                "timeout": 120,
            }
        )

    train_dataloader = DataLoader(**dataloader_kwargs)

    audio_encoder = get_audio_encoder(
        configs=configs,
        ckpt_path=configs["train"]["resume_ckpt_path"],
    ).to(device)

    tokenizer = get_tokenizer(configs=configs)

    llm = get_llm(
        configs=configs,
        audio_latent_dim=cast(Any, audio_encoder).latent_dim,
        vocab_size=len(cast(Any, tokenizer)),
        ckpt_path=configs["train"]["resume_ckpt_path"],
        audio_encoder=audio_encoder,
        tokenizer=tokenizer,
    ).to(device)

    params = get_learnable_params(configs, audio_encoder, llm)
    optimizer, scheduler = get_optimizer_and_scheduler(configs=configs, params=params)

    audio_total, audio_trainable = _count_model_params(audio_encoder)
    llm_total, llm_trainable = _count_model_params(llm)
    logger.info(
        "Audio encoder params total=%d M trainable=%d M | LLM params total=%d M trainable=%d M",
        audio_total // 1024**2,
        audio_trainable // 1024**2,
        llm_total // 1024**2,
        llm_trainable // 1024**2,
    )

    gradient_accumulation = (
        configs["train"]["gradient_accumulation"] if "gradient_accumulation" in configs["train"] else 1
    )
    assert gradient_accumulation > 0
    grad_clip_norm = float(configs["train"]["grad_clip_norm"]) if "grad_clip_norm" in configs["train"] else None

    global_step = 0
    optimizer.zero_grad()

    for micro_step, data in enumerate(tqdm(train_dataloader)):
        audio, question, answering = get_audio_question_answering(data)

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

        llm.train()
        output_seqs = llm(
            seqs=seqs,
            seq_types=seq_types,
            mask=None,
        )

        output_seqs = [seq[:, 0:-1] for seq in output_seqs]
        target_seqs = [seq[:, 1:] for seq in seqs]

        loss = ce_loss(
            output_seqs=output_seqs,
            target_seqs=target_seqs,
            loss_types=loss_types,
            ignore_index=cast(Any, tokenizer).pad_token_id,
        )

        (loss / gradient_accumulation).backward()

        if (micro_step + 1) % gradient_accumulation != 0:
            continue

        grad_norm_value = None
        if grad_clip_norm is not None:
            params_to_clip = [
                p for p in list(audio_encoder.parameters()) + list(llm.parameters()) if p.requires_grad
            ]
            grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip_norm)
            grad_norm_value = float(grad_norm.item())

        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

        if scheduler:
            scheduler.step()

        if global_step % 100 == 0:
            if grad_norm_value is None:
                logger.info("Step: %d, Loss: %.6f", global_step, loss.item())
            else:
                logger.info(
                    "Step: %d, Loss: %.6f, GradNorm: %.6f",
                    global_step,
                    loss.item(),
                    grad_norm_value,
                )

            if wandb_log:
                payload = {"train_loss_step": loss.item()}
                if grad_norm_value is not None:
                    payload["grad_norm"] = grad_norm_value
                wandb.log(data=payload, step=global_step)

        if global_step % configs["train"]["test_every_n_steps"] == 0:
            train_loss = validate(
                configs=configs,
                dataset=train_dataset,
                audio_encoder=audio_encoder,
                tokenizer=tokenizer,
                llm=llm,
            )
            test_loss = validate(
                configs=configs,
                dataset=test_dataset,
                audio_encoder=audio_encoder,
                tokenizer=tokenizer,
                llm=llm,
            )

            logger.info("Train loss: %.6f", train_loss)
            logger.info("Test loss: %.6f", test_loss)

            if wandb_log:
                wandb.log(
                    data={"train_loss": train_loss, "test_loss": test_loss},
                    step=global_step,
                )

        if global_step > 0 and global_step % configs["train"]["save_every_n_steps"] == 0:
            ckpt_path = ckpt_dir / f"step={global_step}.pth"
            ckpt = {}

            if configs["audio_encoder"]["trainable"]:
                ckpt["audio_encoder"] = audio_encoder.state_dict()

            if configs["llm"]["trainable"]:
                ckpt["llm"] = llm.state_dict()

            torch.save(ckpt, ckpt_path)
            logger.info("Save model to %s", ckpt_path)
        if global_step > 0 and global_step * 5 % configs["train"]["save_every_n_steps"] == 0:
            logger.info("Preview onset samples on train split")
            _log_onset_samples_and_eval(
                configs=configs,
                dataset=train_dataset,
                audio_encoder=audio_encoder,
                tokenizer=tokenizer,
                llm=llm,
                logger=logger,
                device=device,
                split_name="train",
                run_batch_eval=False,
            )

            logger.info("Run onset-only eval on test split")
            test_metrics = _log_onset_samples_and_eval(
                configs=configs,
                dataset=test_dataset,
                audio_encoder=audio_encoder,
                tokenizer=tokenizer,
                llm=llm,
                logger=logger,
                device=device,
                split_name="test",
                run_batch_eval=True,
            )

            if wandb_log:
                wandb.log(data=test_metrics, step=global_step)

        if global_step == configs["train"]["training_steps"]:
            break


@hydra.main(version_base=None, config_path="configs", config_name="maestro_onset")
def main(cfg: DictConfig) -> None:
    main_func(cfg)


if __name__ == "__main__":
    main()
