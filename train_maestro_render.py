"""MAESTRO-rendering training script.

Same training loop as train_random.py, but clips real MIDI from MAESTRO
train split and renders with the same VST engine. This isolates the effect
of real vs. synthetic note distributions while keeping audio quality identical.

Flow per round:
    1. render_epoch → sample MAESTRO pieces, clip 5s, render with VST
    2. shuffle + iterate epoch repeat_times times
    3. repeat
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from audidata.collate.default import collate_fn
from audio_understanding.random_rendering.dataset import MaestroRenderingBuffer, make_token_fn
from audio_understanding.random_rendering.rendering import DawDreamerSampler, Sampler as RenderEngine
from audio_understanding.utils import LinearWarmUp, remove_padded_columns
from train import (
    _count_model_params,
    _setup_output_and_logger,
    ce_loss,
    get_audio_encoder,
    get_dataset,
    get_learnable_params,
    get_llm,
    get_optimizer_and_scheduler,
    get_tokenizer,
)
from train_random import (
    get_audio_question_answering,
    get_token_type_ranges,
    _masked_ce,
    validate_maestro,
)


def validate_rendered_maestro(
    configs: dict,
    buffer: MaestroRenderingBuffer,
    audio_encoder,
    tokenizer,
    llm,
    token_type_ranges: dict,
    valid_size: int = 64,
) -> dict:
    """Validate on freshly rendered MAESTRO clips, returning per-token-type CE losses.

    Same structure as validate_random in train_random.py but uses MaestroRenderingBuffer.
    """
    device = next(audio_encoder.parameters()).device
    batch_size = configs["train"]["batch_size_per_device"]
    pad_id = tokenizer.pad_token_id

    onset_id = token_type_ranges["name_onset"][0]
    offset_id = token_type_ranges["name_offset"][0]
    time_lo, time_hi = token_type_ranges["time_index"]
    pitch_lo, pitch_hi = token_type_ranges["pitch"]
    vel_lo, vel_hi = token_type_ranges["velocity"]

    buffer.render_epoch(valid_size, seed=999999)

    accum = {k: [] for k in ["total", "pitch", "onset", "offset", "velocity"]}

    for data in buffer.iterate_epoch(batch_size, repeat=1):
        audio, question, answering = get_audio_question_answering(data)
        audio = audio.to(device)

        audio_latent = audio_encoder.encode(audio=audio, train_mode=False)
        question_ids = tokenizer.texts_to_ids(texts=question, fix_length=configs["max_question_len"]).to(device)
        answering_ids = tokenizer.texts_to_ids(texts=answering, fix_length=configs["max_answering_len"]).to(device)

        if configs["train"]["remove_padded_columns"]:
            answering_ids = remove_padded_columns(ids=answering_ids, pad_token_id=pad_id)

        seqs = [audio_latent, question_ids, answering_ids]
        seq_types = ["audio", "id", "id"]

        with torch.no_grad():
            llm.eval()
            output_seqs = llm(seqs=seqs, seq_types=seq_types, mask=None)

        pred_logits = output_seqs[-1][:, :-1]
        targets = answering_ids[:, 1:]

        B, L = targets.shape
        flat_logits = pred_logits.reshape(B * L, -1)
        flat_targets = targets.reshape(B * L)

        total = F.cross_entropy(flat_logits, flat_targets, ignore_index=pad_id)
        accum["total"].append(total.item())

        is_time = (flat_targets >= time_lo) & (flat_targets <= time_hi)
        is_pitch = (flat_targets >= pitch_lo) & (flat_targets <= pitch_hi)
        is_vel = (flat_targets >= vel_lo) & (flat_targets <= vel_hi)

        next_targets = torch.zeros_like(flat_targets)
        if L > 1:
            next_2d = torch.cat([targets[:, 1:], torch.zeros(B, 1, dtype=targets.dtype, device=device)], dim=1)
            next_targets = next_2d.reshape(B * L)

        accum["pitch"].append(_masked_ce(flat_logits, flat_targets, is_pitch, pad_id).item())
        accum["onset"].append(_masked_ce(flat_logits, flat_targets, is_time & (next_targets == onset_id), pad_id).item())
        accum["offset"].append(_masked_ce(flat_logits, flat_targets, is_time & (next_targets == offset_id), pad_id).item())
        accum["velocity"].append(_masked_ce(flat_logits, flat_targets, is_vel, pad_id).item())

    return {f"{k}_ce": float(np.mean(v)) if v else 0.0 for k, v in accum.items()}


def main_func(cfg: DictConfig) -> None:
    script_name = Path(__file__).stem
    configs = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(configs, dict)

    output_dir, ckpt_dir, log_path, logger = _setup_output_and_logger(configs, script_name)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    wandb_log = not bool(configs.get("no_log", False))
    if wandb_log:
        wandb.init(
            project="audio_understanding",
            group="maestro_rendering",
            name=str(output_dir.name),
            config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
        )

    device = configs["train"]["device"]

    # ---- model (identical to train_random.py) ----
    audio_encoder = cast(Any, get_audio_encoder(
        configs=configs,
        ckpt_path=configs["train"]["resume_ckpt_path"],
    )).to(device)

    tokenizer = cast(Any, get_tokenizer(configs=configs))

    llm = cast(Any, get_llm(
        configs=configs,
        audio_latent_dim=audio_encoder.latent_dim,
        vocab_size=len(tokenizer),
        ckpt_path=configs["train"]["resume_ckpt_path"],
        audio_encoder=audio_encoder,
        tokenizer=tokenizer,
    )).to(device)

    params = get_learnable_params(configs, audio_encoder, llm)
    optimizer, scheduler = get_optimizer_and_scheduler(configs=configs, params=params)

    audio_total, audio_trainable = _count_model_params(audio_encoder)
    llm_total, llm_trainable = _count_model_params(llm)
    logger.info(
        "Audio encoder params total=%dM trainable=%dM | LLM params total=%dM trainable=%dM",
        audio_total // 1024**2, audio_trainable // 1024**2,
        llm_total // 1024**2, llm_trainable // 1024**2,
    )

    # ---- MAESTRO rendering buffer ----
    rendering_cfg = configs["rendering"]
    rendering_engine_type = rendering_cfg.get("rendering_engine", "sampler")
    if rendering_engine_type == "dawdreamer":
        render_engine = DawDreamerSampler(
            time=configs["clip_duration"],
            sr=configs["sample_rate"],
            vst_path=rendering_cfg["vst_name"],
            buffer_size=rendering_cfg.get("buffer_size", 512),
        )
    else:
        assert rendering_engine_type == "sampler", f"Unknown rendering_engine: {rendering_engine_type}"
        render_engine = RenderEngine(
            time=configs["clip_duration"],
            sr=configs["sample_rate"],
            data_dir=rendering_cfg["data_dir"],
        )

    include_program_flag = configs.get("midi_include_program", True)
    token_fn = make_token_fn(
        clip_duration=configs["clip_duration"],
        fps=configs["fps"],
        include_program=include_program_flag,
        event_token_order=configs.get("midi_event_token_order", "time_first"),
    )
    token_fn_kwargs = dict(
        clip_duration=configs["clip_duration"],
        fps=configs["fps"],
        include_program=include_program_flag,
        event_token_order=configs.get("midi_event_token_order", "time_first"),
    )
    if rendering_engine_type == "dawdreamer":
        engine_spec = dict(
            type="dawdreamer",
            time=configs["clip_duration"],
            sr=configs["sample_rate"],
            vst_path=rendering_cfg["vst_name"],
            buffer_size=rendering_cfg.get("buffer_size", 512),
        )
    else:
        engine_spec = dict(
            type="sampler",
            time=configs["clip_duration"],
            sr=configs["sample_rate"],
            data_dir=rendering_cfg["data_dir"],
        )

    maestro_root = rendering_cfg["maestro_root"]
    buffer = MaestroRenderingBuffer(
        maestro_root=maestro_root,
        render_engine=render_engine,
        clip_duration=configs["clip_duration"],
        token_fn=token_fn,
        engine_spec=engine_spec,
        token_fn_kwargs=token_fn_kwargs,
    )

    # ---- independent val buffer (same engine, never used for training) ----
    val_buffer = MaestroRenderingBuffer(
        maestro_root=maestro_root,
        render_engine=render_engine,
        clip_duration=configs["clip_duration"],
        token_fn=token_fn,
        engine_spec=engine_spec,
        token_fn_kwargs=token_fn_kwargs,
    )

    epoch_size = rendering_cfg["epoch_size"]
    repeat_times = rendering_cfg["repeat_times"]
    batch_size = configs["train"]["batch_size_per_device"]

    gradient_accumulation = configs["train"].get("gradient_accumulation", 1)
    grad_clip_norm = configs["train"].get("grad_clip_norm", None)
    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)

    training_steps = configs["train"]["training_steps"]
    test_every_n_steps = configs["train"]["test_every_n_steps"]
    global_step = 0
    epoch_counter = 0
    optimizer.zero_grad()

    render_num_workers = rendering_cfg.get("num_workers", 0)

    # ---- MAESTRO test set for validation (real audio) ----
    test_dataset = None
    token_type_ranges = get_token_type_ranges(tokenizer)
    if "test_datasets" in configs:
        test_dataset = get_dataset(configs, split="test", use_crop=True)
        logger.info("Loaded MAESTRO test set: %d samples", len(test_dataset))

    logger.info("epoch_size=%d  repeat_times=%d  batch_size=%d", epoch_size, repeat_times, batch_size)

    # ---- training loop (identical to train_random.py) ----
    pbar = tqdm(total=training_steps, desc="train", disable=configs.get("no_tqdm", False))
    while global_step < training_steps:
        # 1) render epoch
        before_render_time = datetime.now()
        buffer.render_epoch(epoch_size, seed=epoch_counter * epoch_size, num_workers=render_num_workers)
        after_render_time = datetime.now()
        epoch_counter += 1
        render_duration = (after_render_time - before_render_time).total_seconds() / 60.0
        logger.info("Rendered epoch %d in %.2f minutes, buffer size=%d",
            epoch_counter, render_duration, len(buffer))
        # 2) shuffle + iterate epoch repeat_times times
        for data in buffer.iterate_epoch(batch_size, repeat=repeat_times):
            audio, question, answering = get_audio_question_answering(data)
            audio = audio.to(device)

            audio_latent = audio_encoder.encode(
                audio=audio, train_mode=configs["audio_encoder"]["trainable"],
            )

            question_ids = tokenizer.texts_to_ids(
                texts=question, fix_length=configs["max_question_len"],
            ).to(device)

            answering_ids = tokenizer.texts_to_ids(
                texts=answering, fix_length=configs["max_answering_len"],
            ).to(device)

            if configs["train"]["remove_padded_columns"]:
                answering_ids = remove_padded_columns(
                    ids=answering_ids, pad_token_id=tokenizer.pad_token_id,
                )

            seqs = [audio_latent, question_ids, answering_ids]
            seq_types = ["audio", "id", "id"]
            loss_types = [None, None, "ce"]

            llm.train()
            output_seqs = llm(seqs=seqs, seq_types=seq_types, mask=None)

            output_seqs = [seq[:, :-1] for seq in output_seqs]
            target_seqs = [seq[:, 1:] for seq in seqs]

            loss = ce_loss(
                output_seqs=output_seqs,
                target_seqs=target_seqs,
                loss_types=loss_types,
                ignore_index=tokenizer.pad_token_id,
            )

            (loss / gradient_accumulation).backward()

            if (global_step + 1) % gradient_accumulation != 0:
                global_step += 1
                pbar.update(1)
                continue

            if grad_clip_norm is not None:
                params_to_clip = [p for p in list(audio_encoder.parameters()) + list(llm.parameters()) if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip_norm)

            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

            global_step += 1
            pbar.update(1)

            if global_step % 100 == 0:
                logger.info("step=%d loss=%.6f", global_step, loss.item())
                if wandb_log:
                    wandb.log({"train_loss": loss.item()}, step=global_step)

            # ---- validate on MAESTRO test set (real audio) ----
            if test_dataset is not None and global_step > 0 and global_step % test_every_n_steps == 0:
                val_metrics = validate_maestro(
                    configs=configs,
                    dataset=test_dataset,
                    audio_encoder=audio_encoder,
                    tokenizer=tokenizer,
                    llm=llm,
                    token_type_ranges=token_type_ranges,
                )
                logger.info("step=%d val_maestro: %s", global_step, val_metrics)
                if wandb_log:
                    wandb.log({f"val_maestro/{k}": v for k, v in val_metrics.items()}, step=global_step)

            # ---- validate on rendered MAESTRO ----
            if global_step > 0 and global_step % test_every_n_steps == 0:
                val_rand = validate_rendered_maestro(
                    configs=configs,
                    buffer=val_buffer,
                    audio_encoder=audio_encoder,
                    tokenizer=tokenizer,
                    llm=llm,
                    token_type_ranges=token_type_ranges,
                )
                logger.info("step=%d val_rendered_maestro: %s", global_step, val_rand)
                if wandb_log:
                    wandb.log({f"val_random/{k}": v for k, v in val_rand.items()}, step=global_step)

            if global_step > 0 and global_step % configs["train"]["save_every_n_steps"] == 0:
                ckpt_path = ckpt_dir / f"step={global_step}.pth"
                ckpt = {}
                if configs["audio_encoder"]["trainable"]:
                    ckpt["audio_encoder"] = audio_encoder.state_dict()
                if configs["llm"]["trainable"]:
                    ckpt["llm"] = llm.state_dict()
                torch.save(ckpt, ckpt_path)
                logger.info("Saved %s", ckpt_path)

            if global_step >= training_steps:
                break

    pbar.close()
    logger.info("Done. total steps=%d", global_step)


@hydra.main(version_base=None, config_path="configs/maestro_render", config_name="maestro_render")
def main(cfg: DictConfig) -> None:
    main_func(cfg)


if __name__ == "__main__":
    main()
