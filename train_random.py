"""Online-rendering training script.

Flow:
    1. on-the-fly rendering dataset (__getitem__)
    2. InfiniteSampler + DataLoader for unified backend access
    3. train/validate with the same batch interface
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from audidata.collate.default import collate_fn
from audidata.samplers import InfiniteSampler
from audio_understanding.random_rendering.dataset import (
    RandomTokenRenderingDataset,
    collate_random_token_batch,
    make_token_fn,
)
from audio_understanding.random_rendering.rendering import DawDreamerSampler, Sampler as RenderEngine
from audio_understanding.random_rendering.sampler import MarginalRandomSampler, UniformSampler
from audio_understanding.utils import remove_padded_columns
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


def get_audio_question_answering(data: dict):
    return data["audio"], data["question"], data["token"]


def get_token_type_ranges(tokenizer) -> dict:
    """Get token ID ranges for each MIDI token type from the tokenizer.

    Returns dict with keys: time_index, name_onset, name_offset, pitch, velocity, program.
    Each value is (start_id, end_id) inclusive.
    """
    # BertMIDI adds tokens after original vocab. The order matches __init__:
    #   time_index=0..6000, name=note_onset/sustain/offset, name=pedal_onset/sustain/offset,
    #   pitch=0..127, velocity=0..127, program=0..127
    base = 30522  # bert-base-uncased vocab size
    ranges = {
        "time_index": (base, base + 6000),                   # 6001 tokens
        "name_onset": (base + 6001, base + 6001),             # name=note_onset
        "name_offset": (base + 6003, base + 6003),            # name=note_offset
        "pitch": (base + 6007, base + 6007 + 127),            # pitch=0..127
        "velocity": (base + 6007 + 128, base + 6007 + 255),   # velocity=0..127
        "program": (base + 6007 + 256, base + 6007 + 383),    # program=0..127
    }
    # Verify with tokenizer
    assert tokenizer.tok.convert_tokens_to_ids("time_index=0") == base
    assert tokenizer.tok.convert_tokens_to_ids("name=note_onset") == base + 6001
    assert tokenizer.tok.convert_tokens_to_ids("name=note_offset") == base + 6003
    assert tokenizer.tok.convert_tokens_to_ids("pitch=0") == base + 6007
    return ranges


def _masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """CE loss only at positions where mask is True. Returns 0 if no valid positions."""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return F.cross_entropy(logits[mask], targets[mask], ignore_index=ignore_index)


def validate_maestro(
    configs: dict,
    dataset,
    audio_encoder,
    tokenizer,
    llm,
    token_type_ranges: dict,
    valid_steps: int = 50,
) -> dict:
    """Validate on MAESTRO test set, returning per-token-type CE losses.

    Returns dict with keys: total_ce, pitch_ce, onset_ce, offset_ce, velocity_ce.
    - pitch_ce: CE on pitch tokens
    - onset_ce: CE on time_index tokens that precede name=note_onset
    - offset_ce: CE on time_index tokens that precede name=note_offset
    - velocity_ce: CE on velocity tokens
    """
    device = next(audio_encoder.parameters()).device
    batch_size = configs["train"]["batch_size_per_device"]
    skip_n = max(1, len(dataset) // valid_steps)
    pad_id = tokenizer.pad_token_id

    onset_id = token_type_ranges["name_onset"][0]
    offset_id = token_type_ranges["name_offset"][0]
    time_lo, time_hi = token_type_ranges["time_index"]
    pitch_lo, pitch_hi = token_type_ranges["pitch"]
    vel_lo, vel_hi = token_type_ranges["velocity"]

    accum = {k: [] for k in ["total", "pitch", "onset", "offset", "velocity"]}

    for idx in range(0, len(dataset), skip_n):
        data = [dataset[i] for i in range(idx, min(idx + batch_size, len(dataset)))]
        data = cast(dict, collate_fn(data))

        audio, question, answering = data["audio"], data["question"], data["token"]
        audio = audio.to(device)

        audio_latent = audio_encoder.encode(audio=audio, train_mode=False)
        question_ids = tokenizer.texts_to_ids(texts=question, fix_length=configs["max_question_len"]).to(device)
        answering_ids = tokenizer.texts_to_ids(texts=answering, fix_length=configs["max_answering_len"]).to(device)

        if configs["train"]["remove_padded_columns"]:
            answering_ids = remove_padded_columns(ids=answering_ids, pad_token_id=pad_id)

        seqs = [audio_latent, question_ids, answering_ids]
        seq_types = ["audio", "id", "id"]
        loss_types = [None, None, "ce"]

        with torch.no_grad():
            llm.eval()
            output_seqs = llm(seqs=seqs, seq_types=seq_types, mask=None)

        # shift: predict next token
        pred_logits = output_seqs[-1][:, :-1]   # (B, L-1, V)
        targets = answering_ids[:, 1:]           # (B, L-1)

        B, L = targets.shape
        flat_logits = pred_logits.reshape(B * L, -1)
        flat_targets = targets.reshape(B * L)

        # total CE (same as original validate)
        total = F.cross_entropy(flat_logits, flat_targets, ignore_index=pad_id)
        accum["total"].append(total.item())

        # Build masks based on target token type
        is_time = (flat_targets >= time_lo) & (flat_targets <= time_hi)
        is_pitch = (flat_targets >= pitch_lo) & (flat_targets <= pitch_hi)
        is_vel = (flat_targets >= vel_lo) & (flat_targets <= vel_hi)

        # onset/offset: time_index tokens where the NEXT target is name=note_onset/offset
        # We need to look at targets shifted by +1
        next_targets = torch.zeros_like(flat_targets)
        targets_2d = targets  # (B, L)
        # For each row, next token of position i is position i+1; last position has no next
        if L > 1:
            next_2d = torch.cat([targets_2d[:, 1:], torch.zeros(B, 1, dtype=targets.dtype, device=device)], dim=1)
            next_targets = next_2d.reshape(B * L)

        is_onset_time = is_time & (next_targets == onset_id)
        is_offset_time = is_time & (next_targets == offset_id)

        accum["pitch"].append(_masked_ce(flat_logits, flat_targets, is_pitch, pad_id).item())
        accum["onset"].append(_masked_ce(flat_logits, flat_targets, is_onset_time, pad_id).item())
        accum["offset"].append(_masked_ce(flat_logits, flat_targets, is_offset_time, pad_id).item())
        accum["velocity"].append(_masked_ce(flat_logits, flat_targets, is_vel, pad_id).item())

    return {f"{k}_ce": float(np.mean(v)) if v else 0.0 for k, v in accum.items()}


def validate_random(
    configs: dict,
    dataloader,
    audio_encoder,
    tokenizer,
    llm,
    token_type_ranges: dict,
    valid_size: int = 64,
) -> dict:
    """Validate on freshly rendered random data from a dataset dataloader."""
    device = next(audio_encoder.parameters()).device
    pad_id = tokenizer.pad_token_id

    onset_id = token_type_ranges["name_onset"][0]
    offset_id = token_type_ranges["name_offset"][0]
    time_lo, time_hi = token_type_ranges["time_index"]
    pitch_lo, pitch_hi = token_type_ranges["pitch"]
    vel_lo, vel_hi = token_type_ranges["velocity"]

    accum = {k: [] for k in ["total", "pitch", "onset", "offset", "velocity"]}
    n_seen = 0
    for data in dataloader:
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

        n_seen += audio.shape[0]
        if n_seen >= valid_size:
            break

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
            group="random_rendering",
            name=str(output_dir.name),
            config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
        )

    device = configs["train"]["device"]

    # ---- model ----
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

    # ---- on-the-fly rendering dataset ----
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
    sampler_type = rendering_cfg.get("sampler_type", "marginal")
    deduplication = rendering_cfg.get("deduplication", "disabled")
    if sampler_type == "marginal":
        note_sampler = MarginalRandomSampler(
            time=configs["clip_duration"],
            config_json=rendering_cfg["sampler_config"],
            deduplication=deduplication,
        )
    elif sampler_type == "uniform":
        note_sampler = UniformSampler(
            time=configs["clip_duration"],
            config_json=rendering_cfg["sampler_config"],
            keep_marginal_keys=rendering_cfg.get("keep_marginal_keys", []),
            deduplication=deduplication,
        )
    else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}. Choose 'marginal' or 'uniform'.")
    # midi_include_program (top-level) overrides rendering.include_program so that
    # the random rendering token format stays consistent with the MAESTRO pipeline.
    include_program_flag = configs.get("midi_include_program", True)
    token_fn = make_token_fn(
        clip_duration=configs["clip_duration"],
        fps=configs["fps"],
        include_program=include_program_flag,
        event_token_order=configs.get("midi_event_token_order", "time_first"),
    )
    epoch_size = int(rendering_cfg["epoch_size"])
    repeat_times = int(rendering_cfg["repeat_times"])
    batch_size = configs["train"]["batch_size_per_device"]
    train_dataset = RandomTokenRenderingDataset(
        render_engine=render_engine,
        note_sampler=note_sampler,
        clip_duration=configs["clip_duration"],
        token_fn=token_fn,
        epoch_size=max(1, epoch_size * repeat_times),
    )
    val_dataset = RandomTokenRenderingDataset(
        render_engine=render_engine,
        note_sampler=note_sampler,
        clip_duration=configs["clip_duration"],
        token_fn=token_fn,
        epoch_size=max(1, int(rendering_cfg.get("val_epoch_size", 64))),
    )

    render_num_workers = int(rendering_cfg.get("num_workers", 0))
    train_loader_kwargs = {
        "dataset": train_dataset,
        "batch_size": batch_size,
        "sampler": InfiniteSampler(train_dataset),
        "num_workers": render_num_workers,
        "collate_fn": collate_random_token_batch,
        "pin_memory": True,
    }
    if render_num_workers > 0:
        train_loader_kwargs["multiprocessing_context"] = "spawn"
        train_loader_kwargs["prefetch_factor"] = int(rendering_cfg.get("prefetch_factor", 4))
        train_loader_kwargs["persistent_workers"] = True
    train_dataloader = DataLoader(**train_loader_kwargs)

    val_loader_kwargs = {
        "dataset": val_dataset,
        "batch_size": batch_size,
        "sampler": InfiniteSampler(val_dataset),
        "num_workers": 0,
        "collate_fn": collate_random_token_batch,
        "pin_memory": True,
    }
    val_random_dataloader = DataLoader(**val_loader_kwargs)

    gradient_accumulation = configs["train"].get("gradient_accumulation", 1)
    grad_clip_norm = configs["train"].get("grad_clip_norm", None)
    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)

    training_steps = configs["train"]["training_steps"]
    test_every_n_steps = configs["train"]["test_every_n_steps"]
    global_step = 0
    optimizer.zero_grad()

    # ---- MAESTRO test set for validation ----
    test_dataset = None
    token_type_ranges = get_token_type_ranges(tokenizer)
    if "test_datasets" in configs:
        test_dataset = get_dataset(configs, split="test", use_crop=True)
        logger.info("Loaded MAESTRO test set: %d samples", len(test_dataset))

    logger.info("dataset_epoch_size=%d  repeat_times=%d  batch_size=%d", epoch_size, repeat_times, batch_size)

    # ---- training loop ----
    pbar = tqdm(total=training_steps, desc="train", disable=configs.get("no_tqdm", False))
    train_iter = iter(train_dataloader)
    while global_step < training_steps:
        data = next(train_iter)
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

        # ---- validate on MAESTRO test set ----
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

        # ---- validate on random rendering ----
        if global_step > 0 and global_step % test_every_n_steps == 0:
            val_rand = validate_random(
                configs=configs,
                dataloader=val_random_dataloader,
                audio_encoder=audio_encoder,
                tokenizer=tokenizer,
                llm=llm,
                token_type_ranges=token_type_ranges,
                valid_size=int(rendering_cfg.get("val_epoch_size", 64)),
            )
            logger.info("step=%d val_random: %s", global_step, val_rand)
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


@hydra.main(version_base=None, config_path="configs/random_rendering", config_name="random_rendering")
def main(cfg: DictConfig) -> None:
    main_func(cfg)


if __name__ == "__main__":
    main()
