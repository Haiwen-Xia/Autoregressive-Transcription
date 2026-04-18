"""Fine-tune on MAESTRO validation split.

Usage:
    python finetune_maestro_val.py --ckpt <step=N.pth> --steps 5000 [--device cuda:0] [--lr 1e-5] [--no_log]

Config is auto-loaded from <ckpt>.parent.parent/config.yaml.
train_datasets is overridden to MAESTRO validation split.
wandb group: finetune_maestro
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from audidata.collate.default import collate_fn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from audio_understanding.data.samplers import InfiniteSampler
from audio_understanding.utils import LinearWarmUp, parse_yaml, remove_padded_columns

# Reuse all builder functions from train.py
from train import (
    _count_model_params,
    _log_transcription_samples,
    ce_loss,
    get_audio_encoder,
    get_audio_question_answering,
    get_dataset,
    get_learnable_params,
    get_llm,
    get_optimizer_and_scheduler,
    get_tokenizer,
    validate,
)


def main_func(args: argparse.Namespace) -> None:
    if not args.config:
        assert args.ckpt, "If --config not provided, --ckpt must be specified to locate config.yaml"
        config_yaml = Path(args.ckpt).resolve().parent.parent / "config.yaml"
    else:
        config_yaml = Path(args.config).resolve()
    assert config_yaml.exists(), f"Config not found: {config_yaml}"
    
    # if not args.config and args.ckpt:
    #     ckpt_path = Path(args.ckpt).resolve() if args.ckpt else None
    #     audio_ckpt_path = Path(args.audio_ckpt).resolve() if args.audio_ckpt else ckpt_path
    #     assert ckpt_path.exists() or audio_ckpt_path.exists(), f"Checkpoint not found: {ckpt_path} or {audio_ckpt_path}"
    #     config_yaml = ckpt_path.parent.parent / "config.yaml"
    # elif args.config and not args.ckpt and args.audio_ckpt:
    #     config_yaml = Path(args.config).resolve()
    #     audio_ckpt_path = Path(args.audio_ckpt).resolve()
    #     assert audio_ckpt_path.exists(), f"Audio encoder checkpoint not found: {audio_ckpt_path}"
    # assert config_yaml.exists(), f"Config not found: {config_yaml}"

    cfg = OmegaConf.load(config_yaml)
    configs: dict[str, Any] = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))

    ckpt_path = args.ckpt if args.ckpt else None
    if args.audio_ckpt:
        configs['audio_encoder']['ckpt_path'] = str(args.audio_ckpt)
    # ----- override: finetune on MAESTRO validation split -----
    # Determine MAESTRO root: CLI > test_datasets > train_datasets
    if args.maestro_root:
        maestro_root = args.maestro_root
    elif "MAESTRO" in configs.get("test_datasets", {}):
        maestro_root = configs["test_datasets"]["MAESTRO"]["root"]
    elif "MAESTRO" in configs.get("train_datasets", {}):
        maestro_root = configs["train_datasets"]["MAESTRO"]["root"]
    else:
        raise ValueError("Cannot find MAESTRO root. Pass --maestro_root <path>.")

    configs["train_datasets"] = {
        "MAESTRO": {"root": maestro_root, "split": "validation"}
    }

    # Keep test_datasets as-is (usually "test") for eval comparison

    # ----- override train hyper-params -----
    configs["train"]["training_steps"] = args.steps
    configs["train"]["resume_ckpt_path"] = str(ckpt_path)
    if args.device:
        configs["train"]["device"] = args.device
    if args.lr is not None:
        configs["train"]["lr"] = args.lr
    # fewer workers for small finetune run if not specified
    configs["train"].setdefault("num_workers", 4)

    # reset warm_up to a small value relative to finetune steps
    if args.warm_up_steps is not None:
        configs["train"]["warm_up_steps"] = args.warm_up_steps

    configs["train"]["test_every_n_steps"] = args.test_every
    configs["train"]["save_every_n_steps"] = args.save_every

    # ----- output dir -----
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"finetune_maestro_val_{timestamp}"
    output_dir = Path(configs.get("output_root", "./checkpoints/train")).resolve() / run_name
    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # save overridden config
    OmegaConf.save(OmegaConf.create(configs), output_dir / "config.yaml")

    # ----- logger -----
    logger = logging.getLogger("finetune_maestro_val")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(output_dir / "finetune.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("Config loaded from: %s", config_yaml)
    logger.info("Checkpoint: %s", ckpt_path)
    logger.info("Output dir: %s", output_dir)
    logger.info("Fine-tuning for %d steps on MAESTRO validation split", args.steps)

    # ----- wandb -----
    wandb_log = not args.no_log
    if wandb_log:
        wandb.init(
            project="audio_understanding",
            group="finetune_maestro",
            name=run_name,
            config=configs,
        )

    device = configs["train"]["device"]
    no_metrics = bool(configs["train"].get("no_metrics", False))

    # ----- datasets -----
    train_dataset = cast(Any, get_dataset(configs, split="train"))  # uses overridden "validation"
    test_dataset  = cast(Any, get_dataset(configs, split="test"))

    train_sampler = InfiniteSampler(train_dataset)
    num_workers = configs["train"]["num_workers"]
    dataloader_kwargs: dict[str, Any] = {
        "dataset": train_dataset,
        "batch_size": configs["train"]["batch_size_per_device"],
        "sampler": train_sampler,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": True,
    }
    if num_workers > 0:
        dataloader_kwargs.update({
            "multiprocessing_context": "spawn",
            "persistent_workers": True,
            "prefetch_factor": 2,
            "timeout": 120,
        })
    train_dataloader = DataLoader(**dataloader_kwargs)

    # ----- models -----
    audio_encoder = get_audio_encoder(
        configs=configs,
        ckpt_path=ckpt_path,
    ).to(device)
    print(f"Audio encoder loaded. Trainable: {configs['audio_encoder']['trainable']}")
    tokenizer = get_tokenizer(configs=configs)
    if args.randomize:
        ckpt_path = None
    llm = get_llm(
        configs=configs,
        audio_latent_dim=cast(Any, audio_encoder).latent_dim,
        vocab_size=len(cast(Any, tokenizer)),
        ckpt_path=ckpt_path,
        audio_encoder=audio_encoder,
        tokenizer=tokenizer,
    ).to(device)

    if args.randomize:
        for module in llm.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        logger.info("--randomize: all LLM weights re-initialized (audio_encoder weights kept)")

    params = get_learnable_params(configs, audio_encoder, llm)
    optimizer, scheduler = get_optimizer_and_scheduler(configs=configs, params=params)

    audio_total, audio_trainable = _count_model_params(audio_encoder)
    llm_total, llm_trainable = _count_model_params(llm)
    logger.info(
        "Audio encoder params total=%d M trainable=%d M | LLM params total=%d M trainable=%d M",
        audio_total // 1024**2, audio_trainable // 1024**2,
        llm_total // 1024**2,  llm_trainable // 1024**2,
    )

    gradient_accumulation = configs["train"].get("gradient_accumulation", 1)
    assert gradient_accumulation > 0
    grad_clip_norm_val = configs["train"].get("grad_clip_norm", None)
    grad_clip_norm = float(grad_clip_norm_val) if grad_clip_norm_val is not None else None

    global_step = 0
    optimizer.zero_grad()

    test_every  = configs["train"].get("test_every_n_steps",  500)
    save_every  = configs["train"].get("save_every_n_steps", 1000)

    # ----- training loop -----
    for micro_step, data in enumerate(tqdm(train_dataloader)):

        audio, question, answering = get_audio_question_answering(data)

        audio = audio.to(device)
        audio_latent = cast(Any, audio_encoder).encode(
            audio=audio, train_mode=configs["audio_encoder"]["trainable"]
        )

        question_ids = tokenizer.texts_to_ids(
            texts=question, fix_length=configs["max_question_len"]
        ).to(device)

        answering_ids = tokenizer.texts_to_ids(
            texts=answering, fix_length=configs["max_answering_len"]
        ).to(device)

        if configs["train"]["remove_padded_columns"]:
            answering_ids = remove_padded_columns(
                ids=answering_ids, pad_token_id=tokenizer.pad_token_id
            )

        seqs      = [audio_latent, question_ids, answering_ids]
        seq_types = ["audio", "id", "id"]
        loss_types = [None, None, "ce"]

        llm.train()
        output_seqs = llm(seqs=seqs, seq_types=seq_types, mask=None)

        output_seqs = [seq[:, :-1] for seq in output_seqs]
        target_seqs = [seq[:, 1:]  for seq in seqs]

        loss = ce_loss(
            output_seqs=output_seqs,
            target_seqs=target_seqs,
            loss_types=loss_types,
            ignore_index=tokenizer.pad_token_id,
        )

        (loss / gradient_accumulation).backward()

        if (micro_step + 1) % gradient_accumulation != 0:
            continue

        grad_norm_value = None
        if grad_clip_norm is not None:
            params_to_clip = [
                p for p in list(audio_encoder.parameters()) + list(llm.parameters())
                if p.requires_grad
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
                logger.info("Step: %d, Loss: %.6f, GradNorm: %.6f", global_step, loss.item(), grad_norm_value)
            if wandb_log:
                payload = {"train_loss_step": loss.item()}
                if grad_norm_value is not None:
                    payload["grad_norm"] = grad_norm_value
                wandb.log(data=payload, step=global_step)

        if global_step % test_every == 0:
            train_loss = validate(configs=configs, dataset=train_dataset,
                                  audio_encoder=audio_encoder, tokenizer=tokenizer, llm=llm)
            test_loss  = validate(configs=configs, dataset=test_dataset,
                                  audio_encoder=audio_encoder, tokenizer=tokenizer, llm=llm)
            logger.info("Train loss: %.6f  Test loss: %.6f", train_loss, test_loss)
            if wandb_log:
                wandb.log(data={"train_loss": train_loss, "test_loss": test_loss}, step=global_step)

        if global_step % save_every == 0:
            save_path = ckpt_dir / f"step={global_step}.pth"
            ckpt = {}
            if configs["audio_encoder"]["trainable"]:
                ckpt["audio_encoder"] = audio_encoder.state_dict()
            if configs["llm"]["trainable"]:
                ckpt["llm"] = llm.state_dict()
            torch.save(ckpt, save_path)
            logger.info("Saved checkpoint: %s", save_path)

        if global_step >= args.steps:
            break

    # ----- final checkpoint -----
    final_path = ckpt_dir / f"step={global_step}_final.pth"
    ckpt = {}
    if configs["audio_encoder"]["trainable"]:
        ckpt["audio_encoder"] = audio_encoder.state_dict()
    if configs["llm"]["trainable"]:
        ckpt["llm"] = llm.state_dict()
    torch.save(ckpt, final_path)
    logger.info("Saved final checkpoint: %s", final_path)

    # ----- final eval -----
    logger.info("=== Final evaluation ===")
    test_loss = validate(configs=configs, dataset=test_dataset,
                         audio_encoder=audio_encoder, tokenizer=tokenizer, llm=llm)
    logger.info("Final test loss: %.6f", test_loss)

    if not no_metrics:
        logger.info("Running final transcription batch eval on test set...")
        # cap eval samples to avoid hanging on the full test set
        configs["train"]["transcription_eval_max_samples"] = args.eval_max_samples
        final_metrics = _log_transcription_samples(
            configs=configs,
            dataset=test_dataset,
            audio_encoder=audio_encoder,
            tokenizer=tokenizer,
            llm=llm,
            output_dir=output_dir,
            step=global_step,
            logger=logger,
            device=device,
            n_samples=2,
            split_name="test_final",
            run_batch_eval=True,
        )
        logger.info("Final metrics: %s", final_metrics)
        if wandb_log:
            wandb.log(data={"final/test_loss": test_loss, **{f"final/{k}": v for k, v in final_metrics.items()}},
                      step=global_step)

    if wandb_log:
        wandb.finish()

    logger.info("Done. Output: %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune on MAESTRO validation split")
    parser.add_argument("--ckpt",'-c',          default=None,  help="Path to input checkpoint (.pth)")
    parser.add_argument("--audio_ckpt",  '-a',       default=None,  help="Path to input audio encoder checkpoint (.pth), if separate from --ckpt")
    parser.add_argument('--config',      '-f',       default=None,  help="Path to config.yaml to override checkpoint's config (optional)")
    parser.add_argument("--steps",         default=9000,  type=int, help="Number of fine-tune steps")
    parser.add_argument("--device",        default=None,   help="Override device (e.g. cuda:1)")
    parser.add_argument("--lr",            default=None,   type=float, help="Override learning rate")
    parser.add_argument("--warm_up_steps", default=None,   type=int,   help="Override warm-up steps")
    parser.add_argument("--test_every",    default=1000,   type=int,   help="Test every N steps (default: 1000)")
    parser.add_argument("--save_every",    default=3000,   type=int,   help="Save every N steps (default: 3000)")
    parser.add_argument("--no_log",        action="store_true", help="Disable wandb logging")
    parser.add_argument("--maestro_root",  default=None, help="Override MAESTRO dataset root path")
    parser.add_argument("--randomize",     action="store_true", help="Re-initialize all LLM weights after loading (audio_encoder kept)")
    parser.add_argument("--eval_max_samples", default=100, type=int, help="Max samples for final batch eval (default: 50)")
    args = parser.parse_args()
    main_func(args)


if __name__ == "__main__":
    main()
