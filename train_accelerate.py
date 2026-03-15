from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs as DDPK
from audidata.collate.default import collate_fn
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist

import wandb
from audio_understanding.data.samplers import InfiniteSampler
from audio_understanding.eval.transcription.batch_eval import batch_evaluate
from audio_understanding.utils import remove_padded_columns
from train import (
    ce_loss,
    get_audio_encoder,
    get_audio_question_answering,
    get_dataset,
    get_learnable_params,
    get_llm,
    get_optimizer_and_scheduler,
    get_tokenizer,
    transcribe_audio,
    validate,
    _log_transcription_samples
)


def _build_processed_run_name(run_name: None | str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if run_name:
        return f"{timestamp}_{run_name}"
    return timestamp


def _setup_output_and_logger(
    configs: dict,
    script_name: str,
    processed_run_name: str,
    accelerator: Accelerator,
) -> tuple[Path, Path, Path, logging.Logger]:
    root_dir = Path(get_original_cwd())

    output_root_value = configs.get("output_root", "./checkpoints/train")
    output_root = Path(output_root_value)
    if not output_root.is_absolute():
        output_root = (root_dir / output_root).resolve()
    if configs.get("no_log", False):
        output_root = root_dir / "no_log"

    output_dir = output_root / processed_run_name
    ckpt_dir = output_dir / "ckpt"
    log_path = output_dir / f"{script_name}.log"

    if accelerator.is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

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

    return output_dir, ckpt_dir, log_path, logger


def _count_model_params(model) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def _unwrap(module, accelerator: Accelerator):
    return accelerator.unwrap_model(module)


def _cleanup_runtime(accelerator: Accelerator | None) -> None:
    if accelerator is not None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            accelerator.wait_for_everyone()
        accelerator.end_training()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


#* 目前还不具备
def main_func(cfg: DictConfig) -> None:
    script_name = Path(__file__).stem

    configs = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(configs, dict)

    ddp_find_unused_parameters = (
        bool(configs["train"]["find_unused_parameters"])
        if "find_unused_parameters" in configs["train"]
        else True
    )
    kwargs = DDPK(find_unused_parameters=ddp_find_unused_parameters)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    accelerator_for_cleanup: Accelerator | None = accelerator

    try:
        run_name = configs.get("run_name")
        processed_run_name_list = [""]
        if accelerator.is_main_process:
            processed_run_name_list[0] = _build_processed_run_name(run_name)

        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(processed_run_name_list, src=0)

        processed_run_name = processed_run_name_list[0]
        assert processed_run_name

        output_dir, ckpt_dir, log_path, logger = _setup_output_and_logger(
            configs=configs,
            script_name=script_name,
            processed_run_name=processed_run_name,
            accelerator=accelerator,
        )

        cfg_save_path = output_dir / "config.yaml"
        if accelerator.is_main_process:
            OmegaConf.save(cfg, cfg_save_path)

        wandb_log = not bool(configs.get("no_log", False))

        if wandb_log and accelerator.is_main_process:
            wandb.init(
                project="audio_understanding",
                group=output_dir.parent.name,
                name=str(output_dir.name),
                config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
            )
            wandb.save(str(cfg_save_path))

        train_dataset = cast(Any, get_dataset(configs, split="train"))
        test_dataset = cast(Any, get_dataset(configs, split="test"))

        train_sampler = InfiniteSampler(train_dataset)

        num_workers = configs["train"]["num_workers"]
        dataloader_kwargs = {
            "dataset": train_dataset,
            "batch_size": configs["train"]["batch_size_per_device"],
            "sampler": cast(Any, train_sampler),
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
        )

        tokenizer = get_tokenizer(configs=configs)

        llm = get_llm(
            configs=configs,
            audio_latent_dim=cast(Any, audio_encoder).latent_dim,
            vocab_size=len(cast(Any, tokenizer)),
            ckpt_path=configs["train"]["resume_ckpt_path"],
        )

        audio_total, audio_trainable = _count_model_params(audio_encoder)
        llm_total, llm_trainable = _count_model_params(llm)
        if accelerator.is_main_process:
            logger.info(
                "Audio encoder params total=%d M trainable=%d M  | LLM params total=%d M trainable=%d M",
                audio_total // 1024**2,
                audio_trainable // 1024**2,
                llm_total // 1024**2,
                llm_trainable // 1024**2,
            )
            logger.info("DDP find_unused_parameters: %s", ddp_find_unused_parameters)
            logger.info("Output dir: %s", output_dir)
            logger.info("Log file: %s", log_path)

        params = get_learnable_params(configs, audio_encoder, llm)
        optimizer, scheduler = get_optimizer_and_scheduler(configs=configs, params=params)

        audio_encoder, llm, optimizer, train_dataloader = accelerator.prepare(
            audio_encoder,
            llm,
            optimizer,
            train_dataloader,
        )

        gradient_accumulation = configs["train"]["gradient_accumulation"] if "gradient_accumulation" in configs["train"] else 1
        assert gradient_accumulation > 0
        grad_clip_norm = float(configs["train"]["grad_clip_norm"]) if "grad_clip_norm" in configs["train"] else None
        global_step = 0
        optimizer.zero_grad()

        for micro_step, data in enumerate(tqdm(train_dataloader, disable=not accelerator.is_main_process)):
            audio, question, answering = get_audio_question_answering(data)
            audio = audio.to(accelerator.device)

            encoder_model = _unwrap(audio_encoder, accelerator)
            audio_latent = cast(Any, encoder_model).encode(audio=audio, train_mode=configs["audio_encoder"]["trainable"])
            device = audio_latent.device

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

            accelerator.backward(loss / gradient_accumulation)

            if (micro_step + 1) % gradient_accumulation != 0:
                continue

            grad_norm_value = None
            if grad_clip_norm is not None:
                params_to_clip = [
                    p
                    for p in list(audio_encoder.parameters()) + list(llm.parameters())
                    if p.requires_grad
                ]
                grad_norm = accelerator.clip_grad_norm_(params_to_clip, grad_clip_norm)
                if isinstance(grad_norm, torch.Tensor):
                    grad_norm_value = float(grad_norm.item())
                else:
                    grad_norm_value = float(grad_norm)

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if scheduler:
                scheduler.step()

            if global_step % 100 == 0 and accelerator.is_main_process:
                if grad_norm_value is None:
                    logger.info("Step: %d, Loss: %.6f", global_step, loss.item())
                else:
                    logger.info("Step: %d, Loss: %.6f, GradNorm: %.6f", global_step, loss.item(), grad_norm_value)
                if wandb_log:
                    payload = {"train_loss_step": loss.item()}
                    if grad_norm_value is not None:
                        payload["grad_norm"] = grad_norm_value
                    wandb.log(data=payload, step=global_step)

            if global_step % configs["train"]["test_every_n_steps"] == 500 and accelerator.is_main_process:
                train_loss = validate(
                    configs=configs,
                    dataset=train_dataset,
                    audio_encoder=_unwrap(audio_encoder, accelerator),
                    tokenizer=tokenizer,
                    llm=_unwrap(llm, accelerator),
                )

                test_loss = validate(
                    configs=configs,
                    dataset=test_dataset,
                    audio_encoder=_unwrap(audio_encoder, accelerator),
                    tokenizer=tokenizer,
                    llm=_unwrap(llm, accelerator),
                )

                if wandb_log:
                    wandb.log(
                        data={"train_loss": train_loss, "test_loss": test_loss},
                        step=global_step,
                    )

                logger.info("Train loss: %.6f", train_loss)
                logger.info("Test loss: %.6f", test_loss)

            if global_step > 0 and global_step % configs["train"]["save_every_n_steps"] == 0 and accelerator.is_main_process:
                ckpt_path = ckpt_dir / f"step={global_step}.pth"
                ckpt = {}

                if configs["audio_encoder"]["trainable"]:
                    ckpt["audio_encoder"] = _unwrap(audio_encoder, accelerator).state_dict()

                if configs["llm"]["trainable"]:
                    ckpt["llm"] = _unwrap(llm, accelerator).state_dict()

                torch.save(ckpt, ckpt_path)
                logger.info("Save model to %s", ckpt_path)


            # if global_step == 10:
            #     accelerator.wait_for_everyone()
            #     import sys
            #     sys.exit()          
            if global_step * 10 % configs["train"]["save_every_n_steps"] == 0 and accelerator.is_main_process:
                
                train_log_n_samples = int(configs["train"].get("transcription_train_n_samples", 1))
                test_log_n_samples = int(configs["train"].get("transcription_test_n_samples", 2))
                logger.info("Logging train data samples:")
                _log_transcription_samples(
                    configs=configs,
                    dataset=train_dataset,
                    audio_encoder=_unwrap(audio_encoder, accelerator),
                    tokenizer=tokenizer,
                    llm=_unwrap(llm, accelerator),
                    output_dir=output_dir,
                    step=global_step,
                    logger=logger,
                    device=audio_latent.device,
                    n_samples=train_log_n_samples,
                    split_name="train",
                    run_batch_eval=False,
                )

                logger.info("Logging test data samples and metrics:")
                test_metrics = _log_transcription_samples(
                    configs=configs,
                    dataset=test_dataset,
                    audio_encoder=_unwrap(audio_encoder, accelerator),
                    tokenizer=tokenizer,
                    llm=_unwrap(llm, accelerator),
                    output_dir=output_dir,
                    step=global_step,
                    logger=logger,
                    device=audio_latent.device,
                    n_samples=test_log_n_samples,
                    split_name="test",
                    run_batch_eval=True,
                )
                if wandb_log:
                    wandb.log(data=test_metrics, step=global_step)

            if global_step == configs["train"]["training_steps"]:
                break

    except Exception as e:
        if accelerator_for_cleanup is not None and accelerator_for_cleanup.is_main_process:
            logging.getLogger(script_name).exception("Training failed: %s", e)
        raise
    finally:
        _cleanup_runtime(accelerator_for_cleanup)


@hydra.main(version_base=None, config_path="configs", config_name="piano_transcription_maestro")
def main(cfg: DictConfig) -> None:
    main_func(cfg)


if __name__ == "__main__":
    main()
