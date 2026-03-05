from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from audidata.collate.default import collate_fn
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb
from audio_understanding.data.samplers import InfiniteSampler
from audio_understanding.utils import LinearWarmUp, remove_padded_columns


def _setup_output_and_logger(configs: dict, script_name: str) -> tuple[Path, Path, Path, logging.Logger]:
    root_dir = Path(get_original_cwd())

    output_root_value = configs.get("output_root", "./checkpoints/train")
    output_root = Path(output_root_value)
    if not output_root.is_absolute():
        output_root = (root_dir / output_root).resolve()

    run_name = configs.get("run_name")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if run_name:
        processed_run_name = f"{run_name}_{timestamp}"
    else:
        processed_run_name = timestamp

    output_dir = output_root / processed_run_name
    ckpt_dir = output_dir / "ckpt"
    log_path = output_dir / f"{script_name}.log"

    ckpt_dir.mkdir(parents=True, exist_ok=True)

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


def _count_model_params(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable
def main_func(cfg: DictConfig) -> None:
    script_name = Path(__file__).stem

    configs = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(configs, dict)

    output_dir, ckpt_dir, log_path, logger = _setup_output_and_logger(configs, script_name)

    cfg_save_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, cfg_save_path)

    wandb_log = not bool(configs.get("no_log", False))

    if wandb_log:
        wandb.init(
            project="audio_understanding",
            name=str(output_dir.parent.name+"/"+output_dir.name),
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
    
    # Tokenizer for converting text into IDs and vice versa
    tokenizer = get_tokenizer(configs=configs)
    
    # LLM decoder
    llm = get_llm(
        configs=configs,
        audio_latent_dim=cast(Any, audio_encoder).latent_dim,
        vocab_size=len(cast(Any, tokenizer)),
        ckpt_path=configs["train"]["resume_ckpt_path"],
    ).to(device)

    audio_total, audio_trainable = _count_model_params(audio_encoder)
    llm_total, llm_trainable = _count_model_params(llm)
    logger.info(
        "Audio encoder params total=%d M trainable=%d M  | LLM params total=%d M trainable=%d M",
        audio_total // 1024**2,
        audio_trainable // 1024**2,
        llm_total // 1024**2,
        llm_trainable // 1024**2,
    )
    # Learnable parameters
    params = get_learnable_params(configs, audio_encoder, llm)
    optimizer, scheduler = get_optimizer_and_scheduler(configs=configs, params=params)

    for step, data in enumerate(tqdm(train_dataloader)):

        # ------ 1. Data preparation ------
        # 1.1 Prepare audio, question, and answering
        audio, question, answering = get_audio_question_answering(data)
        # audio: (b, c, t), question: (b, t), answering: (b, t)

        # 1.2 Encode audio into latent
        audio = audio.to(device)
        audio_latent = cast(Any, audio_encoder).encode(audio=audio, train_mode=configs["audio_encoder"]["trainable"])  # shape: (b, t, d)

        # 1.3 Tokenize question text to IDs
        question_ids = tokenizer.texts_to_ids(
            texts=question, 
            fix_length=configs["max_question_len"]
        ).to(device)  # shape: (b, t)

        # 1.4 Tokenize answering text to IDs
        answering_ids = tokenizer.texts_to_ids(
            texts=answering, 
            fix_length=configs["max_answering_len"]
        ).to(device)  # shape: (b, t)

        # 1.5 Remove padded columns to speed up training
        if configs["train"]["remove_padded_columns"]:
            answering_ids = remove_padded_columns(
                ids=answering_ids, 
                pad_token_id=tokenizer.pad_token_id)

        # 1.6 Prepare inputs
        seqs = [audio_latent, question_ids, answering_ids]
        seq_types = ["audio", "id", "id"]
        loss_types = [None, None, "ce"]

        # ------ 2. Training ------
        # 2.1 Forward
        llm.train()
        output_seqs = llm(
            seqs=seqs,
            seq_types=seq_types,
            mask=None
        )  # list

        # 2.2 Prepare data for next ID prediction
        output_seqs = [seq[:, 0 : -1] for seq in output_seqs]
        target_seqs = [seq[:, 1 :] for seq in seqs]
        
        # 2.3 Loss
        loss = ce_loss(
            output_seqs=output_seqs, 
            target_seqs=target_seqs, 
            loss_types=loss_types,
            ignore_index=tokenizer.pad_token_id
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        if step % 100 == 0:
            logger.info("Step: %d, Loss: %.6f", step, loss.item())
            if wandb_log:
                wandb.log(data={"train_loss_step": loss.item()}, step=step)

        if step > 0 and step % configs["train"]["test_every_n_steps"] == 500:
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

            if wandb_log:
                wandb.log(
                    data={"train_loss": train_loss, "test_loss": test_loss},
                    step=step,
                )

            logger.info("Train loss: %.6f", train_loss)
            logger.info("Test loss: %.6f", test_loss)

        if step > 0 and step % configs["train"]["save_every_n_steps"] == 0:
            ckpt_path = ckpt_dir / f"step={step}.pth"
            ckpt = {}

            if configs["audio_encoder"]["trainable"]:
                ckpt["audio_encoder"] = audio_encoder.state_dict()

            if configs["llm"]["trainable"]:
                ckpt["llm"] = llm.state_dict()

            torch.save(ckpt, ckpt_path)
            logger.info("Save model to %s", ckpt_path)

        if step == configs["train"]["training_steps"]:
            break
        
        
def get_dataset(
    configs: dict, 
    split: str
) -> Dataset:
    r"""Get datasets."""

    from audidata.io.crops import RandomCrop, StartCrop
    from audidata.transforms import Mono, TextNormalization, TimeShift

    sr = configs["sample_rate"]
    clip_duration = configs["clip_duration"]
    datasets_split = f"{split}_datasets"

    datasets = []

    for name in configs[datasets_split].keys():
        if name == "GTZAN":
            from audio_understanding.datasets.gtzan import GTZAN

            dataset = GTZAN(
                root=configs[datasets_split][name]["root"],
                split=configs[datasets_split][name]["split"],
                sr=sr,
                crop=RandomCrop(clip_duration=clip_duration),
                transform=Mono(),
            )
            datasets.append(dataset)

        elif name == "LibriSpeech":
            from audio_understanding.datasets.librispeech import LibriSpeech

            dataset = LibriSpeech(
                root=configs[datasets_split][name]["root"],
                split=configs[datasets_split][name]["split"],
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration),
                transform=[Mono(), TimeShift(sr=sr, shift=(0.0, 0.5))],
            )
            datasets.append(dataset)

        elif name == "Clotho":
            from audio_understanding.datasets.clotho import Clotho

            dataset = Clotho(
                root=configs[datasets_split][name]["root"],
                split=configs[datasets_split][name]["split"],
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration),
                transform=[Mono(), TimeShift(sr=sr, shift=(0.0, 0.5))],
                target_transform=TextNormalization(),
            )
            datasets.append(dataset)

        elif name == "MAESTRO":
            from audidata.transforms.midi import PianoRoll

            from audio_understanding.datasets.maestro import MAESTRO
            from audio_understanding.target_transforms.midi import MIDI2Tokens

            if configs["midi_to_tokens"] == "MIDI2Tokens":
                midi_transform = MIDI2Tokens(fps=configs["fps"])
            else:
                raise NotImplementedError

            dataset = MAESTRO(
                root=configs[datasets_split][name]["root"],
                split=configs[datasets_split][name]["split"],
                sr=sr,
                crop=RandomCrop(clip_duration=clip_duration, end_pad=clip_duration - 0.1),
                transform=Mono(),
                load_target=True,
                extend_pedal=True,
                target_transform=[PianoRoll(fps=100, pitches_num=128), midi_transform], #* actually, only midi_transform is needed, but we keep PianoRoll for potential future use
            )
            datasets.append(dataset)

        elif name == "Slakh2100":
            raise NotImplementedError("Slakh2100 dataset is not implemented yet.")

        elif name == "AudioCaps":
            from audio_understanding.datasets.audiocaps import AudioCaps

            dataset = AudioCaps(
                root=configs[datasets_split][name]["root"],
                split=configs[datasets_split][name]["split"],
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration),
                transform=Mono(),
                target_transform=TextNormalization(),
            )
            datasets.append(dataset)

        elif name == "WavCaps":
            from audio_understanding.datasets.wavcaps import WavCaps

            dataset = WavCaps(
                root=configs[datasets_split][name]["root"],
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration),
                transform=Mono(),
                target_transform=TextNormalization(),
            )
            datasets.append(dataset)

        else:
            raise ValueError(name)

    if len(datasets) == 1:
        return datasets[0]

    raise ValueError("Do not support multiple datasets in this file.")


def get_audio_encoder(configs: dict, ckpt_path: str) -> nn.Module:
    r"""Load pretrained audio encoder."""

    name = configs["audio_encoder"]["name"]
    sr = configs["sample_rate"]
    trainable = configs["audio_encoder"]["trainable"]

    if name == "Whisper":
        from audio_understanding.audio_encoders.whisper import Whisper

        model = Whisper(sr=sr, trainable=trainable)

    elif name == "PianoTranscriptionCRnn":
        from audio_understanding.audio_encoders.piano_transcription_crnn import PianoTranscriptionCRnn

        random_init = configs["audio_encoder"].get("random_init", False)
        model = PianoTranscriptionCRnn(sr=sr, trainable=trainable, random=random_init)

    elif name == "PannsCnn14":
        from audio_understanding.audio_encoders.panns import PannsCnn14

        model = PannsCnn14(sr=sr, trainable=trainable)

    elif name == "Conformer2d":
        from audio_understanding.audio_encoders.conformer2d import Conformer2D
        model = Conformer2D(sr=sr, trainable=True)

    else:
        raise ValueError(name)

    if ckpt_path and configs["audio_encoder"]["trainable"]:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["audio_encoder"])

    return model


def get_tokenizer(configs: dict) -> Any:
    r"""Get tokenizer."""

    name = configs["tokenizer"]["name"]

    if name == "Bert":
        from audio_understanding.tokenizers.bert import Bert

        tokenizer = Bert()

    elif name == "BertMIDI":
        from audio_understanding.tokenizers.bert_midi import BertMIDI

        tokenizer = BertMIDI()

    else:
        raise ValueError(name)

    return tokenizer


def get_llm(configs: dict, audio_latent_dim: int, vocab_size: int, ckpt_path: str) -> nn.Module:
    r"""Initialize LLM decoder."""

    name = configs["llm"]["name"]

    if name == "Llama":
        from audio_understanding.llm.llama import Llama, LlamaConfig

        block_size = configs["llm"]["block_size"]
        n_layer = configs["llm"]["n_layer"]
        n_head = configs["llm"]["n_head"]
        n_embd = configs["llm"]["n_embd"]

        config = LlamaConfig(
            block_size=block_size,
            audio_latent_dim=audio_latent_dim,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
        )
        model = Llama(config=config)

    else:
        raise ValueError(name)

    if ckpt_path and configs["llm"]["trainable"]:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["llm"])

    return model


def get_learnable_params(configs: dict, audio_encoder: nn.Module, llm: nn.Module) -> list:
    params = []

    if configs["audio_encoder"]["trainable"]:
        params += list(audio_encoder.parameters())

    if configs["llm"]["trainable"]:
        params += list(llm.parameters())

    return params


def get_optimizer_and_scheduler(
    configs: dict,
    params: list[torch.Tensor],
) -> tuple[optim.Optimizer, None | optim.lr_scheduler.LambdaLR]:
    r"""Get optimizer and scheduler."""

    lr = float(configs["train"]["lr"])
    warm_up_steps = configs["train"]["warm_up_steps"]
    optimizer_name = configs["train"]["optimizer"]

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(params=params, lr=lr)
    else:
        raise ValueError(optimizer_name)

    if warm_up_steps:
        lr_lambda = LinearWarmUp(warm_up_steps)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    return optimizer, scheduler


def get_audio_question_answering(
    data: dict
) -> tuple[torch.Tensor, list[str], list[str]]:
    r"""Process data to audio, question, and answering according to different 
    datasets.

    Returns:
        audio: (b, c, t)
        question: (b, t)
        answering: (b, t)
    """

    name = data["dataset_name"][0]

    if name in ["GTZAN"]:
        return data["audio"], data["question"], data["label"]

    elif name in ["AudioCaps", "Clotho", "LibriSpeech", "WavCaps"]:
        return data["audio"], data["question"], data["caption"]

    elif name in ["MAESTRO"]:
        return data["audio"], data["question"], data["token"]

    else:
        raise ValueError(name)


def ce_loss(
    output_seqs: list[torch.Tensor],
    target_seqs: list[torch.Tensor],
    loss_types: list[str | None],
    ignore_index: int,
) -> torch.Tensor:
    r"""Calculate loss."""

    total_loss = torch.tensor(0.0, device=output_seqs[0].device)

    for i in range(len(output_seqs)):
        if loss_types[i] is None:
            continue

        elif loss_types[i] == "ce":
            total_loss += F.cross_entropy(
                input=output_seqs[i].flatten(0, 1),
                target=target_seqs[i].flatten(0, 1),
                ignore_index=ignore_index,
            )
        else:
            raise ValueError(loss_types[i])

    return total_loss


def validate(
    configs: dict,
    dataset: Any,
    audio_encoder: nn.Module,
    tokenizer: Any,
    llm: nn.Module,
    valid_steps: int = 50,
) -> float:
    r"""Validate the model on part of data."""

    device = next(audio_encoder.parameters()).device
    losses = []

    batch_size = configs["train"]["batch_size_per_device"]
    skip_n = max(1, len(dataset) // valid_steps)

    for idx in range(0, len(dataset), skip_n):
        data = [dataset[i] for i in range(idx, min(idx + batch_size, len(dataset)))]
        data = collate_fn(data)

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


@hydra.main(version_base=None, config_path="configs", config_name="piano_transcription_maestro")
def main(cfg: DictConfig) -> None:
    main_func(cfg)


if __name__ == "__main__":
    main()
