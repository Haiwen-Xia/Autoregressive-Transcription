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
from audio_understanding.eval.transcription.batch_eval import batch_evaluate
from audio_understanding.utils import LinearWarmUp, remove_padded_columns
from inference_transcription import transcribe_audio, format_tokens_by_event, tokens_to_midi
import soundfile as sf


def _setup_output_and_logger(configs: dict, script_name: str) -> tuple[Path, Path, Path, logging.Logger]:
    root_dir = Path(get_original_cwd())

    output_root_value = configs.get("output_root", "./checkpoints/train")
    output_root = Path(output_root_value)
    if not output_root.is_absolute():
        output_root = (root_dir / output_root).resolve()
    if configs.get("no_log", False):
        output_root = root_dir / "no_log" 
    run_name = configs.get("run_name")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if run_name:
        processed_run_name = f"{timestamp}_{run_name}"
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
            group=output_dir.parent.name, # another option is to use HydraConfig.get().runtime.config_name
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
    
    # Tokenizer for converting text into IDs and vice versa
    tokenizer = get_tokenizer(configs=configs)
    
    # LLM decoder
    llm = get_llm(
        configs=configs,
        audio_latent_dim=cast(Any, audio_encoder).latent_dim,
        vocab_size=len(cast(Any, tokenizer)),
        ckpt_path=configs["train"]["resume_ckpt_path"],
    ).to(device)

    # Learnable parameters
    params = get_learnable_params(configs, audio_encoder, llm)
    optimizer, scheduler = get_optimizer_and_scheduler(configs=configs, params=params)
    
    audio_total, audio_trainable = _count_model_params(audio_encoder)
    llm_total, llm_trainable = _count_model_params(llm)
    logger.info(
        "Audio encoder params total=%d M trainable=%d M  | LLM params total=%d M trainable=%d M",
        audio_total // 1024**2,
        audio_trainable // 1024**2,
        llm_total // 1024**2,
        llm_trainable // 1024**2,
    )
    
    gradient_accumulation = configs["train"]["gradient_accumulation"] if "gradient_accumulation" in configs["train"] else 1
    assert gradient_accumulation > 0
    grad_clip_norm = float(configs["train"]["grad_clip_norm"]) if "grad_clip_norm" in configs["train"] else None
    global_step = 0
    optimizer.zero_grad()

    for micro_step, data in enumerate(tqdm(train_dataloader)):

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

        (loss / gradient_accumulation).backward()

        if (micro_step + 1) % gradient_accumulation != 0:
            continue

        grad_norm_value = None
        if grad_clip_norm is not None:
            params_to_clip = [
                p
                for p in list(audio_encoder.parameters()) + list(llm.parameters())
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

        if global_step % configs["train"]["test_every_n_steps"] == 500:
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
                    step=global_step,
                )

            logger.info("Train loss: %.6f", train_loss)
            logger.info("Test loss: %.6f", test_loss)
        #* might open specific log around here
        if global_step > 0 and global_step % configs["train"]["save_every_n_steps"] == 0:
            ckpt_path = ckpt_dir / f"step={global_step}.pth"
            ckpt = {}

            if configs["audio_encoder"]["trainable"]:
                ckpt["audio_encoder"] = audio_encoder.state_dict()

            if configs["llm"]["trainable"]:
                ckpt["llm"] = llm.state_dict()

            torch.save(ckpt, ckpt_path)
            logger.info("Save model to %s", ckpt_path)

        # ------ Transcription sample logging (at save steps) ------
        if global_step * 10 % configs["train"]["save_every_n_steps"] == 0:
            train_log_n_samples = int(configs["train"].get("transcription_train_n_samples", 1))
            test_log_n_samples = int(configs["train"].get("transcription_test_n_samples", 2))

            logger.info("Logging train data samples:")
            _log_transcription_samples(
                configs=configs,
                dataset=train_dataset,
                audio_encoder=audio_encoder,
                tokenizer=tokenizer,
                llm=llm,
                output_dir=output_dir,
                step=global_step,
                logger=logger,
                device=device,
                n_samples=train_log_n_samples,
                split_name="train",
                run_batch_eval=False,
            )
            
            logger.info("Logging test data samples and metrics:")
            test_metrics = _log_transcription_samples(
                configs=configs,
                dataset=test_dataset,
                audio_encoder=audio_encoder,
                tokenizer=tokenizer,
                llm=llm,
                output_dir=output_dir,
                step=global_step,
                logger=logger,
                device=device,
                n_samples=test_log_n_samples,
                split_name="test",
                run_batch_eval=True,
            )
            if wandb_log:
                wandb.log(data=test_metrics, step=global_step)

        if global_step == configs["train"]["training_steps"]:
            break
        
        
def _log_transcription_samples(
    configs: dict,
    dataset,
    audio_encoder: nn.Module,
    tokenizer,
    llm: nn.Module,
    output_dir: Path,
    step: int,
    logger: logging.Logger,
    device: str,
    n_samples: int = 2,
    split_name: str = "test",
    run_batch_eval: bool = True,
) -> dict[str, float]:
    """Pick n_samples from dataset, run constrained decoding, log results."""
    
    # import soundfile as sf

    include_program = configs.get("midi_include_program", False)
    write_program_tracks = bool(configs.get("midi_write_program_tracks", include_program))
    sr = configs["sample_rate"]
    outputs_dir = output_dir / "outputs" / f"step={step}" / split_name
    outputs_dir.mkdir(parents=True, exist_ok=True)

    skip_n = max(1, len(dataset) // n_samples)
    sample_indices = list(range(0, len(dataset), skip_n))[:n_samples]

    for idx in sample_indices:
        data = dataset[idx]
        audio_path = data.get("audio_path", "unknown")
        audio_paths = data.get("audio_paths", [])
        midi_path = data.get("midi_path", "unknown")
        midi_paths = data.get("midi_paths", [])
        input_track_ids = data.get("input_track_ids", [])
        target_track_ids = data.get("target_track_ids", [])
        question = data.get("question", "Music transcription.")
        gt_tokens = data.get("token", [])

        logger.info("=== Transcription sample split=%s idx=%d step=%d ===", split_name, idx, step)
        logger.info("  audio_path: %s", audio_path)
        if len(audio_paths) > 0:
            logger.info("  audio_paths (%d): %s", len(audio_paths), audio_paths)
        logger.info("  midi_path:  %s", midi_path)
        if len(midi_paths) > 0:
            logger.info("  midi_paths (%d): %s", len(midi_paths), midi_paths)
        if len(input_track_ids) > 0:
            logger.info("  input_track_ids (%d): %s", len(input_track_ids), input_track_ids)
        if len(target_track_ids) > 0:
            logger.info("  target_track_ids (%d): %s", len(target_track_ids), target_track_ids)
        logger.info("  question:   %s", question)
        logger.info("  GT tokens (%d):", len(gt_tokens))
        format_str = format_tokens_by_event(gt_tokens, include_program=include_program)
        logger.info(f"  GT notes: {len(format_str.splitlines())}")
        sample = "\n".join(format_str.splitlines()[:10])
        logger.info(f"Sample GT tokens:\n{sample}\n...")

        # Prepare audio tensor (1, 1, samples)
        audio = data["audio"]  # (c, samples) numpy or tensor
        if not isinstance(audio, torch.Tensor):
            audio = torch.Tensor(audio)
        audio_tensor = audio.unsqueeze(0).to(device)  # (1, c, samples)

        # Save 5s audio clip
        clip_path = outputs_dir / f"sample_{idx}.wav"
        audio_np = audio.squeeze().cpu().numpy()
        sf.write(str(clip_path), audio_np, sr)

        # Save ground truth MIDI from tokens
        gt_midi_path = str(outputs_dir / f"sample_{idx}_gt.mid")
        if gt_tokens:
            tokens_to_midi(tokens=gt_tokens, fps=configs["fps"],
                           output_path=gt_midi_path,
                           include_program=include_program,
                           write_program_tracks=write_program_tracks)

        # Run constrained decoding
        midi_out_path = str(outputs_dir / f"sample_{idx}.mid")
        result = transcribe_audio(
            audio_encoder=audio_encoder,
            llm=llm,
            tokenizer=tokenizer,
            audio_tensor=audio_tensor,
            configs=configs,
            output_midi_path=midi_out_path,
            question=question,
            temperature=1.0,
            top_k=1,
        )

        logger.info("  Output tokens (%d):", len(result["tokens"]))
        result_format_str = format_tokens_by_event(result["tokens"], include_program=include_program)
        logger.info(f"  Output notes: {len(result_format_str.splitlines())}")
        sample = "\n".join(result_format_str.splitlines()[:10])
        logger.info(f"Sample output tokens:\n{sample}\n...")
       # logger.info("\n%s", format_tokens_by_event(result["tokens"], include_program=include_program))
        logger.info("  Violations: %d/%d", result["violations"], result["total_topk"])
        logger.info("  Saved audio clip: %s", clip_path)
        logger.info("  Saved GT MIDI: %s", gt_midi_path)
        logger.info("  Saved output MIDI: %s", midi_out_path)

    def inference_fn(item: dict) -> list[str]:
        audio = item["audio"]
        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio)
        audio_tensor = audio.unsqueeze(0).to(device)
        question = item.get("question", "Music transcription.")
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

    if not run_batch_eval:
        logger.info("Skip BatchEval for split=%s at step=%d", split_name, step)
        return {}

    eval_max_samples = None
    if "transcription_eval_max_samples" in configs["train"]:
        eval_max_samples = int(configs["train"]["transcription_eval_max_samples"])

    summary = batch_evaluate(
        dataset=dataset,
        inference_fn=inference_fn,
        fps=float(configs["fps"]),
        include_program=include_program,
        max_samples=eval_max_samples,
        verbose=False,
    )

    onset_f1 = float(summary["note_onset"]["f1"])
    offset_f1 = float(summary["note_offset"]["f1"])
    program_f1 = 0.0
    if "program_aware" in summary:
        program_f1 = float(summary["program_aware"]["f1"])

    logger.info(
        "BatchEval split=%s step=%d | onset_f1=%.6f | offset_f1=%.6f | program_f1=%.6f",
        split_name,
        step,
        onset_f1,
        offset_f1,
        program_f1,
    )

    return {
        "batch_eval/onset_f1": onset_f1,
        "batch_eval/offset_f1": offset_f1,
        "batch_eval/program_f1": program_f1,
    }


def get_dataset(
    configs: dict, 
    split: str,
    use_crop: bool = True,
) -> Dataset:
    r"""Get datasets."""

    from audidata.io.crops import RandomCrop, StartCrop
    from audidata.transforms import Mono, TextNormalization, TimeShift

    sr = configs["sample_rate"]
    clip_duration = configs["clip_duration"]
    include_program = configs["midi_include_program"] if "midi_include_program" in configs else False
    include_drum = configs["include_drum"] if "include_drum" in configs else True
    datasets_split = f"{split}_datasets"

    datasets = []

    for name in configs[datasets_split].keys():
        if name == "GTZAN":
            from audio_understanding.datasets.gtzan import GTZAN

            dataset = GTZAN(
                root=configs[datasets_split][name]["root"],
                split=configs[datasets_split][name]["split"],
                sr=sr,
                crop=RandomCrop(clip_duration=clip_duration) if use_crop else None,
                transform=Mono(),
            )
            datasets.append(dataset)

        elif name == "LibriSpeech":
            from audio_understanding.datasets.librispeech import LibriSpeech

            dataset = LibriSpeech(
                root=configs[datasets_split][name]["root"],
                split=configs[datasets_split][name]["split"],
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration) if use_crop else None,
                transform=[Mono(), TimeShift(sr=sr, shift=(0.0, 0.5))],
            )
            datasets.append(dataset)

        elif name == "Clotho":
            from audio_understanding.datasets.clotho import Clotho

            dataset = Clotho(
                root=configs[datasets_split][name]["root"],
                split=configs[datasets_split][name]["split"],
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration) if use_crop else None,
                transform=[Mono(), TimeShift(sr=sr, shift=(0.0, 0.5))],
                target_transform=TextNormalization(),
            )
            datasets.append(dataset)

        elif name == "MAESTRO":
            from audio_understanding.datasets.maestro import MAESTRO
            from audio_understanding.target_transforms.midi import MIDI2Tokens

            if configs["midi_to_tokens"] == "MIDI2Tokens":
                midi_transform = MIDI2Tokens(fps=configs["fps"], include_program=include_program)
            else:
                raise NotImplementedError

            dataset = MAESTRO(
                root=configs[datasets_split][name]["root"],
                split=configs[datasets_split][name]["split"],
                sr=sr,
                crop=RandomCrop(clip_duration=clip_duration, end_pad=clip_duration - 0.1) if use_crop else None,
                transform=Mono(),
                load_target=True,
                extend_pedal=True,
                include_program=include_program,
                target_transform=midi_transform,
            )
            datasets.append(dataset)

        elif name == "Slakh2100":
            from audio_understanding.datasets.slakh2100 import Slakh2100
            from audio_understanding.target_transforms.midi import MIDI2Tokens

            if configs["midi_to_tokens"] == "MIDI2Tokens":
                midi_transform = MIDI2Tokens(fps=configs["fps"], include_program=include_program)
            else:
                raise NotImplementedError

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
                include_drum=include_drum,
                keep_track_info=keep_track_info,
                question_config=question_config,
            )
            datasets.append(dataset)

        elif name == "AudioCaps":
            from audio_understanding.datasets.audiocaps import AudioCaps

            dataset = AudioCaps(
                root=configs[datasets_split][name]["root"],
                split=configs[datasets_split][name]["split"],
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration) if use_crop else None,
                transform=Mono(),
                target_transform=TextNormalization(),
            )
            datasets.append(dataset)

        elif name == "WavCaps":
            from audio_understanding.datasets.wavcaps import WavCaps

            dataset = WavCaps(
                root=configs[datasets_split][name]["root"],
                sr=sr,
                crop=StartCrop(clip_duration=clip_duration) if use_crop else None,
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
    if "ckpt_path" not in configs["audio_encoder"]:
        configs["audio_encoder"]["ckpt_path"] = None
    name = configs["audio_encoder"]["name"]
    sr = configs["sample_rate"]
    trainable = configs["audio_encoder"]["trainable"]
    use_decoder = configs["audio_encoder"].get("use_decoder", True)
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

    elif name == "Conformer2D":
        from audio_understanding.audio_encoders.conformer2d import Conformer2D
        model = Conformer2D(sr=sr, trainable=trainable, use_decoder=use_decoder)
        
    elif name == "Conformer2D_nopool":
        from audio_understanding.audio_encoders.conformer2d_nopool import Conformer2D #* slakh and maestro only differ between heads, don't matter that much
        model = Conformer2D(sr=sr, trainable=trainable, use_decoder=use_decoder)

    elif name == "MERT":
        from audio_understanding.audio_encoders.mert import MERT

        target_layer = configs["audio_encoder"]["target_layer"] if "target_layer" in configs["audio_encoder"] else -1
        pretrained_model_name = configs["audio_encoder"]["pretrained_model_name"] if "pretrained_model_name" in configs["audio_encoder"] else "m-a-p/MERT-v1-330M"
        model = MERT(
            sr=sr,
            trainable=trainable,
            target_layer=target_layer,
            pretrained_model_name=pretrained_model_name,
        )
    elif name == "MuQ":
        from audio_understanding.audio_encoders.muq import MuQ

        model = MuQ(sr=sr, trainable=trainable)
    
    else:
        raise ValueError(name)

    for param in model.parameters():
        param.requires_grad = trainable

    if configs["audio_encoder"]["ckpt_path"]:
        ckpt = torch.load(configs["audio_encoder"]["ckpt_path"], map_location="cpu")
        # Filter out keys with mismatched shapes (e.g. rope buffer size differs)
        model_state = model.state_dict()
        skipped_shape = {k for k, v in ckpt.items() if k in model_state and v.shape != model_state[k].shape}
        unexpected = {k for k in ckpt if k not in model_state}
        missing = {k for k in model_state if k not in ckpt}
        if skipped_shape:
            logging.warning("Skipped keys due to shape mismatch: (ckpt, model) %s", {k: (ckpt[k].shape, model_state[k].shape) for k in skipped_shape})
        if unexpected:
            logging.warning("Unexpected keys in checkpoint (ignored): %s", unexpected)
        if missing:
            logging.warning("Missing keys in checkpoint (using init): %s", missing)
        filtered_ckpt = {k: v for k, v in ckpt.items() if k in model_state and k not in skipped_shape}
        model.load_state_dict(filtered_ckpt, strict=False) #* there is no state dict key for audio encoder key 
        logging.info("Loaded audio encoder weights from %s", configs["audio_encoder"]["ckpt_path"])
        
    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        if "audio_encoder" in ckpt:
            model.load_state_dict(ckpt["audio_encoder"])
            logging.info("Loaded audio encoder weights from joint checkpoint %s", ckpt_path)

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
    if "ckpt_path" not in configs["llm"]:
        configs["llm"]["ckpt_path"] = None
    trainable = configs["llm"]["trainable"]

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

    elif name == "T5":
        from audio_understanding.llm.t5 import T5, T5Config

        block_size = configs["llm"]["block_size"]
        n_layer = configs["llm"]["n_layer"]
        n_head = configs["llm"]["n_head"]
        n_embd = configs["llm"]["n_embd"]

        config = T5Config(
            block_size=block_size,
            audio_latent_dim=audio_latent_dim,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
        )
        model = T5(config=config)

    else:
        raise ValueError(name)

    for param in model.parameters():
        param.requires_grad = trainable

    if configs["llm"]["ckpt_path"]:
        ckpt = torch.load(configs["llm"]["ckpt_path"], map_location="cpu")
        model_state = model.state_dict()
        skipped_shape = {k for k, v in ckpt.items() if k in model_state and v.shape != model_state[k].shape}
        unexpected = {k for k in ckpt if k not in model_state}
        missing = {k for k in model_state if k not in ckpt}
        if skipped_shape:
            logging.warning("LLM skipped keys due to shape mismatch: %s", {k: (ckpt[k].shape, model_state[k].shape) for k in skipped_shape})
        if unexpected:
            logging.warning("LLM unexpected keys in checkpoint (ignored): %s", unexpected)
        if missing:
            logging.warning("LLM missing keys in checkpoint (using init): %s", missing)
        filtered_ckpt = {k: v for k, v in ckpt.items() if k in model_state and k not in skipped_shape}
        model.load_state_dict(filtered_ckpt, strict=False) #* there is no state dict key for llm key 
        logging.info("Loaded LLM weights from %s", configs["llm"]["ckpt_path"])
    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        if "llm" in ckpt:
            model.load_state_dict(ckpt["llm"])
            logging.info("Loaded LLM weights from joint checkpoint %s", ckpt_path)

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

    elif name in ["MAESTRO", "Slakh2100"]:
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


@hydra.main(version_base=None, config_path="configs", config_name="piano_transcription_maestro")
def main(cfg: DictConfig) -> None:
    main_func(cfg)


if __name__ == "__main__":
    main()
