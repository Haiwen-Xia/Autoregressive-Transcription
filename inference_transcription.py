"""Music transcription inference with constrained decoding.

Constrained decoding enforces the MIDI token grammar (MIDI2Tokens construction order):
    time_first: time_index -> name -> pitch -> velocity (onset only) -> program (optional)
    name_first: name -> time_index -> pitch -> velocity (onset only) -> program (optional)
and tracks how many top-k candidates violate the grammar at each step.

Usage:
    python inference_transcription.py \
        --config_yaml configs/piano_transcription_maestro.yaml \
        --ckpt_path checkpoints/train/.../ckpt/step_XXXX.pth \
        --audio_path /path/to/audio.wav \
        --output_path output.mid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
import torch
from torch import nn

from audio_understanding.target_transforms.midi_constrained import MidiConstrainedDecoder


# ── callable core function ──────────────────────────────────────────────

def transcribe_audio(
    audio_encoder: nn.Module,
    llm: nn.Module,
    tokenizer,
    audio_tensor: torch.Tensor,
    configs: dict,
    output_midi_path: str | None = None,
    question: str = "Music transcription.",
    temperature: float = 1.0,
    top_k: int = 1,
) -> dict:
    """Run constrained music transcription on a single audio clip.

    Args:
        audio_encoder: pretrained audio encoder (already on device)
        llm: Llama LLM (already on device)
        tokenizer: BertMIDI tokenizer
        audio_tensor: (1, 1, samples) mono audio tensor on device
        configs: parsed yaml config dict (needs fps, max_question_len, max_answering_len, midi_include_program)
        output_midi_path: write MIDI file here if not None
        question: question text fed to the model
        temperature: softmax temperature
        top_k: top-k width (also used for violation counting)

    Returns:
        dict with keys:
            tokens     : list[str] generated token strings
            violations : int, number of top-k violations
            total_topk : int, total top-k candidates checked
            midi_path  : str | None, path to written MIDI (if output_midi_path given)
    """
    device = audio_tensor.device
    include_program = configs.get("midi_include_program", False)
    midi_event_token_order = configs.get("midi_event_token_order", "time_first")

    # Encode audio
    audio_latent = audio_encoder.encode(audio=audio_tensor, train_mode=False)  # (1, t, d)

    # Question IDs
    question_ids = tokenizer.texts_to_ids(
        texts=[question],
        fix_length=configs["max_question_len"],
    ).to(device)  # (1, t)

    # Start token
    answering_ids = torch.LongTensor([[tokenizer.cls_token_id]]).to(device)  # (1, 1)

    seqs = [audio_latent, question_ids, answering_ids]
    seq_types = ["audio", "id", "id"]

    # Constrained decoding
    constraint = MidiConstrainedDecoder(
        tokenizer=tokenizer,
        vocab_size=len(tokenizer),
        include_program=include_program,
        token_order=midi_event_token_order,
        device=device,
    )

    was_training = llm.training
    llm.eval()
    with torch.no_grad():
        output_seqs = llm.generate_constrained(
            seqs=seqs,
            seq_types=seq_types,
            max_new_ids=configs["max_answering_len"],
            constraint=constraint,
            temperature=temperature,
            top_k=top_k,
        )
    if was_training:
        llm.train()

    # Decode output IDs → token strings
    out_ids = output_seqs[2][0]  # (t,)
    tokens = tokenizer.tok.convert_ids_to_tokens(out_ids)
    if tokens and tokens[0] == "[CLS]":
        tokens = tokens[1:]

    # Write MIDI
    midi_path = None
    if output_midi_path is not None:
        fps = configs["fps"]
        write_program_tracks = bool(configs.get("midi_write_program_tracks", include_program))
        tokens_to_midi(tokens=tokens, fps=fps, output_path=output_midi_path,
                       include_program=include_program,
                       write_program_tracks=write_program_tracks)
        midi_path = output_midi_path

    return {
        "tokens": tokens,
        "violations": constraint.violations,
        "total_topk": constraint.total_topk_candidates,
        "midi_path": midi_path,
    }


# ── tokens formatting ──────────────────────────────────────────────────

def _extract_event_chunks(tokens: list[str]) -> list[list[str]]:
    chunks: list[list[str]] = []
    i = 0
    n = len(tokens)

    while i < n:
        tok = tokens[i]
        if tok in ["name=note_onset", "name=note_offset"]:
            j = i + 1
            while j < n:
                if tokens[j] in ["name=note_onset", "name=note_offset"]:
                    break
                if tokens[j].startswith("time_index=") and j + 1 < n and tokens[j + 1] in ["name=note_onset", "name=note_offset"]:
                    break
                j += 1
            chunks.append(tokens[i:j])
            i = j
            continue

        if tok.startswith("time_index=") and i + 1 < n and tokens[i + 1] in ["name=note_onset", "name=note_offset"]:
            j = i + 2
            while j < n:
                if tokens[j].startswith("time_index="):
                    break
                if tokens[j] in ["name=note_onset", "name=note_offset"]:
                    break
                j += 1
            chunks.append(tokens[i:j])
            i = j
            continue

        chunks.append([tok])
        i += 1

    return chunks


def format_tokens_by_event(tokens: list[str], include_program: bool = False) -> str:
    """Pretty-print token list, one line per event.

    Token order: time_index, name, pitch, (velocity), (program)
    onset event len = 4 + program, offset event len = 3 + program
    """
    lines = []
    chunks = _extract_event_chunks(tokens)
    for event_idx, chunk in enumerate(chunks):
        lines.append(f"  [{event_idx}] " + " | ".join(chunk))

    return "\n".join(lines)


# ── tokens → MIDI ──────────────────────────────────────────────────────

def tokens_to_midi(
    tokens: list[str],
    fps: float,
    output_path: str,
    include_program: bool = False,
    write_program_tracks: bool = False,
) -> None:
    """Convert generated token list to a MIDI file.

    Supported token order per event:
        time_first: time_index=X, name=note_onset|note_offset, ...
        name_first: name=note_onset|note_offset, time_index=X, ...
    """
    note_dict: dict[tuple[int, int], list[dict]] = {}
    chunks = _extract_event_chunks(tokens)

    for chunk in chunks:
        name_tok = None
        time_index = None
        event: dict[str, int | str] = {}

        for tok in chunk:
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            if k == "name" and v in ["note_onset", "note_offset"]:
                name_tok = "name={}".format(v)
            elif k == "time_index" and v.isdigit():
                time_index = int(v)
            elif k in ["pitch", "drum_pitch", "velocity", "program"]:
                event[k] = v

        if name_tok not in ["name=note_onset", "name=note_offset"]:
            continue
        if time_index is None:
            continue

        is_onset = name_tok == "name=note_onset"
        pitch_key = "drum_pitch" if "drum_pitch" in event else "pitch"
        if pitch_key not in event:
            continue
        pitch = int(event[pitch_key])
        program = int(event["program"]) if "program" in event else (128 if pitch_key == "drum_pitch" else 0)
        key_pitch_program = (pitch, program)

        if is_onset:
            if "velocity" not in event:
                continue
            velocity = int(event["velocity"])
            note = {
                "onset_time_index": time_index,
                "pitch": pitch,
                "velocity": velocity,
                "program": program,
            }
            if key_pitch_program not in note_dict:
                note_dict[key_pitch_program] = []
            note_dict[key_pitch_program].append(note)
        else:
            if key_pitch_program in note_dict and len(note_dict[key_pitch_program]) > 0:
                note_dict[key_pitch_program][-1]["offset_time_index"] = time_index

    # Collect all notes
    midi_data = pretty_midi.PrettyMIDI()

    if write_program_tracks and include_program:
        track_dict: dict[int, pretty_midi.Instrument] = {}
        for (_, _), events in note_dict.items():
            for e in events:
                program = int(e.get("program", 0))
                if program not in track_dict:
                    is_drum = program == 128
                    track_program = 0 if is_drum else max(0, min(127, program))
                    instrument = pretty_midi.Instrument(program=track_program)
                    instrument.is_drum = is_drum
                    track_dict[program] = instrument

                start_time = e["onset_time_index"] / fps
                end_time = e.get("offset_time_index", e["onset_time_index"] + int(fps * 0.1)) / fps
                assert 0 <= int(e["pitch"]) <= 127
                assert 0 <= int(e["velocity"]) <= 127
                note = pretty_midi.Note(
                    pitch=e["pitch"],
                    start=start_time,
                    end=end_time,
                    velocity=e["velocity"],
                )
                track_dict[program].notes.append(note)

        for program in sorted(track_dict):
            midi_data.instruments.append(track_dict[program])

    else:
        track = pretty_midi.Instrument(program=0)
        track.is_drum = False
        for (_, _), events in note_dict.items():
            for e in events:
                start_time = e["onset_time_index"] / fps
                end_time = e.get("offset_time_index", e["onset_time_index"] + int(fps * 0.1)) / fps
                assert 0 <= int(e["pitch"]) <= 127
                assert 0 <= int(e["velocity"]) <= 127
                note = pretty_midi.Note(
                    pitch=e["pitch"],
                    start=start_time,
                    end=end_time,
                    velocity=e["velocity"],
                )
                track.notes.append(note)
        midi_data.instruments.append(track)

    midi_data.write(output_path)
    print("Write out to {}".format(output_path))


# ── CLI main ────────────────────────────────────────────────────────────

def main_func(args) -> None:
    from audio_understanding.utils import parse_yaml
    from train import get_audio_encoder, get_llm, get_tokenizer

    config_yaml = args.config_yaml
    configs = parse_yaml(config_yaml)
    sr = configs["sample_rate"]
    clip_duration = configs["clip_duration"]
    clip_samples = round(clip_duration * sr)
    device = args.device

    audio_encoder = get_audio_encoder(configs=configs, ckpt_path=args.ckpt_path).to(device)
    tokenizer = get_tokenizer(configs=configs)
    llm = get_llm(
        configs=configs,
        audio_latent_dim=audio_encoder.latent_dim,
        vocab_size=len(tokenizer),
        ckpt_path=args.ckpt_path,
        audio_encoder=audio_encoder,
        tokenizer=tokenizer,
    ).to(device)

    # Load audio segment
    audio, _ = librosa.load(path=args.audio_path, sr=sr, mono=True)
    start = args.segment_idx * clip_samples
    audio_segment = audio[start : start + clip_samples]
    audio_segment = librosa.util.fix_length(data=audio_segment, size=clip_samples, axis=0)
    print("Segment {}: offset {:.2f}s, duration {:.2f}s, sr={}, samples={}".format(
        args.segment_idx, start / sr, clip_duration, sr, clip_samples))
    audio_tensor = torch.Tensor(audio_segment[None, None, :]).to(device)

    result = transcribe_audio(
        audio_encoder=audio_encoder,
        llm=llm,
        tokenizer=tokenizer,
        audio_tensor=audio_tensor,
        configs=configs,
        output_midi_path=args.output_path,
        question=args.question or "Music transcription.",
        temperature=args.temperature,
        top_k=args.top_k,
    )

    include_program = configs.get("midi_include_program", False)
    print("Generated {} tokens".format(len(result["tokens"])))
    print("Violations: {}/{}".format(result["violations"], result["total_topk"]))
    print(format_tokens_by_event(result["tokens"], include_program=include_program))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music transcription with constrained decoding")
    parser.add_argument("--config_yaml", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output.mid")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--segment_idx", type=int, default=0,
                        help="Which clip_duration-sized segment to transcribe (0-indexed)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1,
                        help="Top-k sampling; also used for violation counting")
    parser.add_argument("--question", type=str, default=None,
                        help="Override the default question text")
    args = parser.parse_args()
    main_func(args)
