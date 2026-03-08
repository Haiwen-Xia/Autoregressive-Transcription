from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional

import yaml
import os
import librosa
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate_fn_map
import random

from audidata.io.audio import load
from audidata.io.crops import RandomCrop
from audidata.transforms.audio import Mono
from audidata.transforms.midi import PianoRoll
from audidata.io.midi import read_multi_track_midi, read_single_track_midi, clip_notes
from audidata.collate.base import collate_list_fn
from audidata.utils import call


default_collate_fn_map.update({list: collate_list_fn})


def load_target_from_mix_midi(
    mixed_midi_path: str,
    start_time: float,
    duration: float,
) -> dict:
    midi_tracks = read_multi_track_midi(midi_path=mixed_midi_path)

    data = {
        "start_time": start_time,
        "duration": duration,
        "tracks": [],
    }

    for idx, midi_track in enumerate(midi_tracks):
        track = {
            "track_name": "mix_track_{:02d}".format(idx),
            "inst_class": "unknown",
            "is_drum": bool(midi_track["is_drum"]),
            "plugin_name": "unknown",
            "program_num": int(midi_track["program"]),
            "note": midi_track["notes"],
            "pedal": midi_track["pedals"],
        }
        data["tracks"].append(track)

    return data


def load_target_from_midi_files(
    meta: dict,
    midis_dir: str,
    track_ids: list[str],
    start_time: float,
    duration: float,
    extend_pedal: bool,
) -> dict:

    data = {
        "start_time": start_time,
        "duration": duration,
        "tracks": []
    }

    for stem_name, stem_data in meta["stems"].items():

        if stem_name not in track_ids:
            continue

        if not stem_data["midi_saved"]:
            continue

        inst_class = stem_data["inst_class"]
        is_drum = stem_data["is_drum"]
        plugin_name = stem_data["plugin_name"]
        program_num = stem_data["program_num"]

        midi_path = Path(midis_dir, "{}.mid".format(stem_name))

        notes, pedals = read_single_track_midi(
            midi_path=str(midi_path),
            extend_pedal=extend_pedal
        )

        track = {
            "track_name": stem_name,
            "inst_class": inst_class,
            "is_drum": is_drum,
            "plugin_name": plugin_name,
            "program_num": program_num,
            "note": notes,
            "pedal": pedals,
        }

        data["tracks"].append(track)

    return data


class Slakh2100(Dataset):
    r"""Slakh2100 [1] is a multiple track MIDI-audio paired dataset containing
    145 hours of 2,100 audio audio files rendered by MIDI files. Audios are 
    sampled at 44,100 Hz. After decompression, the dataset is 101 GB.

    [1] E. Manilow, Cutting music source separation some Slakh: A dataset to 
    study the impact of training data quality and quantity, WASPAA, 2019

    After decompression, dataset looks like:

        dataset_root (131 GB)
        ├── train (1500 songs)
        │   ├── Track00001
        │   │   ├── all_src.mid
        │   │   ├── metadata.yaml
        │   │   ├── MIDI
        │   │   │   ├── S00.mid
        │   │   │   ├── S01.mid
        │   │   │   └── ...
        │   │   ├── mix.flac
        │   │   └── stems
        │   │       ├── S00.flac
        │   │       ├── S01.flac
        │   │       └── ...
        │   ├── Track00002
        │   └── ...
        ├── validation (375 songs)
        └── test (225 songs) 
    """

    url = "https://zenodo.org/records/4599666"

    duration = 521806.45  # Dataset duration (s), 145 hours, including training, 
    # validation, and testing.

    def __init__(
        self, 
        root: str, 
        split: str = "train",
        sr: float = 16000,
        crop: Optional[Callable] = RandomCrop(clip_duration=10., end_pad=9.9),
        transform: Optional[Callable] = Mono(),
        target: bool = True,
        extend_pedal: bool = True,
        target_transform: Optional[Callable | list[Callable]] = PianoRoll(fps=100, pitches_num=128),
        sample_num: int = 1,
        mode: str = "all",  # options: all/all2all, single, all2one, all2several, mix2one
        keep_track_info: bool = False, #* 没啥用
        question_config: Optional[dict] = None,
    ):

        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.target = target
        self.extend_pedal = extend_pedal
        self.transform = transform
        self.target_transform = target_transform
        valid_modes = [
            "all", "all2all", "single",
            "all2one",
            "all2several", 
            "mix2one", 
            "rand_mix",
        ]
        assert mode in valid_modes, "Invalid mode!"
        assert sample_num >= 1
        self.mode = mode
        self.sample_num = sample_num
        self.keep_track_info = keep_track_info
        self.question_config = question_config if question_config is not None else {}
        
        audios_dir = Path(self.root, self.split)
        self.meta_dict = {"audio_name": sorted(os.listdir(audios_dir))}
        
    def __getitem__(self, index: int) -> dict:

        prefix = Path(self.root, self.split, self.meta_dict["audio_name"][index])

        meta_yaml = Path(prefix, "metadata.yaml")
        mix_midi_path = Path(prefix, "all_src.mid")
        stems_dir = Path(prefix, "stems")
        midis_dir = Path(prefix, "MIDI")
        meta = self.load_meta(meta_yaml)
        audio_track_ids = self.get_available_audio_track_ids(stems_dir, meta)
        midi_track_ids = self.get_available_midi_track_ids(midis_dir, meta)
        candidate_track_ids = sorted(set(audio_track_ids).intersection(midi_track_ids))
        assert len(candidate_track_ids) > 0

        input_track_ids, target_track_ids = self.sample_input_and_target_track_ids(candidate_track_ids)

        if self.mode == "single":
            audio_path = Path(stems_dir, "{}.flac".format(input_track_ids[0]))
            midi_path = Path(midis_dir, "{}.mid".format(target_track_ids[0]))
            audio_data = self.load_audio(path=str(audio_path))
            audio_paths = [str(audio_path.resolve())]
            midi_paths = [str(midi_path)]

        elif self.mode in ["all", "all2all", "all2one", "all2several"]:
            audio_path = Path(prefix, "mix.flac")
            midi_path = Path(" + ".join(["{}.mid".format(track_id) for track_id in target_track_ids]))
            audio_data = self.load_audio(path=str(audio_path))
            audio_paths = [str(audio_path.resolve())]
            midi_paths = [str(Path(midis_dir, "{}.mid".format(track_id))) for track_id in target_track_ids]

        else:
            stem_audio_paths = [Path(stems_dir, "{}.flac".format(track_id)) for track_id in input_track_ids]
            audio_data = self.load_audio_mix(paths=stem_audio_paths)
            audio_paths = [str(path.resolve()) for path in stem_audio_paths]
            audio_path = Path(" + ".join(audio_paths))
            midi_path = Path(" + ".join(["{}.mid".format(track_id) for track_id in target_track_ids]))
            midi_paths = [str(Path(midis_dir, "{}.mid".format(track_id))) for track_id in target_track_ids]


        full_data = {
            "dataset_name": "Slakh2100",
            "audio_path": str(audio_path.resolve()) if len(audio_paths) == 1 else str(audio_path),
            "audio_paths": audio_paths,
            "midi_path": str(midi_path),
            "midi_paths": midi_paths,
            "input_track_ids": input_track_ids,
            "target_track_ids": target_track_ids,
            "track_ids": target_track_ids,
        }

        full_data.update(audio_data)

        # Load question
        question_data = self.load_question_data(meta=meta, track_ids=target_track_ids)
        full_data.update(question_data)

        # Load target
        if self.target:
            target_data = self.load_target_data(
                meta=meta,
                mixed_midi_path=str(mix_midi_path),
                midis_dir=str(midis_dir),
                track_ids=target_track_ids,
                start_time=audio_data["start_time"],
                duration=audio_data["duration"],
            )
            full_data.update(target_data)

        return full_data

    def __len__(self):

        audios_num = len(self.meta_dict["audio_name"])

        return audios_num

    def load_meta(self, meta_yaml: Path) -> dict:
        with open(meta_yaml, "r") as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)

        return meta

    def get_available_audio_track_ids(self, stems_dir: Path, meta: dict) -> list[str]:
        track_ids = []

        for path in sorted(stems_dir.glob("*.flac")):
            stem_name = path.stem
            if stem_name not in meta["stems"]:
                continue
            if not meta["stems"][stem_name].get("audio_rendered", True):
                continue
            track_ids.append(stem_name)

        assert len(track_ids) > 0

        return sorted(track_ids)

    def get_available_midi_track_ids(self, midis_dir: Path, meta: dict) -> list[str]:
        track_ids = []

        for stem_name, stem_data in meta["stems"].items():
            if not stem_data["midi_saved"]:
                continue
            midi_path = Path(midis_dir, "{}.mid".format(stem_name))
            if not midi_path.exists():
                continue
            track_ids.append(stem_name)

        assert len(track_ids) > 0

        return sorted(track_ids)

    def sample_track_ids(self, all_track_ids: list[str]) -> list[str]:

        if self.mode == "all":
            return all_track_ids

        if self.mode == "single":
            return [random.choice(all_track_ids)]

        sample_n = min(self.sample_num, len(all_track_ids))
        sampled = random.sample(all_track_ids, sample_n)

        return sorted(sampled)

    def sample_input_and_target_track_ids(self, all_track_ids: list[str]) -> tuple[list[str], list[str]]:
        if self.mode in ["all", "all2all"]:
            return sorted(all_track_ids), sorted(all_track_ids)

        if self.mode == "single":
            chosen = [random.choice(all_track_ids)]
            return chosen, chosen

        if self.mode in ["all2one"]:
            target_ids = [random.choice(all_track_ids)]
            return sorted(all_track_ids), target_ids

        if self.mode in ["all2several"]:
            sample_n = random.randint(1, min(self.sample_num, len(all_track_ids)))
            target_ids = sorted(random.sample(all_track_ids, sample_n))
            return sorted(all_track_ids), target_ids

        if self.mode in ["mix2one"]:
            target_ids = [random.choice(all_track_ids)]
            remaining = [track_id for track_id in all_track_ids if track_id not in target_ids]
            if len(remaining) == 0:
                input_ids = target_ids
            else:
                mix_n = min(self.sample_num, len(remaining))
                input_ids = sorted(random.sample(remaining, mix_n))
            return input_ids, target_ids

        if self.mode == "rand_mix":
            sample_n = random.randint(1, min(self.sample_num, len(all_track_ids)))
            chosen = sorted(random.sample(all_track_ids, sample_n))
            return chosen, chosen

        raise ValueError(self.mode)

    def load_question_data(self, meta: dict, track_ids: list[str]) -> dict:

        default_question = self.question_config.get("default", "Transcribe this audio.")
        question = default_question

        question_instrument_repr = self.question_config.get("instrument_repr", "name")
        question_template = self.question_config.get("template")
        instrument_info_dropout = float(self.question_config.get("instrument_info_dropout", 1.0))
        include_info = (random.random() >= instrument_info_dropout)
        instru_dropout = self.question_config.get("drop_instrument", True)

        if include_info:
            instruments = []
            for track_id in track_ids:
                stem_data = meta["stems"][track_id]
                if question_instrument_repr == "name":
                    instruments.append(stem_data["inst_class"])
                elif question_instrument_repr == "program":
                    instruments.append("program {}".format(stem_data["program_num"]))
                else:
                    instruments.append("{} (program {})".format(stem_data["inst_class"], stem_data["program_num"]))
            random.shuffle(instruments)
            if instru_dropout:
                rand_target_length = random.randint(1, len(instruments))
            else:
                rand_target_length = len(instruments)
                
            rand_target_length = random.randint(1, len(instruments))

            rand_targets = ", ".join(instruments[:rand_target_length])
            targets = ", ".join(instruments)
            if question_template is not None:
                question = question_template.format(targets=targets)
                return {"question": question}

            if self.mode in ["all2several", "all2one", "mix2one"]: #* since here is to transcribe mixes only, ...
                question = "Transcribe only the target instruments from this audio. Target instruments: {}.".format(targets)
            else:
                question = "Transcribe this audio. Target instruments may include: {}.".format(rand_targets)

        return {"question": question}
    
    def load_audio(self, path: str) -> dict:

        audio_duration = librosa.get_duration(path=path)
        
        if self.crop:
            start_time, clip_duration = self.crop(audio_duration=audio_duration)
        else:
            start_time = 0.
            clip_duration = None

        audio = load(
            path=path, 
            sr=self.sr, 
            offset=start_time, 
            duration=clip_duration
        )
        # shape: (channels, audio_samples)

        data = {
            "audio": audio, 
            "start_time": start_time,
            "duration": clip_duration if clip_duration else audio_duration
        }

        if self.transform is not None:
            data["audio"] = call(transform=self.transform, x=data["audio"])

        return data

    def load_audio_mix(self, paths: list[Path]) -> dict:
        assert len(paths) > 0

        first_duration = librosa.get_duration(path=str(paths[0]))

        if self.crop:
            start_time, clip_duration = self.crop(audio_duration=first_duration)
        else:
            start_time = 0.
            clip_duration = None

        mixed_audio = None
        for path in paths:
            audio = load(
                path=str(path),
                sr=self.sr,
                offset=start_time,
                duration=clip_duration,
            )
            if mixed_audio is None:
                mixed_audio = audio
            else:
                min_len = min(mixed_audio.shape[-1], audio.shape[-1])
                mixed_audio = mixed_audio[..., :min_len] + audio[..., :min_len]

        assert mixed_audio is not None

        peak = np.max(np.abs(mixed_audio))
        if peak > 1.0:
            mixed_audio = mixed_audio / peak

        data = {
            "audio": mixed_audio,
            "start_time": start_time,
            "duration": clip_duration if clip_duration else first_duration,
        }

        if self.transform is not None:
            data["audio"] = call(transform=self.transform, x=data["audio"])

        return data

    def compress_to_one_track(self, data: dict) -> dict:
        note_items = []
        pedal_items = []

        for track in data["tracks"]:
            notes = clip_notes(track["note"], data["start_time"], data["duration"])
            pedals = clip_notes(track["pedal"], data["start_time"], data["duration"])

            for note in notes:
                note_items.append(
                    {
                        "note": note,
                        "program_num": track["program_num"],
                        "inst_class": track["inst_class"],
                        "track_name": track["track_name"],
                        "is_drum": track["is_drum"],
                    }
                )

            for pedal in pedals:
                pedal_items.append(
                    {
                        "pedal": pedal,
                        "program_num": track["program_num"],
                        "track_name": track["track_name"],
                    }
                )

        note_items.sort(key=lambda item: (item["note"].start, item["note"].pitch, item["program_num"], item["note"].end))
        pedal_items.sort(key=lambda item: (item["pedal"].start, item["program_num"], item["pedal"].end))

        data.update(note=[item["note"] for item in note_items])
        data.update(pedal=[item["pedal"] for item in pedal_items])
        data.update(note_program=[item["program_num"] for item in note_items])
        data.update(note_inst_class=[item["inst_class"] for item in note_items])
        data.update(note_track_name=[item["track_name"] for item in note_items])
        data.update(note_is_drum=[item["is_drum"] for item in note_items])
        data.update(pedal_program=[item["program_num"] for item in pedal_items])
        data.update(pedal_track_name=[item["track_name"] for item in pedal_items])

        if not self.keep_track_info:
            data.pop("tracks")

        return data

    def load_target_data(
        self, 
        meta: dict,
        mixed_midi_path: str,
        midis_dir: str,
        track_ids: list[str],
        start_time: float, 
        duration: float,
        compress_to_one_track: bool = True
    ) -> dict:
        use_mix_source = self.mode in ["all", "all2all"]

        if use_mix_source:
            data = load_target_from_mix_midi(
                mixed_midi_path=mixed_midi_path,
                start_time=start_time,
                duration=duration,
            )
        else:
            data = load_target_from_midi_files(
                meta=meta,
                midis_dir=midis_dir,
                track_ids=track_ids,
                start_time=start_time,
                duration=duration,
                extend_pedal=self.extend_pedal,
            )

        if compress_to_one_track:
            data = self.compress_to_one_track(data)

        if self.target_transform:
            data = call(transform=self.target_transform, x=data)

        if "token" in data:
            assert isinstance(data["token"], list)

        return data

    def evaluate(
        self,
        data: dict,
        output_tokens: list[str],
        fps: float,
        include_program: bool = False,
    ) -> dict:
        r"""Evaluate model output tokens against ground truth for one sample.

        Always computes note onset F1 and note-with-offset F1.

        When instrument metadata is available in *data* (i.e. ``"note_program"``
        is present) and either *include_program* is ``True`` or the sample
        targets a single instrument (X2one setting such as ``all2one`` /
        ``single`` mode), program-aware F1 and per-instrument metrics are also
        returned.

        **Modes without instrument prediction** (e.g. ``all``/``all2all``,
        ``rand_mix``): pass ``include_program=False``.  Only onset and offset
        F1 are returned.

        **Modes with instrument prediction** (``include_program=True``): the
        model output tokens contain ``program=X`` fields.  Program-aware F1
        and per-instrument metrics are returned.

        **X2one modes** (``all2one`` / ``single``): the model does not predict
        instrument labels, but the ground truth always targets a single
        instrument whose identity is known from the dataloader.  This is
        detected automatically when all reference notes share the same program,
        and per-instrument metrics are computed by assigning that program to
        all estimated notes.

        Args:
            data: dict returned by :meth:`__getitem__` (must contain ``"note"``,
                ``"start_time"``, and—for per-instrument evaluation—
                ``"note_program"`` and ``"note_is_drum"``).  The ``"note"``
                value should be a list of Note objects (use ``MIDI2Tokens`` or
                ``target_transform=None`` when creating the dataset).
            output_tokens: flat list of MIDI token strings produced by the
                model.
            fps: frames per second used when encoding tokens (must match the
                value used during training).
            include_program: whether *output_tokens* contain ``program=X``
                fields.

        Returns:
            Dict always containing:

            * ``"note_onset"``  – ``{"precision": float, "recall": float, "f1": float}``
            * ``"note_offset"`` – ``{"precision": float, "recall": float, "f1": float}``

            And, when applicable:

            * ``"program_aware"``  – ``{"precision": float, "recall": float, "f1": float}``
            * ``"per_instrument"`` – per-instrument onset F1 dict (see
              :func:`~audio_understanding.eval.transcription.metrics.per_instrument_metrics`)
        """
        from audio_understanding.eval.transcription.metrics import (
            parse_tokens_to_notes,
            note_onset_f1,
            note_with_offset_f1,
            program_aware_f1,
            per_instrument_metrics,
        )

        start_time = data["start_time"]
        notes = data.get("note", [])
        note_programs = data.get("note_program", [])
        note_is_drum = data.get("note_is_drum", [])
        note_inst_class = data.get("note_inst_class", [])

        has_inst_meta = len(note_programs) == len(notes) and len(notes) > 0

        # Build reference note list (times relative to clip start)
        ref_notes = []
        for idx, note in enumerate(notes):
            note_dict: dict = {
                "onset_time":  note.start - start_time,
                "offset_time": note.end   - start_time,
                "pitch":       note.pitch,
                "velocity":    note.velocity,
            }
            if has_inst_meta:
                note_dict["program"]    = int(note_programs[idx])
                note_dict["is_drum"]    = bool(note_is_drum[idx])
                note_dict["inst_class"] = (
                    note_inst_class[idx] if note_inst_class else "unknown"
                )
            ref_notes.append(note_dict)

        # Parse model output tokens into note dicts
        est_notes = parse_tokens_to_notes(
            tokens=output_tokens,
            fps=fps,
            include_program=include_program,
            start_time=0.0,
        )

        results = {
            "note_onset":  note_onset_f1(ref_notes, est_notes),
            "note_offset": note_with_offset_f1(ref_notes, est_notes),
        }

        # Detect X2one scenario: model has no program output but ground truth
        # targets exactly one instrument — assign that program to all estimates.
        single_program: Optional[int] = None
        single_is_drum: bool = False
        if has_inst_meta and not include_program:
            unique_non_drum_progs = set(
                int(p)
                for p, d in zip(note_programs, note_is_drum)
                if not d
            )
            all_drum = all(bool(d) for d in note_is_drum)
            if len(unique_non_drum_progs) == 1 or (
                len(unique_non_drum_progs) == 0 and all_drum
            ):
                single_program = int(note_programs[0])
                single_is_drum = bool(note_is_drum[0])

        if include_program or single_program is not None:
            if single_program is not None and not include_program:
                # Assign the known single program to all estimated notes
                est_notes_prog = [
                    {**n, "program": single_program, "is_drum": single_is_drum}
                    for n in est_notes
                ]
            else:
                est_notes_prog = est_notes

            results["program_aware"]  = program_aware_f1(ref_notes, est_notes_prog)
            results["per_instrument"] = per_instrument_metrics(ref_notes, est_notes_prog)

        return results


'''
from audidata.transforms.midi import ReductToPianoRoll

from piano_transcription.update_collate import default_collate_fn_map

target_transform = ReductToPianoRoll()

dataset = Slakh2100(
    root=configs[datasets_split][name]["root"],
    split="train",
    sr=sr,
    crop=RandomCrop(clip_duration=10., end_pad=9.9),
    target_transform=target_transform,
)

return dataset

'''    