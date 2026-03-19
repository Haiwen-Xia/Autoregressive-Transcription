from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio
from piano_transcription_inference import config as pt_config
from piano_transcription_inference import PianoTranscription
from piano_transcription_inference.models import Regress_onset_offset_frame_velocity_CRNN


class PianoTranscriptionCRnn(nn.Module): #! 精度

    def __init__(self, sr: float, trainable: bool, random: bool=False) -> None:
        r"""Piano transcription encoder [1]

        [1] Q. Kong, et al., High-resolution Piano Transcription with Pedals by 
        Regressing Onsets and Offsets Times, TASLP, 2022

        Code: https://github.com/qiuqiangkong/piano_transcription_inference
        """

        super().__init__()

        self.audio_sr = sr
        self.model_sr = 16000  # Piano transcription encoder sampling rate
        self.trainable = trainable

        if random:
            self.model = Regress_onset_offset_frame_velocity_CRNN(
                frames_per_second=pt_config.frames_per_second,
                classes_num=pt_config.classes_num,
            )
        else:
            self.model = PianoTranscription(device="cpu", checkpoint_path=None).model
        self.latent_dim = 88 * 4

    def _encode_before_post_fn(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract note logits before post-fn refinement in piano_transcription_inference."""
        note_model = self.model.note_model if hasattr(self.model, "note_model") else self.model

        required_attrs = [
            "spectrogram_extractor",
            "logmel_extractor",
            "bn0",
            "frame_model",
            "reg_onset_model",
            "reg_offset_model",
            "velocity_model",
        ]
        if all(hasattr(note_model, attr) for attr in required_attrs):
            x = note_model.spectrogram_extractor(audio)  # (b, 1, t, f)
            x = note_model.logmel_extractor(x)  # (b, 1, t, mel)
            x = x.transpose(1, 3)
            x = note_model.bn0(x)
            x = x.transpose(1, 3)

            # These are outputs before the post-fn refinement blocks
            frame_output = note_model.frame_model(x)
            reg_onset_output = note_model.reg_onset_model(x)
            reg_offset_output = note_model.reg_offset_model(x)
            velocity_output = note_model.velocity_model(x)

            return torch.cat(
                (reg_onset_output, reg_offset_output, frame_output, velocity_output),
                dim=-1,
            )

        output_dict = self.model(audio)
        return torch.cat(
            (
                output_dict["reg_onset_output"],
                output_dict["reg_offset_output"],
                output_dict["frame_output"],
                output_dict["velocity_output"],
            ),
            dim=-1,
        )

    def encode(self, audio: torch.Tensor, train_mode) -> torch.Tensor:
        r"""Extract audio latent.

        Args:
            audio: (b, c, t)

        Returns:
            latent: (b, t, d)
        """

        # Resample audio
        audio = torchaudio.functional.resample(
            waveform=audio, 
            orig_freq=self.audio_sr, 
            new_freq=self.model_sr
        )

        # To mono
        audio = torch.mean(audio, dim=1)  # shape: (b, t)

        if self.trainable and train_mode:
            self.model.train()
        else:
            self.model.eval()

        # Forward
        with torch.set_grad_enabled(self.trainable and train_mode):
            latent = self._encode_before_post_fn(audio)  # shape: (b, t, d)

        assert latent.shape[-1] == self.latent_dim, (
            f"Unexpected latent dim: {latent.shape[-1]} != {self.latent_dim}"
        )

        return latent