from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio
from muq import MuQ as MuQModel


class MuQ(nn.Module):

    def __init__(
        self,
        sr: float,
        trainable: bool,
        target_layer: int = -1,
        pretrained_model_name: str = "OpenMuQ/MuQ-large-msd-iter",
    ) -> None:
        r"""MuQ audio encoder.

        Notes:
            - MuQ requires 24kHz waveform input.
            - `target_layer` convention follows hidden-state indexing:
              0 means encoder input representation, N means output of N-th transformer layer,
              and -1 means the last layer.
        """

        super().__init__()

        self.audio_sr = sr
        self.model_sr = 24000
        self.trainable = trainable
        self.pretrained_model_name = pretrained_model_name

        self.model = MuQModel.from_pretrained(pretrained_model_name)
        
        #* perhaps a better way is to just run a sample first
        self.latent_dim = 1024
        self.num_hidden_layers = 13
        self.fps = 25.0

        self.target_layer = target_layer
        self._validate_target_layer()

        if not self.trainable:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def _validate_target_layer(self) -> None:
        if self.target_layer == -1:
            return
        assert 0 <= self.target_layer <= self.num_hidden_layers, (
            f"target_layer must be in [0, {self.num_hidden_layers}] or -1, "
            f"but got {self.target_layer}."
        )

    def _resolve_target_layer(self) -> int:
        if self.target_layer == -1:
            return self.num_hidden_layers
        return self.target_layer

    def encode(self, audio: torch.Tensor, train_mode: bool) -> torch.Tensor:
        r"""Extract audio latent.

        Args:
            audio: (b, c, t)

        Returns:
            latent: (b, t, d)
        """

        audio = torchaudio.functional.resample(
            waveform=audio,
            orig_freq=self.audio_sr,
            new_freq=self.model_sr,
        )
        audio = torch.mean(audio, dim=1)

        if self.trainable and train_mode:
            self.model.train()
        else:
            self.model.eval()

        target_layer = self._resolve_target_layer()

        with torch.set_grad_enabled(self.trainable and train_mode):
            if target_layer == self.num_hidden_layers:
                outputs = self.model(audio, output_hidden_states=False)
                latent = outputs.last_hidden_state
            else:
                outputs = self.model(audio, output_hidden_states=True)
                assert outputs.hidden_states is not None
                latent = outputs.hidden_states[target_layer]

        return latent

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.encode(audio=audio, train_mode=self.trainable)