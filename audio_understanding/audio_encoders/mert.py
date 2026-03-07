from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio
from transformers import AutoModel, Wav2Vec2FeatureExtractor


class MERT(nn.Module):

    def __init__(
        self,
        sr: float,
        trainable: bool,
        target_layer: int = -1,
        pretrained_model_name: str = "m-a-p/MERT-v1-330M",
    ) -> None:
        r"""MERT wav2vec-style audio encoder.

        Notes:
            - Default checkpoint is MERT-v1-95M.
            - `target_layer` follows hidden-state indexing convention:
              0 means encoder input representation, N means output of N-th transformer layer,
              and -1 means the last layer.
        """

        super().__init__()

        self.audio_sr = sr
        self.trainable = trainable
        self.pretrained_model_name = pretrained_model_name

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
        )

        self.model_sr = self.feature_extractor.sampling_rate
        self.do_normalize = self.feature_extractor.do_normalize
        self.latent_dim = self.model.config.hidden_size

        self.target_layer = target_layer
        self.num_hidden_layers = len(self.model.encoder.layers)
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

    def _normalize_waveform(self, audio: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(audio, dim=-1, keepdim=True)
        var = torch.var(audio, dim=-1, keepdim=True, unbiased=False)
        return (audio - mean) / torch.sqrt(var + 1e-7)

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

        if self.do_normalize:
            audio = self._normalize_waveform(audio)

        attention_mask = torch.ones(
            size=audio.shape,
            dtype=torch.long,
            device=audio.device,
        )

        if self.trainable and train_mode:
            self.model.train()
        else:
            self.model.eval()

        target_layer = self._resolve_target_layer()

        with torch.set_grad_enabled(self.trainable and train_mode):
            if target_layer == self.num_hidden_layers:
                outputs = self.model(
                    input_values=audio,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    return_dict=True,
                )
                latent = outputs.last_hidden_state
            else:
                outputs = self.model(
                    input_values=audio,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                assert outputs.hidden_states is not None
                latent = outputs.hidden_states[target_layer]

        return latent

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.encode(audio=audio, train_mode=self.trainable)