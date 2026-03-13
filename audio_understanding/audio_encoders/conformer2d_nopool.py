"""
model:
    name: Conformer2D
    n_fft: 2048 
    hop_length: 160
    
NO summary tokens in model architecture

"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from torchaudio.transforms import MelSpectrogram

import logging
from ..llm.rope import apply_rope, build_rope
from .fourier import Fourier


@dataclass
class Conformer2DConfig:
    sr: int = 16000
    n_fft: int = 2048
    hop_length: int = 160
    block_size: int = 1024 #* currently, 10s audio -> 1000 frames --> 250 frames after downsample, which can be handled by RoPE. If block_size is too small, the performance may drop.
    enc_layers: int = 6
    dec_layers: int = 6
    n_head: int = 16
    n_embd: int = 1024


class Conformer2D(Fourier):
    def __init__(self, sr: int, trainable: bool=True, use_decoder: bool=True):
        config = Conformer2DConfig(sr=sr)
        super(Conformer2D, self).__init__(
            n_fft=config.n_fft, 
            hop_length=config.hop_length, 
            return_complex=True, 
            normalized=True
        )

        self.downsample_factor = 4
        self.n_layer = config.enc_layers + config.dec_layers

        self.mel_extractor = MelSpectrogram(
            sample_rate=config.sr,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            f_min=0.,
            f_max=config.sr / 2,
            n_mels=256,
            power=2.0,
            normalized=True,
        )
        # Two downsampling blocks here, 4 times of downsample
        #! 
        self.conv1 = ConvBlock(in_channels=1, out_channels=32)
        self.conv2 = ConvBlock(in_channels=32, out_channels=64)

        self.pre_fc = nn.Linear(4096, 1024)

        self.encoder_blocks = nn.ModuleList(Block(config) for _ in range(config.enc_layers))
        self.decoder_blocks = nn.ModuleList(Block(config) for _ in range(config.dec_layers))

        # Build RoPE cache
        rope = build_rope(
            seq_len=config.block_size,
            head_dim=config.n_embd // config.n_head,
        )  # shape: (t, head_dim/2, 2)
        self.register_buffer(name="rope", tensor=rope)
        
        self.train_mode = trainable
        self.latent_dim = config.n_embd
        self.use_decoder = use_decoder
        
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer))
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        r"""Forward.

        b: batch_size
        c: channels_num
        l: samples_num
        t: frames_num
        d: latent_dim

        Args:
            audio: (b, c, l)

        Outputs:
            x: (b, t, d)
        """
        x = self.encode(audio, self.train_mode)
        return x

    def encode(self, audio: torch.Tensor, train_mode: bool) -> torch.Tensor:
        r"""Encode.

        b: batch_size
        c: channels_num
        l: samples_num
        t: frames_num
        d: latent_dim

        Args:
            audio: (b, c, l)

        Outputs:
            x: (b, t, d)
        """

        x = self.mel_extractor(audio)  # shape: (b, c, f, t)
        x = rearrange(x, 'b c f t -> b c t f')  # shape: (b, c, t, f)

        x, _ = self.pad_tensor(x)
        x = torch.log10(torch.clamp(x, 1e-10))

        x = self.conv1(x)  # shape: (b, c, t, f)
        x = self.conv2(x)  # shape: (b, c, t, f)

        x = rearrange(x, 'b c t f -> b t (c f)')  # shape: (b, t, d)
        x = self.pre_fc(x)  # shape: (b, t, d)

        if x.shape[1] > self.rope.shape[0]:
            logging.info("Input frames num {} is larger than RoPE cache size {}, which may cause performance drop.".format(x.shape[1], self.rope.shape[0]))
        for block in self.encoder_blocks:
            x = block(x, self.rope)  # shape: (b, t, d)


        if self.use_decoder:
            for block in self.decoder_blocks:
                x = block(x, self.rope)  # shape: (b, t, d)

        return x

    def pad_tensor(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Pad a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (b, c, t=1001, f)
        
        Outpus:
            output: E.g., (b, c, t=1004, f)
        """

        T = x.shape[2]
        pad_t = -T % self.downsample_factor
        x = F.pad(x, pad=(0, 0, 0, pad_t))

        return x, pad_t

    def unpad_tensor(self, x: torch.Tensor, pad_t: int) -> torch.Tensor:
        """Unpad a spectrum to the original shape.
        
        Args:
            x: E.g., (b, c, t=1004, f)
        
        Outpus:
            x: E.g., (b, c, t=1001, f)
        """
        x = x[:, :, 0 : -pad_t]

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=(3, 3), 
            padding=(1, 1)
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        r"""

        Args:
            x: (b, c, t, f)

        Returns:
            x: (b, c, t, f/2)
        """

        x = F.relu_(self.bn(self.conv(x)))
        x = F.avg_pool2d(x, kernel_size=(1, 2))
        
        return x 



class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.att_norm = RMSNorm(config.n_embd)
        self.att = SelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
    ) -> torch.Tensor:
        r"""

        Args:
            x: (b, t, d)
            rope: (t, head_dim/2)
            mask: (1, 1, t, t)

        Outputs:
            x: (b, t, d)
        """
        x = x + self.att(self.att_norm(x), rope)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class RMSNorm(nn.Module):
    r"""Root Mean Square Layer Normalization.

    Ref: https://github.com/meta-llama/llama/blob/main/llama/model.py
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""RMSNorm.

        Args:
            x: (b, t, d)
           
        Outputs:
            x: (b, t, d)
        """
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        output = x * torch.rsqrt(norm_x + self.eps) * self.scale
        return output


class SelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
    ) -> torch.Tensor:
        r"""Causal self attention.

        b: batch size
        t: time steps
        d: latent dim
        h: heads num

        Args:
            x: (b, t, d)
            rope: (t, head_dim/2, 2)
            mask: (1, 1, )

        Outputs:
            x: (b, t, d)
        """
        B, T, D = x.shape

        # Calculate query, key, values
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # q, k, v shapes: (b, t, d)

        k = k.view(B, T, self.n_head, D // self.n_head)
        q = q.view(B, T, self.n_head, D // self.n_head)
        v = v.view(B, T, self.n_head, D // self.n_head)
        # q, k, v shapes: (b, t, h, head_dim)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)
        # q, k shapes: (b, t, h, head_dim)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # q, k, v shapes: (b, h, t, head_dim)

        # Efficient attention using Flash Attention CUDA kernels
        x = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=None, 
            dropout_p=0.0
        )
        # shape: (b, h, t, head_dim)

        x = x.transpose(1, 2).contiguous().view(B, T, D)  # shape: (b, t, d)

        # output projection
        x = self.c_proj(x)  # shape: (b, t, d)
        
        return x


class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        # The hyper-parameters follow https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3) 

        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Causal self attention.

        Args:
            x: (b, t, d)
           
        Outputs:
            x: (b, t, d)
        """
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x