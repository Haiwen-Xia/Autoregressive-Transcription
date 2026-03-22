r"""
Modified from https://github.com/qiuqiangkong/mini_llm/blob/main/models/llama.py
"""
import math
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.nn import functional as F
from audio_understanding.llm.rope import RotaryEmbedding, RotaryInput
from audio_understanding.llm.time_rope import (
    DEFAULT_EVENT_ATTRIBUTE_PREFIXES,
    build_position_time_inputs,
    infer_current_event_time_from_tokens,
    update_decode_state,
)


#* is one singular rope for all modalities good?
@dataclass
class LlamaConfig:
    block_size: int = 2048
    audio_latent_dim: None | int = None
    vocab_size: int = 32000  # Better to be divied by 64
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096
    audio_use_absolute_pe: bool = False
    rope_scope: str = "all"
    rope_mode: str = "ordinary"
    time_rope_mix_weight: float = 0.5
    time_rope_use_linear: bool = False
    time_rope_audio_fps: float = 100.0
    time_rope_token_fps: float = 100.0
    time_rope_alpha: None | float = None
    time_rope_event_attribute_prefixes: tuple[str, ...] = DEFAULT_EVENT_ATTRIBUTE_PREFIXES
    id_to_token: None | Callable[[list[int]], list[str]] = None


# Default Llama configurations
llama_configs = {
    "7B": dict(n_layer=32, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192),
}


class Llama(nn.Module):
    r"""Llama model."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()

        self.config = config
        assert self.config.rope_scope in ["all", "text_only"]
        assert self.config.rope_mode in [
            "ordinary",
            "1d",
            "1d_linear",
            "2d",
            "time_aware",
            "time_aware_2d",
        ]
        assert 0.0 <= self.config.time_rope_mix_weight <= 1.0
        if self.config.time_rope_alpha is None:
            self.config.time_rope_alpha = float(self.config.time_rope_audio_fps)
        assert self.config.time_rope_alpha > 0

        # Audio to embedding
        self.a2e = nn.Linear(config.audio_latent_dim, config.n_embd)

        # Word to embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))

        # Output layers
        self.ln_f = RMSNorm(config.n_embd)
        self.audio_head = nn.Linear(config.n_embd, config.audio_latent_dim, bias=False) #* seems uncessary
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Build RoPE module
        self.rope = RotaryEmbedding(
            seq_len=config.block_size,
            head_dim=config.n_embd // config.n_head,
            mode=config.rope_mode,
            mix_weight=config.time_rope_mix_weight,
            use_linear=config.time_rope_use_linear,
        )

        if config.audio_use_absolute_pe:
            abs_pe = build_sincos_absolute_pe(
                seq_len=config.block_size,
                dim=config.n_embd,
            )  # shape: (t, d)
            self.register_buffer(name="audio_abs_pe", tensor=abs_pe)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

    def forward(
        self, 
        seqs: list[torch.Tensor],
        # input_seq_types: list[str],
        # output_seq_types: list[str],
        seq_types: list[str],
        mask: None | torch.Tensor = None,
    ) -> torch.Tensor:
        r"""Next ID prediction with Llama.

        b: batch_size
        t: time_steps
        d: hidden_size
        v: vocab_size

        Args:
            IDs: (b, t)
            mask: None | (1, 1, t, t)

        Outputs:
            logits: (b, t, v)
        """
        
        # Transform and concatenate audio embeddings and text IDs into latent
        x = self.seqs_to_latent(seqs=seqs, seq_types=seq_types)  # shape: (b, t, d)

        device = x.device
        B, T, D = x.shape

        assert T <= self.config.block_size, "Can not forward sequence of {T} > {self.config.block_size}"

        if mask is None:
            mask = build_causal_mask(seq_len=T).to(device)

        rope_apply_mask = self.build_rope_apply_mask(seqs=seqs, seq_types=seq_types, seq_len=T, device=device)

        time_coords = None
        if self.config.rope_mode != "ordinary":
            alpha = self.config.time_rope_alpha
            assert alpha is not None
            _, time_coords = build_position_time_inputs(
                seqs=seqs,
                seq_types=seq_types,
                id_to_token=self.config.id_to_token,
                audio_fps=self.config.time_rope_audio_fps,
                token_fps=self.config.time_rope_token_fps,
                alpha=float(alpha),
                event_attribute_prefixes=self.config.time_rope_event_attribute_prefixes,
                strict_event_time=True,
            )

        rope_input = RotaryInput(
            rope_apply_mask=rope_apply_mask,
            time_coords=time_coords,
        )

        # Transformer
        for block in self.blocks:
            x = block(
                x=x,
                rope=self.rope,
                rope_input=rope_input,
                mask=mask,
            )
        # x: (b, t, d)

        # Output layers
        x = self.ln_f(x)  # shape: (b, t, d)

        # Split and transform latent into audio latents and text IDs.
        seq_lens = [seq.shape[1] for seq in seqs]
        output_seqs = self.latent_to_seqs(latent=x, seq_lens=seq_lens, seq_types=seq_types)

        return output_seqs

    def seqs_to_latent(
        self, 
        seqs: list[torch.Tensor], 
        seq_types: list[str]
    ) -> torch.Tensor:
        r"""Transform audio latents and IDs into latents with same dimensinos 
        and concatenate them."""
        
        latent = []

        for seq, seq_type in zip(seqs, seq_types):

            if seq_type == "audio":
                x = self.a2e(seq)  # shape: (b, t_audio, d)
                if self.config.audio_use_absolute_pe:
                    t_audio = x.shape[1]
                    x = x + self.audio_abs_pe[:t_audio].to(x.dtype).unsqueeze(0)

            elif seq_type == "id":
                x = self.wte(seq)  # shape: (b, t_text, d)

            else:
                raise ValueError(seq_type)

            latent.append(x)

        latent = torch.cat(latent, dim=1)  # shape: (b, t, d)

        return latent

    def build_rope_apply_mask(
        self,
        seqs: list[torch.Tensor],
        seq_types: list[str],
        seq_len: int,
        device: torch.device,
    ) -> None | torch.Tensor:
        if self.config.rope_scope == "all":
            return None

        assert self.config.rope_scope == "text_only"

        mask_parts = []
        for seq, seq_type in zip(seqs, seq_types):
            t = seq.shape[1]
            if seq_type == "audio":
                mask_parts.append(torch.zeros(t, dtype=torch.bool, device=device))
            else:
                assert seq_type == "id"
                mask_parts.append(torch.ones(t, dtype=torch.bool, device=device))

        rope_apply_mask = torch.cat(mask_parts, dim=0)
        assert rope_apply_mask.shape[0] == seq_len
        return rope_apply_mask

    def _decode_ids_to_tokens(self, ids: list[int]) -> None | list[str]:
        if self.config.id_to_token is None:
            return None
        return self.config.id_to_token(ids)

    def update_decode_time_state(
        self,
        new_token: str,
        current_event_time: None | float,
    ) -> None | float:
        return update_decode_state(
            new_token=new_token,
            current_event_time=current_event_time,
            token_fps=self.config.time_rope_token_fps,
            event_attribute_prefixes=self.config.time_rope_event_attribute_prefixes,
        )

    def latent_to_seqs(
        self, 
        latent: torch.Tensor, 
        seq_lens: list[int], 
        seq_types: list[str]
    ) -> list[torch.Tensor]:
        r"""Split latent into sequences and transform them into audio latents 
        and IDs.
        """

        seqs = []
        start_idx = 0

        for seq_len, seq_type in zip(seq_lens, seq_types):

            x = latent[:, start_idx : start_idx + seq_len, :]
            start_idx += seq_len

            if seq_type == "audio":
                x = self.audio_head(x)

            elif seq_type == "id":
                x = self.lm_head(x)  # shape: (b, t_text, d)

            else:
                raise ValueError(seq_type)

            seqs.append(x)

        return seqs

    @torch.no_grad()
    def generate(
        self, 
        seqs: torch.LongTensor, 
        seq_types: list[str],
        max_new_ids: int, 
        temperature: float = 1.0, 
        top_k: None | int = None
    ):
        r"""Next ID sampling with auto-regression. Make sure to use model.eval()

        b: batch_size
        t: time_steps
        v: vocab_size

        Args:
            ids: (b, 1)
            max_new_ids: int
            temperature: float
            top_k: None | int

        Returns:
            new_ids: (b, t), sampled IDs
        """
        # input_len = ids.shape[1]

        decode_state: None | float = None
        if self.config.rope_mode != "ordinary" and self.config.id_to_token is not None:
            seed_tokens = self.config.id_to_token(seqs[-1][0].detach().cpu().tolist())
            decode_state = infer_current_event_time_from_tokens(
                seed_tokens,
                token_fps=self.config.time_rope_token_fps,
                event_attribute_prefixes=self.config.time_rope_event_attribute_prefixes,
            )

        for t in range(max_new_ids):
            # Forward
            outputs = self(seqs=seqs, seq_types=seq_types)

            # Text logits
            logits = outputs[-1]

            # Take the final step logits
            logits = logits[:, -1, :] / temperature  # shape: (b, v)

            # Crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)  # shape: (b, v)

            # Sample the next token
            next_id = torch.multinomial(probs, num_samples=1)  # shape: (b, 1)

            # Append the sampled token to the last seq
            seqs[-1] = torch.cat((seqs[-1], next_id), dim=1)  # shape: (b, t)

            if self.config.rope_mode != "ordinary" and self.config.id_to_token is not None:
                next_token = self.config.id_to_token([int(next_id[0, 0].item())])[0]
                decode_state = self.update_decode_time_state(next_token, decode_state)

        return seqs

    @torch.no_grad()
    def generate_constrained(
        self,
        seqs: list[torch.Tensor],
        seq_types: list[str],
        max_new_ids: int,
        constraint,
        temperature: float = 1.0,
        top_k: None | int = None,
    ):
        r"""Constrained auto-regressive generation.

        At each step the *constraint* object restricts which token IDs are
        allowed and counts how many of the original top-k candidates would
        have violated the grammar.

        Args:
            seqs: input sequences (audio latent, question ids, answering ids)
            seq_types: type tag per seq ("audio" / "id")
            max_new_ids: maximum tokens to generate
            constraint: object exposing get_allowed_mask(), count_violations(), update()
            temperature: softmax temperature
            top_k: top-k sampling width (applied *after* constraint masking)

        Returns:
            seqs: updated sequences with generated tokens appended to seqs[-1]
        """
        decode_state: None | float = None
        if self.config.rope_mode != "ordinary" and self.config.id_to_token is not None:
            seed_tokens = self.config.id_to_token(seqs[-1][0].detach().cpu().tolist())
            decode_state = infer_current_event_time_from_tokens(
                seed_tokens,
                token_fps=self.config.time_rope_token_fps,
                event_attribute_prefixes=self.config.time_rope_event_attribute_prefixes,
            )

        for _t in range(max_new_ids):
            outputs = self(seqs=seqs, seq_types=seq_types)

            # Text logits at the last time step
            logits = outputs[-1][:, -1, :] / temperature  # shape: (b, v)

            # --- violation counting on raw top-k (before constraint) ---
            if top_k is not None:
                _, raw_topk_ids = torch.topk(logits, min(top_k, logits.size(-1)))
                constraint.count_violations(raw_topk_ids[0])  # batch=0

            # --- apply constraint mask ---
            allowed_mask = constraint.get_allowed_mask()  # (v,)  bool
            logits[:, ~allowed_mask] = -float('Inf')

            # --- top-k on constrained logits ---
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)  # shape: (b, v)
            next_id = torch.multinomial(probs, num_samples=1)  # shape: (b, 1)

            if self.config.rope_mode != "ordinary" and self.config.id_to_token is not None:
                next_token = self.config.id_to_token([int(next_id[0, 0].item())])[0]
                decode_state = self.update_decode_time_state(next_token, decode_state)

            # --- state transition; stop on SEP ---
            should_continue = constraint.update(next_id[0, 0].item())
            if not should_continue:
                break

            seqs[-1] = torch.cat((seqs[-1], next_id), dim=1)  # shape: (b, t)

        return seqs

    @torch.no_grad()
    def generate_constrained_batch(
        self,
        seqs: list[torch.Tensor],
        seq_types: list[str],
        max_new_ids: int,
        constraints: list,
        sep_token_id: int,
        temperature: float = 1.0,
        top_k: None | int = None,
    ):
        r"""Constrained auto-regressive generation for batched inputs.

        Each sample uses an independent constraint state machine.

        Args:
            seqs: input sequences (audio latent, question ids, answering ids)
            seq_types: type tag per seq ("audio" / "id")
            max_new_ids: maximum tokens to generate
            constraints: list of constraint objects, one per batch item
            sep_token_id: tokenizer [SEP] token id
            temperature: softmax temperature
            top_k: top-k sampling width (applied after constraint masking)

        Returns:
            seqs: updated sequences with generated tokens appended to seqs[-1]
        """
        batch_size = seqs[-1].shape[0]
        assert len(constraints) == batch_size

        active = [True] * batch_size
        decode_states: None | list[None | float] = None
        if self.config.rope_mode != "ordinary" and self.config.id_to_token is not None:
            decode_states = []
            for b in range(batch_size):
                seed_tokens = self.config.id_to_token(seqs[-1][b].detach().cpu().tolist())
                decode_state = infer_current_event_time_from_tokens(
                    seed_tokens,
                    token_fps=self.config.time_rope_token_fps,
                    event_attribute_prefixes=self.config.time_rope_event_attribute_prefixes,
                )
                decode_states.append(decode_state)

        for _t in range(max_new_ids):
            outputs = self(seqs=seqs, seq_types=seq_types)
            logits = outputs[-1][:, -1, :] / temperature  # shape: (b, v)

            for b in range(batch_size):
                if not active[b]:
                    logits[b, :] = -float("Inf")
                    logits[b, sep_token_id] = 0.0
                    continue

                if top_k is not None:
                    _, raw_topk_ids = torch.topk(logits[b], min(top_k, logits.shape[-1]))
                    constraints[b].count_violations(raw_topk_ids)

                allowed_mask = constraints[b].get_allowed_mask()
                logits[b, ~allowed_mask] = -float("Inf")

                if top_k is not None:
                    kth = torch.topk(logits[b], min(top_k, logits.shape[-1])).values[-1]
                    logits[b, logits[b] < kth] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (b, 1)

            if self.config.rope_mode != "ordinary" and self.config.id_to_token is not None:
                assert decode_states is not None
                for b in range(batch_size):
                    if not active[b]:
                        continue
                    next_token = self.config.id_to_token([int(next_id[b, 0].item())])[0]
                    decode_states[b] = self.update_decode_time_state(next_token, decode_states[b])

            seqs[-1] = torch.cat((seqs[-1], next_id), dim=1)

            for b in range(batch_size):
                if not active[b]:
                    continue
                should_continue = constraints[b].update(int(next_id[b, 0].item()))
                if not should_continue:
                    active[b] = False

            if not any(active):
                break

        return seqs


class Block(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.att_norm = RMSNorm(config.n_embd)
        self.att = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        rope_input: RotaryInput,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        r"""

        Args:
            x: (b, t, d)
            rope: (t, head_dim/2)
            mask: (1, 1, t, t)

        Outputs:
            x: (b, t, d)
        """
        x = x + self.att(
            x=self.att_norm(x),
            rope=rope,
            rope_input=rope_input,
            mask=mask,
        )
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


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
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
        rope: RotaryEmbedding,
        rope_input: RotaryInput,
        mask: torch.Tensor,
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

        q = rope(x=q, rope_input=rope_input)
        k = rope(x=k, rope_input=rope_input)
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
            attn_mask=mask, 
            dropout_p=0.0
        )
        # shape: (b, h, t, head_dim)

        x = x.transpose(1, 2).contiguous().view(B, T, D)  # shape: (b, t, d)

        # output projection
        x = self.c_proj(x)  # shape: (b, t, d)
        
        return x


class MLP(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
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


def build_causal_mask(seq_len: int) -> torch.Tensor:
    r"""Build causal mask."""
    ones = torch.ones((seq_len, seq_len), dtype=torch.bool)  # shape: (t, t)
    mask = torch.tril(ones)[None, None, :, :]  # shape: (1, 1, t, t)
    return mask


def build_sincos_absolute_pe(seq_len: int, dim: int, base: int = 10000) -> torch.Tensor:
    r"""Build sin-cos absolute positional embedding.

    Outputs:
        abs_pe: (t, d)
    """
    assert dim % 2 == 0
    theta = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
    seq_idx = torch.arange(seq_len)
    idx_theta = torch.outer(seq_idx, theta).float()
    abs_pe = torch.zeros((seq_len, dim), dtype=idx_theta.dtype)
    abs_pe[:, 0::2] = torch.sin(idx_theta)
    abs_pe[:, 1::2] = torch.cos(idx_theta)
    return abs_pe