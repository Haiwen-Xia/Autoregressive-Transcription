r"""
T5-like decoder with audio cross-attention.

Design goal: keep interfaces and generation APIs aligned with Llama in this repo,
while changing each block to token self-attention + audio cross-attention.
"""

from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
from torch.nn import functional as F

from audio_understanding.llm.rope import apply_rope, build_rope
from .llama import CausalSelfAttention, MLP, RMSNorm, build_causal_mask


@dataclass
class T5Config:
    block_size: int = 2048
    audio_latent_dim: None | int = None
    vocab_size: int = 32000  # Better to be divied by 64
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096


class T5(nn.Module):
    r"""T5-like model with token decoder and audio cross-attention memory."""

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.config = config
        assert config.audio_latent_dim is not None

        # Audio to embedding
        self.a2e = nn.Linear(config.audio_latent_dim, config.n_embd)

        # Token embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Transformer blocks: token self-attention + cross-attention(audio)
        self.blocks = nn.ModuleList(T5Block(config) for _ in range(config.n_layer))

        # Output layers
        self.ln_f = RMSNorm(config.n_embd)
        self.audio_head = nn.Linear(config.n_embd, config.audio_latent_dim, bias=False)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        rope = build_rope(
            seq_len=config.block_size,
            head_dim=config.n_embd // config.n_head,
        )
        self.register_buffer(name="rope", tensor=rope)

    def forward(
        self,
        seqs: list[torch.Tensor],
        seq_types: list[str],
        mask: None | torch.Tensor = None,
    ) -> list[torch.Tensor]:
        r"""Forward pass.

        Args:
            seqs: list of audio latents and token IDs
            seq_types: per-seq type tag ("audio" | "id")
            mask: optional causal mask for token self-attention

        Returns:
            output_seqs: list aligned with input seqs. "id" entries are logits,
                "audio" entries are projected audio-latent predictions.
        """
        embedded_seqs = []
        audio_latent_list = []
        token_latent_list = []
        id_seq_lens = []

        for seq, seq_type in zip(seqs, seq_types):
            if seq_type == "audio":
                x = self.a2e(seq)  # (b, t_audio, d)
                audio_latent_list.append(x)
                embedded_seqs.append(x)
            elif seq_type == "id":
                x = self.wte(seq)  # (b, t_text, d)
                token_latent_list.append(x)
                id_seq_lens.append(seq.shape[1])
                embedded_seqs.append(x)
            else:
                raise ValueError(seq_type)

        assert len(audio_latent_list) > 0, "T5 requires at least one audio sequence"
        assert len(token_latent_list) > 0, "T5 requires at least one id sequence"

        audio_latent = torch.cat(audio_latent_list, dim=1)  # (b, t_audio, d)
        token_latent = torch.cat(token_latent_list, dim=1)  # (b, t_text, d)

        device = token_latent.device
        token_len = token_latent.shape[1]
        assert token_len <= self.config.block_size, f"Can not forward sequence of {token_len} > {self.config.block_size}"

        if mask is None:
            mask = build_causal_mask(seq_len=token_len).to(device)

        for block in self.blocks:
            token_latent = block(token_latent, audio_latent, self.rope, mask)

        token_latent = self.ln_f(token_latent)  # (b, t_text, d)
        token_logits = self.lm_head(token_latent)  # (b, t_text, v)

        # Split token logits back to each id sequence and restore original seq order.
        split_token_logits = []
        start_idx = 0
        for seq_len in id_seq_lens:
            split_token_logits.append(token_logits[:, start_idx : start_idx + seq_len, :])
            start_idx += seq_len

        output_seqs = []
        id_idx = 0
        audio_idx = 0
        for seq_type in seq_types:
            if seq_type == "audio":
                output_seqs.append(self.audio_head(audio_latent_list[audio_idx]))
                audio_idx += 1
            elif seq_type == "id":
                output_seqs.append(split_token_logits[id_idx])
                id_idx += 1
            else:
                raise ValueError(seq_type)

        return output_seqs

    @torch.no_grad()
    def generate(
        self,
        seqs: list[torch.Tensor],
        seq_types: list[str],
        max_new_ids: int,
        temperature: float = 1.0,
        top_k: None | int = None,
    ):
        r"""Next ID sampling with auto-regression."""
        for _t in range(max_new_ids):
            outputs = self(seqs=seqs, seq_types=seq_types)

            logits = outputs[-1]
            logits = logits[:, -1, :] / temperature  # (b, v)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (b, 1)

            seqs[-1] = torch.cat((seqs[-1], next_id), dim=1)

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
        r"""Constrained auto-regressive generation."""
        for _t in range(max_new_ids):
            outputs = self(seqs=seqs, seq_types=seq_types)

            logits = outputs[-1][:, -1, :] / temperature  # (b, v)

            if top_k is not None:
                _, raw_topk_ids = torch.topk(logits, min(top_k, logits.size(-1)))
                constraint.count_violations(raw_topk_ids[0])

            allowed_mask = constraint.get_allowed_mask()
            logits[:, ~allowed_mask] = -float("Inf")

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            should_continue = constraint.update(next_id[0, 0].item())
            if not should_continue:
                break

            seqs[-1] = torch.cat((seqs[-1], next_id), dim=1)

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
        r"""Constrained auto-regressive generation for batched inputs."""
        batch_size = seqs[-1].shape[0]
        assert len(constraints) == batch_size

        active = [True] * batch_size

        for _t in range(max_new_ids):
            outputs = self(seqs=seqs, seq_types=seq_types)
            logits = outputs[-1][:, -1, :] / temperature  # (b, v)

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
            next_id = torch.multinomial(probs, num_samples=1)

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


class T5Block(nn.Module):
    r"""Token self-attention + audio cross-attention + MLP."""

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.self_att_norm = RMSNorm(config.n_embd)
        self.self_att = CausalSelfAttention(cast(Any, config))

        self.cross_q_norm = RMSNorm(config.n_embd)
        self.cross_kv_norm = RMSNorm(config.n_embd)
        self.cross_att = CrossAttention(config)

        self.ffn_norm = RMSNorm(config.n_embd)
        self.mlp = MLP(cast(Any, config))

    def forward(
        self,
        token_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        rope: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        token_latent = token_latent + self.self_att(self.self_att_norm(token_latent), rope, mask)
        token_latent = token_latent + self.cross_att(
            q_x=self.cross_q_norm(token_latent),
            kv_x=self.cross_kv_norm(audio_latent),
            rope=rope,
        )
        token_latent = token_latent + self.mlp(self.ffn_norm(token_latent))
        return token_latent


class CrossAttention(nn.Module):
    r"""Cross-attention from token queries to audio keys/values."""

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.kv_proj = nn.Linear(config.n_embd, 2 * config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        rope: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Args:
            q_x: token latent, shape (b, t_text, d)
            kv_x: audio latent, shape (b, t_audio, d)
            rope: rotary cache, shape (t, head_dim/2, 2)
        """
        B, Tq, D = q_x.shape
        Tk = kv_x.shape[1]

        q = self.q_proj(q_x)
        k, v = self.kv_proj(kv_x).split(self.n_embd, dim=2)

        q = q.view(B, Tq, self.n_head, D // self.n_head)
        k = k.view(B, Tk, self.n_head, D // self.n_head)
        v = v.view(B, Tk, self.n_head, D // self.n_head)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=0.0,
        )

        x = x.transpose(1, 2).contiguous().view(B, Tq, D)
        x = self.out_proj(x)
        return x