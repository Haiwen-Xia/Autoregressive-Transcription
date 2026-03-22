"""
Modified from: https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
"""
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RotaryInput:
    rope_apply_mask: None | torch.Tensor = None
    time_coords: None | torch.Tensor = None


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        seq_len: int,
        head_dim: int,
        mode: str = "ordinary",
        mix_weight: float = 0.5,
        use_linear: bool = False,
        base: int = 10000,
    ) -> None:
        super().__init__()
        mode_alias = {
            "time_aware": "1d",
            "time_aware_2d": "2d",
        }
        mode = mode_alias.get(mode, mode)
        if mode == "1d" and use_linear:
            mode = "1d_linear"

        assert mode in ["ordinary", "1d", "1d_linear", "2d"]
        assert 0.0 <= mix_weight <= 1.0

        self.mode = mode
        self.mix_weight = mix_weight
        self.base = base
        self.head_dim = head_dim

        rope = build_rope(seq_len=seq_len, head_dim=head_dim, base=base)
        self.register_buffer(name="rope_cache", tensor=rope)

        if mode == "2d":
            assert head_dim % 4 == 0

    def forward(
        self,
        x: torch.Tensor,
        rope_input: None | RotaryInput = None,
    ) -> torch.Tensor:
        rope_apply_mask = None
        time_coords = None
        if rope_input is not None:
            rope_apply_mask = rope_input.rope_apply_mask
            time_coords = rope_input.time_coords

        if self.mode == "ordinary":
            return apply_rope(
                x=x,
                rope_cache=self.rope_cache,
                rope_apply_mask=rope_apply_mask,
            )

        assert time_coords is not None

        if self.mode == "1d":
            x_pos = apply_rope(
                x=x,
                rope_cache=self.rope_cache,
                rope_apply_mask=rope_apply_mask,
            )
            x_time = apply_rope_with_coords(x=x, coords=time_coords, base=self.base)
            x_mix = self.mix_weight * x_pos + (1.0 - self.mix_weight) * x_time

            if rope_apply_mask is None:
                return x_mix.type_as(x)

            t = x.shape[1]
            mask = rope_apply_mask[:t].view(1, t, 1, 1)
            return torch.where(mask, x_mix, x_pos).type_as(x)

        if self.mode == "1d_linear":
            t = x.shape[1]
            pos_coords = torch.arange(t, device=time_coords.device, dtype=torch.float32).unsqueeze(0)
            mixed_coords = (1.0 - self.mix_weight) * time_coords[:, :t] + self.mix_weight * pos_coords
            x_rot = apply_rope_with_coords(x=x, coords=mixed_coords, base=self.base)
            return _apply_rotation_mask(x_orig=x, x_rot=x_rot, rope_apply_mask=rope_apply_mask)

        assert self.mode == "2d"
        x_rot = apply_rope_2d(
            x=x,
            time_coords=time_coords,
            base=self.base,
        )
        return _apply_rotation_mask(x_orig=x, x_rot=x_rot, rope_apply_mask=rope_apply_mask)


def _apply_rotation_mask(
    x_orig: torch.Tensor,
    x_rot: torch.Tensor,
    rope_apply_mask: None | torch.Tensor,
) -> torch.Tensor:
    if rope_apply_mask is None:
        return x_rot

    t = x_orig.shape[1]
    mask = rope_apply_mask[:t].view(1, t, 1, 1)
    output = torch.where(mask, x_rot.float(), x_orig.float())
    return output.type_as(x_orig)


def build_rope(
    seq_len: int, head_dim: int, base: int = 10000
) -> torch.Tensor:
    r"""Rotary Position Embedding.
    Modified from: https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py

    Args:
        seq_len: int, e.g., 1024
        head_dim: head dim, e.g., 768/24
        base: int

    Outputs:
        cache: (t, head_dim/2, 2)
    """
    
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))

    seq_idx = torch.arange(seq_len)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float()

    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    return cache


def apply_rope(
    x: torch.Tensor,
    rope_cache: torch.Tensor,
    rope_apply_mask: None | torch.Tensor = None,
) -> torch.Tensor:
    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # cast because the reference does
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)

    if rope_apply_mask is None:
        return x_out2.type_as(x)

    # rope_apply_mask: (t,), True means applying RoPE at that position.
    mask = rope_apply_mask[:T].view(1, T, 1, 1)
    output = torch.where(mask, x_out2, x.float())
    return output.type_as(x)


def apply_rope_with_coords(
    x: torch.Tensor,
    coords: torch.Tensor,
    base: int = 10000,
) -> torch.Tensor:
    r"""Apply RoPE rotation using real-valued coordinates.

    Args:
        x: (b, t, h, head_dim)
        coords: (b, t) or (t,)
        base: RoPE base
    """
    B, T, _, head_dim = x.shape

    if coords.ndim == 1:
        coords = coords.unsqueeze(0).expand(B, -1)

    assert coords.shape[0] == B
    assert coords.shape[1] >= T

    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim))
    angle = coords[:, :T].float().unsqueeze(-1) * theta.view(1, 1, -1)
    cos = torch.cos(angle).view(B, T, 1, -1, 1)
    sin = torch.sin(angle).view(B, T, 1, -1, 1)

    x_pair = x.float().reshape(B, T, x.shape[2], -1, 2)
    x_rot = torch.stack(
        [
            x_pair[..., 0] * cos[..., 0] - x_pair[..., 1] * sin[..., 0],
            x_pair[..., 1] * cos[..., 0] + x_pair[..., 0] * sin[..., 0],
        ],
        dim=-1,
    )
    return x_rot.flatten(3).type_as(x)


def apply_rope_2d(
    x: torch.Tensor,
    time_coords: torch.Tensor,
    base: int = 10000,
) -> torch.Tensor:
    r"""Apply 2D RoPE with per-token (pos_coord, time_coord).

    The head dimension is grouped by 4 as [x0, x1, y0, y1].
    For each 4D block, apply block-diagonal rotations diag(R_pos, R_time).
    """
    b, t, h, head_dim = x.shape
    assert head_dim % 4 == 0

    if time_coords.ndim == 1:
        time_coords = time_coords.unsqueeze(0).expand(b, -1)

    assert time_coords.shape[0] == b
    assert time_coords.shape[1] >= t

    quarter_dim = head_dim // 4
    theta = 1.0 / (
        base ** (torch.arange(0, quarter_dim, device=x.device, dtype=torch.float32) / quarter_dim)
    )

    pos_coords = torch.arange(t, device=x.device, dtype=torch.float32).unsqueeze(0).expand(b, -1)
    angle_pos = pos_coords[:, :t].unsqueeze(-1) * theta.view(1, 1, -1)
    angle_time = time_coords[:, :t].float().unsqueeze(-1) * theta.view(1, 1, -1)

    cos_pos = torch.cos(angle_pos).view(b, t, 1, quarter_dim)
    sin_pos = torch.sin(angle_pos).view(b, t, 1, quarter_dim)
    cos_time = torch.cos(angle_time).view(b, t, 1, quarter_dim)
    sin_time = torch.sin(angle_time).view(b, t, 1, quarter_dim)

    x_4 = x.float().reshape(b, t, h, quarter_dim, 4)

    x0 = x_4[..., 0]
    x1 = x_4[..., 1]
    y0 = x_4[..., 2]
    y1 = x_4[..., 3]

    xr0 = x0 * cos_pos - x1 * sin_pos
    xr1 = x1 * cos_pos + x0 * sin_pos
    yr0 = y0 * cos_time - y1 * sin_time
    yr1 = y1 * cos_time + y0 * sin_time

    x_rot = torch.stack([xr0, xr1, yr0, yr1], dim=-1)
    return x_rot.reshape(b, t, h, head_dim).type_as(x)


def apply_mixed_rope(
    x: torch.Tensor,
    rope_cache: torch.Tensor,
    rope_apply_mask: None | torch.Tensor,
    time_coords: None | torch.Tensor,
    use_time_rope: None | torch.Tensor,
    mix_weight: float,
    angle_interpolate: bool = False,
) -> torch.Tensor:
    r"""Apply mixed RoPE with either linear angle interpolation or output mixing.

    Args:
        x: (b, t, h, head_dim)
        rope_cache: (t, head_dim/2, 2)
        rope_apply_mask: (t,) bool, which positions apply standard RoPE
        time_coords: (b, t)
        use_time_rope: (b, t) bool, which tokens use time-aware RoPE
        mix_weight: float in [0, 1]
        angle_interpolate: if True, linearly interpolate position and time coordinates
                          before applying RoPE; if False, mix output vectors

    Returns:
        x_out: (b, t, h, head_dim)
    """
    if angle_interpolate:
        assert time_coords is not None
        assert use_time_rope is not None

        # Linear interpolation of coordinates (angles) before applying RoPE
        T = x.shape[1]
        pos_coords = torch.arange(T, device=time_coords.device, dtype=torch.float32)
        
        # Interpolate: mixed_coords = (1-mix_weight) * time_coords + mix_weight * pos_coords
        mixed_coords = (1.0 - mix_weight) * time_coords + mix_weight * pos_coords.unsqueeze(0)
        
        # For non-temporal tokens, use pure position coordinates
        mixed_coords = torch.where(use_time_rope, mixed_coords, pos_coords.unsqueeze(0))
        
        x_out = apply_rope_with_coords(x=x, coords=mixed_coords)
        return x_out

    # Original output mixing approach
    x_pos = apply_rope(x=x, rope_cache=rope_cache, rope_apply_mask=rope_apply_mask)

    if time_coords is None or use_time_rope is None:
        return x_pos

    x_time = apply_rope_with_coords(x=x, coords=time_coords)

    mask = use_time_rope[:, : x.shape[1]]
    if rope_apply_mask is not None:
        mask = mask & rope_apply_mask[: x.shape[1]].view(1, -1)

    mix_mask = mask.view(x.shape[0], x.shape[1], 1, 1)
    x_mix = mix_weight * x_pos + (1.0 - mix_weight) * x_time
    return torch.where(mix_mask, x_mix, x_pos)