"""
Modified from: https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
"""
import torch


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