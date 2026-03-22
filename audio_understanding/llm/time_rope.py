from __future__ import annotations

from typing import Callable

import torch


DEFAULT_EVENT_ATTRIBUTE_PREFIXES = (
    "onset",
    "offset",
    "duration",
    "pitch=",
    "drum_pitch=",
    "velocity=",
    "program=",
)


def token_type(
    token: str,
    event_attribute_prefixes: tuple[str, ...] = DEFAULT_EVENT_ATTRIBUTE_PREFIXES,
) -> str:
    if token.startswith("time=") or token.startswith("time_index="):
        return "timestamp"

    for prefix in event_attribute_prefixes:
        if token == prefix or token.startswith(prefix):
            return "event_attribute"

    return "non_temporal"


def parse_time_token(token: str, token_fps: float = 100.0) -> float:
    assert token_fps > 0
    if token.startswith("time="):
        return float(token.split("=", 1)[1]) / float(token_fps)

    if token.startswith("time_index="):
        return float(token.split("=", 1)[1]) / float(token_fps)

    raise ValueError(f"Not a timestamp token: {token}")


def update_decode_state(
    new_token: str,
    current_event_time: None | float,
    token_fps: float = 100.0,
    event_attribute_prefixes: tuple[str, ...] = DEFAULT_EVENT_ATTRIBUTE_PREFIXES,
) -> None | float:
    new_token_type = token_type(new_token, event_attribute_prefixes=event_attribute_prefixes)
    if new_token_type == "timestamp":
        return parse_time_token(new_token, token_fps=token_fps)

    if new_token_type == "event_attribute":
        assert current_event_time is not None, "Event attribute token seen before any timestamp token"

    return current_event_time


def assign_time_coords(
    item_types: list[str],
    items: list[None | str],
    audio_times: list[None | float],
    audio_fps: float,
    token_fps: float,
    alpha: float,
    strict_event_time: bool = True,
) -> tuple[list[int], list[float], list[bool]]:
    pos_ids: list[int] = []
    time_coords: list[float] = []
    use_time_rope: list[bool] = []

    current_event_time: None | float = None

    for i, item_type in enumerate(item_types):
        pos_ids.append(i)

        if item_type == "audio_latent":
            t = audio_times[i]
            assert t is not None
            time_coords.append(float(alpha) * float(t))
            use_time_rope.append(True)

        elif item_type == "timestamp":
            tok = items[i]
            assert tok is not None
            current_event_time = parse_time_token(tok, token_fps=token_fps)
            time_coords.append(float(alpha) * float(current_event_time))
            use_time_rope.append(True)

        elif item_type == "event_attribute":
            if strict_event_time:
                assert current_event_time is not None, "Event attribute token seen before any timestamp token"

            if current_event_time is None:
                time_coords.append(0.0)
                use_time_rope.append(False)
            else:
                time_coords.append(float(alpha) * float(current_event_time))
                use_time_rope.append(True)

        elif item_type == "non_temporal":
            time_coords.append(0.0)
            use_time_rope.append(False)

        else:
            raise ValueError(item_type)

    return pos_ids, time_coords, use_time_rope


def build_position_time_inputs(
    seqs: list[torch.Tensor],
    seq_types: list[str],
    id_to_token: None | Callable[[list[int]], list[str]],
    audio_fps: float,
    token_fps: float,
    alpha: float,
    event_attribute_prefixes: tuple[str, ...] = DEFAULT_EVENT_ATTRIBUTE_PREFIXES,
    strict_event_time: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(seqs) == len(seq_types)
    assert audio_fps > 0
    assert token_fps > 0
    assert alpha > 0

    batch_size = seqs[0].shape[0]
    seq_len_total = sum(seq.shape[1] for seq in seqs)

    pos_ids = torch.arange(seq_len_total, device=seqs[0].device, dtype=torch.long)
    time_coords = torch.zeros((batch_size, seq_len_total), device=seqs[0].device, dtype=torch.float32)
    use_time_rope = torch.zeros((batch_size, seq_len_total), device=seqs[0].device, dtype=torch.bool)

    for b in range(batch_size):
        item_types: list[str] = []
        items: list[None | str] = []
        audio_times: list[None | float] = []

        for seq, seq_type in zip(seqs, seq_types):
            length = seq.shape[1]

            if seq_type == "audio":
                for i in range(length):
                    item_types.append("audio_latent")
                    items.append(None)
                    audio_times.append(i / float(audio_fps))

            elif seq_type == "id":
                if id_to_token is None:
                    for _ in range(length):
                        item_types.append("non_temporal")
                        items.append(None)
                        audio_times.append(None)
                else:
                    ids = seq[b].detach().cpu().tolist()
                    tokens = id_to_token(ids)
                    assert len(tokens) == length
                    for tok in tokens:
                        ttype = token_type(tok, event_attribute_prefixes=event_attribute_prefixes)
                        item_types.append(ttype)
                        items.append(tok)
                        audio_times.append(None)
            else:
                raise ValueError(seq_type)

        _, tc, utr = assign_time_coords(
            item_types=item_types,
            items=items,
            audio_times=audio_times,
            audio_fps=audio_fps,
            token_fps=token_fps,
            alpha=alpha,
            strict_event_time=strict_event_time,
        )

        time_coords[b] = torch.tensor(tc, device=seqs[0].device, dtype=torch.float32)
        use_time_rope[b] = torch.tensor(utr, device=seqs[0].device, dtype=torch.bool)

    return pos_ids, time_coords, use_time_rope


def infer_current_event_time_from_tokens(
    tokens: list[str],
    token_fps: float = 100.0,
    event_attribute_prefixes: tuple[str, ...] = DEFAULT_EVENT_ATTRIBUTE_PREFIXES,
) -> None | float:
    current_event_time: None | float = None
    for tok in tokens:
        current_event_time = update_decode_state(
            new_token=tok,
            current_event_time=current_event_time,
            token_fps=token_fps,
            event_attribute_prefixes=event_attribute_prefixes,
        )
    return current_event_time