# Implement Time-Aware RoPE for Decoder-Only Music Transcription

You are modifying a decoder-only Transformer used for autoregressive music transcription.

## Goal

Implement a **time-aware positional encoding** on top of standard RoPE.

The key idea:

- standard RoPE models **sequence order**
- time-aware RoPE models **physical time on the audio timeline**
- for each token, we mix both:

  $$

  \text{mixed}(x_i) = c \cdot \text{RoPE}(x_i, p_i) + (1-c) \cdot \text{RoPE}_t(x_i, u_i)

$$

where:

- $p_i$ = token position in sequence
- $u_i$ = real-valued time coordinate
- $c \in [0,1]$ = mixing weight

---

## Task assumptions

### Input sequence structure

The model input is conceptually:

```text
[audio_latents] + [optional prompt tokens] + [autoregressive output tokens]
```

### Output tokenization

Use event-style tokenization like:

```text
time=τ  onset  pitch=p  velocity=v
```

All 4 tokens are **time-related** and belong to the **same event time τ**.

That means:

- `time=τ` uses time coordinate `τ`
- `onset` also uses time coordinate `τ`
- `pitch=p` also uses time coordinate `τ`
- `velocity=v` also uses time coordinate `τ`

Important: in music transcription, **almost all output tokens are temporal**.

### Non-temporal tokens

Examples:

- `Question: Please transcribe this audio`
- `The instruments are violin...`

These do **not** belong to any event timestamp.

Implementation rule:

- simplest choice: use **standard RoPE only**
- do **not** apply time-aware branch to these tokens unless explicitly needed

### Audio latent tokens

Each audio latent corresponds to a real audio time (or frame center time).
Audio latents should also use time-aware RoPE.

---

## Required behavior

Implement a time coordinate assignment rule:

1. audio latent token
   -> `u_i = alpha_audio * audio_time_i`

2. timestamp token `time=τ`
   -> `u_i = alpha_out * τ`

3. event tokens after a timestamp, such as:
   - `onset`
   - `pitch=p`
   - `velocity=v`
   - future extensible tokens like `offset`, `duration`

   -> inherit the most recent event time:
   `u_i = alpha_out * current_event_time`

4. non-temporal prompt/control tokens
   -> no time-aware branch; use standard RoPE only

---

## Mathematical definition

For standard RoPE:

$$

\text{RoPE}(x_i, p_i)

$$

For time-aware RoPE:

$$

\text{RoPE}_t(x_i, u_i) = R(u_i)x_i

$$

where each 2D pair is rotated by time coordinate $u_i$:

$$

R(u)=
\begin{pmatrix}
\cos(\omega u) & -\sin(\omega u) \\
\sin(\omega u) & \cos(\omega u)
\end{pmatrix}

$$

Then for query/key:

$$

\tilde q_i = c \cdot \text{RoPE}(q_i, p_i) + (1-c)\cdot \text{RoPE}_t(q_i, u_i)

$$

$$

\tilde k_i = c \cdot \text{RoPE}(k_i, p_i) + (1-c)\cdot \text{RoPE}_t(k_i, u_i)

$$

For non-temporal tokens, just use:

$$

\tilde q_i = \text{RoPE}(q_i, p_i),\quad \tilde k_i = \text{RoPE}(k_i, p_i)

$$

---

## What to implement

### 1. Token type handling

Add token categorization helpers:

- `audio_latent`
- `timestamp`
- `event_attribute`
- `non_temporal`

Suggested mapping:

- `time=...` -> `timestamp`
- `onset`, `pitch=...`, `velocity=...`, `offset`, `duration` -> `event_attribute`
- prompt/instruction/description tokens -> `non_temporal`

### 2. Time state during decoding

Maintain:

```python
current_event_time: Optional[float]
```

Rules:

- when token is `time=τ`, parse τ and set `current_event_time = τ`
- all following event-attribute tokens inherit that time
- when next `time=τ'` appears, update `current_event_time = τ'`

### 3. Time coordinates for full context

For every forward pass, build per-token arrays:

- `pos_ids[i]`
- `time_coords[i]`
- `use_time_rope[i]`

### 4. Mixed RoPE application

Modify Q/K preparation so that:

- temporal tokens use mixed RoPE
- non-temporal tokens use standard RoPE only

### 5. Keep implementation extensible

Design so more event tokens can be added later:

- `offset`
- `duration`
- `program`
- etc.

Any token that semantically belongs to the current timestamp should inherit `current_event_time`.

---

## Pseudocode

```python
# this should be handled by your tokenizer, but we include it here for completeness
# or, the tokens between timestamps are all event attributes by design, so we can just check if we're in an "event attribute mode"
def token_type(token: str) -> str:
    if is_audio_latent(token):
        return "audio_latent"
    if token.startswith("time="):
        return "timestamp"
    if token == "onset":
        return "event_attribute"
    if token.startswith("pitch="):
        return "event_attribute"
    if token.startswith("program="):
        return "event_attribute"
    if token.startswith("drum_pitch="):
        return "event_attribute"
    if token.startswith("velocity="):
        return "event_attribute"
    if token.startswith("offset="):
        return "event_attribute"
    if token.startswith("duration="):
        return "event_attribute"
    return "non_temporal"
```

```python
def assign_time_coords(tokens, audio_times, alpha_audio, alpha_out):
    """
    tokens: full context tokens
    audio_times: aligned times for audio latent tokens
    returns:
        pos_ids: List[int]
        time_coords: List[float]
        use_time_rope: List[bool]
    """
    pos_ids = []
    time_coords = []
    use_time_rope = []

    current_event_time = None
    audio_ptr = 0

    for i, tok in enumerate(tokens):
        pos_ids.append(i)
        ttype = token_type(tok)

        if ttype == "audio_latent":
            t = audio_times[audio_ptr]
            audio_ptr += 1
            time_coords.append(alpha_audio * t)
            use_time_rope.append(True)

        elif ttype == "timestamp":
            tau = parse_time_token(tok)   # e.g. "time=1.24" -> 1.24, for our code needs to * 0.01 to convert to seconds 
            current_event_time = tau
            time_coords.append(alpha_out * tau)
            use_time_rope.append(True)

        elif ttype == "event_attribute":
            assert current_event_time is not None, \
                "Event attribute token seen before any timestamp token"
            time_coords.append(alpha_out * current_event_time)
            use_time_rope.append(True)

        else:
            time_coords.append(0.0)      # placeholder only
            use_time_rope.append(False)

    return pos_ids, time_coords, use_time_rope
```

```python
def apply_rotary(x, coord, inv_freq):
    """
    x: [..., dim]
    coord: scalar or tensor broadcastable to x
    inv_freq: [dim // 2]
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    angle = coord[..., None] * inv_freq
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out
```

```python
def apply_mixed_rope(q, k, pos_ids, time_coords, use_time_rope, inv_freq, c):
    """
    q, k: [N, D]
    pos_ids: [N]
    time_coords: [N]
    use_time_rope: [N] bool
    """
    q_pos = apply_rotary(q, pos_ids, inv_freq)
    k_pos = apply_rotary(k, pos_ids, inv_freq)

    q_out = q_pos.clone()
    k_out = k_pos.clone()

    temporal_idx = use_time_rope.bool()
    if temporal_idx.any():
        q_time = apply_rotary(q[temporal_idx], time_coords[temporal_idx], inv_freq)
        k_time = apply_rotary(k[temporal_idx], time_coords[temporal_idx], inv_freq)

        q_out[temporal_idx] = c * q_pos[temporal_idx] + (1.0 - c) * q_time
        k_out[temporal_idx] = c * k_pos[temporal_idx] + (1.0 - c) * k_time

    return q_out, k_out
```

```python
def update_decode_state(new_token, current_event_time):
    ttype = token_type(new_token)
    if ttype == "timestamp":
        current_event_time = parse_time_token(new_token)
    return current_event_time
```

---

## Expected semantics

For this sequence:

```text
time=1.24 onset pitch=67 velocity=72 time=1.56 onset pitch=69 velocity=75
```

the time coordinates should be:

```text
time=1.24   -> 1.24
onset       -> 1.24
pitch=67    -> 1.24
velocity=72 -> 1.24

time=1.56   -> 1.56
onset       -> 1.56
pitch=69    -> 1.56
velocity=75 -> 1.56
```

after applying the output-side scale `alpha_out`.

---

## Engineering guidance

- keep `alpha_audio`, `alpha_out`, and `c` configurable
- okay to start with fixed scalars
- keep the design compatible with KV-cache decoding
- do not hardcode only `onset/pitch/velocity`; make event-attribute detection extensible
- fail loudly if an event-attribute token appears before any timestamp token
- preserve causal masking; only change the Q/K positional encoding path



Please implement:

1. token type classification
2. time coordinate assignment
3. mixed RoPE application on Q/K
4. decode-state update for `current_event_time`
5. minimal unit tests covering:
   - prompt tokens
   - audio latent tokens
   - timestamp inheritance
   - multiple consecutive events
   - malformed sequence where event attributes appear before timestamp
```

One implementation caveat to keep in mind: your note suggests using a scale factor `α` to align temporal scales across audio latents and output tokens, and mixing standard RoPE with time-aware RoPE instead of fully replacing it.
For simplicity, you can start with α="the fps for audio latents"


