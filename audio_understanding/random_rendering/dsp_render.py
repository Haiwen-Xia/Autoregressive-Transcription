import numpy as np
from scipy.signal import butter, sosfilt


def midi_to_hz(pitch: float) -> float:
    return 440.0 * (2.0 ** ((pitch - 69.0) / 12.0))


def gaussian_weight(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-8)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def random_lowpass(x: np.ndarray, cutoff_hz: float, sr: int, order: int = 2) -> np.ndarray:
    cutoff_hz = np.clip(cutoff_hz, 20.0, 0.45 * sr)
    sos = butter(order, cutoff_hz, btype="low", fs=sr, output="sos")
    y = sosfilt(sos, x)
    return y.astype(x.dtype, copy=False)


def make_envelope(
    t: np.ndarray,
    duration: float,
    release_time: float,
    attack_time: float,
    attack_shape: float,
    body_decay_time: float,
) -> np.ndarray:
    """
    duration = attack + decay body
    release is appended after duration
    """
    env = np.zeros_like(t, dtype=np.float64)

    attack_mask = t < attack_time
    if np.any(attack_mask):
        env[attack_mask] = (t[attack_mask] / max(attack_time, 1e-8)) ** attack_shape

    body_mask = (t >= attack_time) & (t < duration)
    if np.any(body_mask):
        env[body_mask] = np.exp(-(t[body_mask] - attack_time) / max(body_decay_time, 1e-8))

    note_off_level = float(np.exp(-max(duration - attack_time, 0.0) / max(body_decay_time, 1e-8)))

    release_mask = t >= duration
    if np.any(release_mask):
        env[release_mask] = note_off_level * np.exp(-(t[release_mask] - duration) / max(release_time, 1e-8))

    return np.maximum(env, 1e-8)


# improvements: more randomized amplitude, e.g. random rolloff;
def generate_core(
    t: np.ndarray,
    f0: float,
    sr: int,
    rng: np.random.Generator,
    velocity: float,
    min_harmonics: int = 8,
    max_harmonics: int = 40,
    harmonic_rolloff_range: tuple[float, float] = (1.2, 2.3),
    harmonic_jitter_db: float = 3.0,
    bw_base_range: tuple[float, float] = (0.8, 2.5),
    bw_growth_range: tuple[float, float] = (0.12, 0.35),
    decay_base_range: tuple[float, float] = (1.2, 2.8),
    decay_highfreq_range: tuple[float, float] = (10.0, 1000.0),
):
    """
    Generate a randomized harmonic core directly in time domain.

    Returns
    -------
    core : np.ndarray
        Unenveloped sum of harmonics with per-harmonic decay.
    freq_decay_info : dict
        Debug/tuning info for the frequency-decay calculation.
    """
    n = len(t)
    nyquist = 0.5 * sr
    vel = float(np.clip(velocity, 0.0, 1.0))

    k_max = min(max_harmonics, max(min_harmonics, int(nyquist / max(f0, 1.0))))
    rolloff = rng.uniform(*harmonic_rolloff_range)
    bw_base = rng.uniform(*bw_base_range)
    bw_growth = rng.uniform(*bw_growth_range)

    decay_base = rng.uniform(*decay_base_range)
    decay_highfreq = rng.uniform(*decay_highfreq_range)
    decay_shape = 1.2 + 0.8 * rng.random()

    jitter_sigma = harmonic_jitter_db / 20.0 * np.log(10.0)

    harmonic_freqs = []
    harmonic_amps = []
    harmonic_bandwidths = []
    harmonic_decay_rates = []

    core = np.zeros(n, dtype=np.float64)

    for k in range(1, k_max + 1):
        fk = k * f0
        if fk >= 0.48 * sr:
            break

        # harmonic amplitude
        amp = (k ** (-rolloff))
        amp *= np.exp(-fk / (3500.0 + 2500.0 * vel))
        amp *= np.exp(jitter_sigma * rng.standard_normal())

        # randomized "bandwidth" control
        # used here as a detuned side-partial spread amount
        bw = bw_base * (k ** bw_growth) * rng.uniform(0.85, 1.25)

        # frequency-dependent decay:
        # higher harmonics decay faster
        freq_norm = fk / max(nyquist, 1.0)
        decay_rate = decay_base + decay_highfreq * (freq_norm ** decay_shape)

        phase0 = 2.0 * np.pi * rng.random()

        # main partial
        partial = np.sin(2.0 * np.pi * fk * t + phase0)

        # gaussian-like local blur around the harmonic:
        # approximate by two weak detuned side partials
        detune_hz = min(bw, 0.15 * f0, 0.02 * fk + 1e-6)
        if detune_hz > 1e-6:
            side_weight = 0.18 + 0.10 * rng.random()
            p1 = np.sin(2.0 * np.pi * (fk - detune_hz) * t + 2.0 * np.pi * rng.random())
            p2 = np.sin(2.0 * np.pi * (fk + detune_hz) * t + 2.0 * np.pi * rng.random())
            partial = partial + side_weight * (p1 + p2)

        partial *= np.exp(-decay_rate * t)

        core += amp * partial

        harmonic_freqs.append(fk)
        harmonic_amps.append(amp)
        harmonic_bandwidths.append(bw)
        harmonic_decay_rates.append(decay_rate)

    freq_decay_info = {
        "harmonic_freqs": np.array(harmonic_freqs, dtype=np.float64),
        "harmonic_amps": np.array(harmonic_amps, dtype=np.float64),
        "harmonic_bandwidths": np.array(harmonic_bandwidths, dtype=np.float64),
        "harmonic_decay_rates": np.array(harmonic_decay_rates, dtype=np.float64),
        "rolloff": rolloff,
        "decay_base": decay_base,
        "decay_highfreq": decay_highfreq,
        "decay_shape": decay_shape,
    }
    return core, freq_decay_info

def generate_attack_flavor(
    t: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    amp_range: tuple[float, float] = (0.01, 0.05),
    decay_range: tuple[float, float] = (0.004, 0.018),
    band_center_range: tuple[float, float] = (1500.0, 5000.0),
    band_bw_range: tuple[float, float] = (500.0, 2200.0),
) -> np.ndarray:
    n = len(t)
    amp = rng.uniform(*amp_range)
    decay_time = rng.uniform(*decay_range)

    noise = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    noise_spec = np.fft.rfft(noise)

    band_center = rng.uniform(*band_center_range)
    band_bw = rng.uniform(*band_bw_range)
    band = gaussian_weight(freqs, band_center, band_bw)

    atk = np.fft.irfft(noise_spec * band, n=n)
    atk *= amp * np.exp(-t / max(decay_time, 1e-8))
    return atk


def pianoish_fast(
    duration: float = 1.2,
    pitch: float = 60,
    velocity: float = 0.8,
    sr: int = 48000,
    seed: int | None = None,
    release_range: tuple[float, float] = (0.04, 0.14),
    attack_time_range: tuple[float, float] = (0.004, 0.012),
    attack_shape_range: tuple[float, float] = (1.6, 2.8),
    body_decay_base: float = 0.22,
    body_decay_span: float = 1.00,
    lowpass_range: tuple[float, float] = (2500.0, 8500.0),
    lowpass_order: int = 2,
    use_lowpass: bool = True,
    core_mode: str = "harmonic",
) -> np.ndarray:
    """
    Fast piano-ish MVP.

    Inputs
    ------
    duration : float
        Duration of attack + decay body.
    pitch : float
        MIDI pitch.
    velocity : float
        0..1 normalized velocity.
    sr : int
        Sample rate.

    Returns
    -------
    y : np.ndarray
        Float32 waveform.
    """
    rng = np.random.default_rng(seed)
    vel = float(np.clip(velocity, 0.0, 1.0))
    f0 = midi_to_hz(pitch)

    release_time = rng.uniform(*release_range)
    total_duration = duration + release_time

    n = int(np.round(total_duration * sr))
    t = np.arange(n, dtype=np.float64) / sr

    # envelope
    attack_time = rng.uniform(*attack_time_range)
    attack_shape = rng.uniform(*attack_shape_range)

    # attack time is NOT velocity-dependent
    # body decay can remain mildly velocity-dependent
    body_decay_time = body_decay_base + 0.20 * rng.random()

    env = make_envelope(
        t=t,
        duration=duration,
        release_time=release_time,
        attack_time=attack_time,
        attack_shape=attack_shape,
        body_decay_time=body_decay_time,
    )

    # core
    if core_mode == "harmonic":
        core, _freq_decay_info = generate_core(
            t=t,
            f0=f0,
            sr=sr,
            rng=rng,
            velocity=vel,
        )
    else:
        raise ValueError("core_mode must be one of: 'harmonic', 'saw_fft_decay'")

    # attack flavor
    attack_flavor = generate_attack_flavor(
        t=t,
        sr=sr,
        rng=rng,
    )

    # combine
    y = core * env + attack_flavor

    # optional lowpass
    if use_lowpass:
        lp_cutoff = rng.uniform(*lowpass_range)
        y = random_lowpass(y, lp_cutoff, sr, order=lowpass_order)

    # gentle final shaping
    y *= np.minimum(1.10 * env, 1.0)

    # normalize
    peak = np.max(np.abs(y)) + 1e-9
    y = 0.95 * y / peak

    return y.astype(np.float32)


if __name__ == "__main__":
    # example
    y = pianoish_fast(
        duration=1.0,
        pitch=60,
        velocity=0.7,
        sr=48000,
        seed=0,
        use_lowpass=False,
        core_mode="harmonic",
    )
    print(y.shape, y.dtype, y.min(), y.max())
    import soundfile as sf
    sf.write("pianoish_fast_example.wav", y, 48000)
    import matplotlib.pyplot as plt
    plt.specgram(y, Fs=48000, scale="dB")
    plt.show()
