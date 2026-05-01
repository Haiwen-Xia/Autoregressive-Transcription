"""
Ablation runner for DSP piano variants.

Variants compared:
  1. pianoish_fast     -- original, randomized harmonic core + lowpass
  2. pianoish_physics  -- inharmonic partials + per-harmonic tau decay, randomized
  3. pianoish_fixed    -- derandomized: fixed envelope/decay, no filter, no attack flavor

For each variant, renders:
  - a single note (C4, C5, C6) at two velocities (soft=0.3, loud=0.9)
  - a short chord (C4+E4+G4)

Outputs go to ablation_out/<variant>/<note_or_chord>.wav
Mel spectrograms are saved to ablation_out/spectrogram_<variant>.png
"""
import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa

from dsp_render import pianoish_fast, pianoish_physics, pianoish_fixed

SR = 48000
OUT_DIR = "ablation_out"

SINGLE_NOTES = [
    ("C4", 60),
    ("C5", 72),
    ("C6", 84),
]
VELOCITIES = [("soft", 0.3), ("loud", 0.9)]

CHORD_PITCHES = [60, 64, 67]  # C4 E4 G4

VARIANTS = {
    "pianoish_fast": lambda pitch, vel, dur, seed: pianoish_fast(
        duration=dur, pitch=pitch, velocity=vel, sr=SR, seed=seed,
        use_lowpass=True,
    ),
    "pianoish_physics": lambda pitch, vel, dur, seed: pianoish_physics(
        duration=dur, pitch=pitch, velocity=vel, sr=SR, seed=seed,
        use_lowpass=True,
    ),
    "pianoish_fixed": lambda pitch, vel, dur, seed: pianoish_fixed(
        duration=dur, pitch=pitch, velocity=vel, sr=SR, seed=seed,
    ),
}


def mix_to_stereo(y: np.ndarray) -> np.ndarray:
    """Mono float32 -> stereo float32."""
    return np.stack([y, y], axis=1)


def render_chord(variant_fn, pitches, velocity, duration=2.0, seed_base=0):
    total = int(np.round((duration + 0.2) * SR))
    buf = np.zeros(total, dtype=np.float32)
    for i, pitch in enumerate(pitches):
        y = variant_fn(pitch, velocity, duration, seed_base + i)
        buf[:len(y)] += y
    peak = float(np.max(np.abs(buf))) + 1e-9
    if peak > 1.0:
        buf = 0.95 * buf / peak
    return buf


def render_all():
    os.makedirs(OUT_DIR, exist_ok=True)

    for variant_name, fn in VARIANTS.items():
        vdir = os.path.join(OUT_DIR, variant_name)
        os.makedirs(vdir, exist_ok=True)

        # single notes
        for note_name, pitch in SINGLE_NOTES:
            for vel_name, vel in VELOCITIES:
                y = fn(pitch, vel, 1.5, seed=42)
                fname = os.path.join(vdir, f"{note_name}_{vel_name}.wav")
                sf.write(fname, mix_to_stereo(y), SR)
                print(f"  {fname}")

        # chord
        for vel_name, vel in VELOCITIES:
            y = render_chord(fn, CHORD_PITCHES, vel, duration=2.0, seed_base=100)
            fname = os.path.join(vdir, f"chord_C4E4G4_{vel_name}.wav")
            sf.write(fname, mix_to_stereo(y), SR)
            print(f"  {fname}")

    print("Done rendering.")


def plot_spectrograms():
    """
    For each variant, plot mel spectrogram of C4 loud single note.
    Saves ablation_out/spectrograms.png
    """
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(5 * len(VARIANTS), 4), squeeze=False)
    axes = axes[0]

    for ax, (variant_name, fn) in zip(axes, VARIANTS.items()):
        y = fn(60, 0.9, 1.5, seed=42)
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=2048, hop_length=256, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        img = ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma",
                        extent=[0, len(y) / SR, 0, SR // 2])
        ax.set_title(variant_name, fontsize=9)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("freq (Hz)")
        plt.colorbar(img, ax=ax, format="%+2.0f dB")

    plt.suptitle("DSP Piano Ablation — C4 loud (mel spectrogram)")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "spectrograms.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


def main():
    print("=== Rendering variants ===")
    render_all()
    print("\n=== Plotting spectrograms ===")
    plot_spectrograms()


if __name__ == "__main__":
    main()
