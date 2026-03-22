from dataclasses import dataclass

import soundfile as sf
import torch
import torchaudio

from audio_understanding.audio_encoders.conformer2d import Conformer2D
from audio_understanding.audio_encoders.conformer2d_nopool import Conformer2D as Conformer2DNoPool
from audio_understanding.audio_encoders.mert import MERT
from audio_understanding.audio_encoders.muq import MuQ
from audio_understanding.audio_encoders.panns import PannsCnn14
from audio_understanding.audio_encoders.piano_transcription_crnn import PianoTranscriptionCRnn


@dataclass
class EncoderSpec:
    name: str
    cls: type
    expected_fps: float


def _load_audio_mono_16k(audio_path: str) -> torch.Tensor:
    audio, sr = sf.read(audio_path)
    audio_tensor = torch.tensor(audio, dtype=torch.float32)

    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor[None, :]
    else:
        audio_tensor = audio_tensor.transpose(0, 1)

    audio_tensor = audio_tensor[None, :, :]
    if sr != 16000:
        audio_tensor = torchaudio.functional.resample(
            waveform=audio_tensor,
            orig_freq=sr,
            new_freq=16000,
        )

    audio_tensor = torch.mean(audio_tensor, dim=1, keepdim=True)

    return audio_tensor


def main_func(audio_path: str) -> None:
    specs = [
        EncoderSpec(name="Conformer2D", cls=Conformer2D, expected_fps=25.0),
        EncoderSpec(name="Conformer2DNoPool", cls=Conformer2DNoPool, expected_fps=100.0),
        EncoderSpec(name="PannsCnn14", cls=PannsCnn14, expected_fps=1.0),
        EncoderSpec(name="MERT", cls=MERT, expected_fps=75.0),
        EncoderSpec(name="MuQ", cls=MuQ, expected_fps=25.0),
        EncoderSpec(name="PianoTranscriptionCRnn", cls=PianoTranscriptionCRnn, expected_fps=100.0),
    ]

    audio_tensor = _load_audio_mono_16k(audio_path)
    duration_sec = audio_tensor.shape[-1] / 16000.0
    print(f"audio shape={tuple(audio_tensor.shape)}, sr=16000, duration={duration_sec:.3f}s")

    for spec in specs:
        encoder = spec.cls(sr=16000, trainable=False)
        latent = encoder.encode(audio=audio_tensor, train_mode=False)
        observed_fps = latent.shape[1] / duration_sec
        print(
            f"{spec.name}: latent={tuple(latent.shape)}, "
            f"encoder.fps={encoder.fps:.3f}, expected={spec.expected_fps:.3f}, "
            f"observed={observed_fps:.3f}"
        )
        # Some encoders return chunk-level summaries; observed_fps can differ from encoder.fps.
        assert abs(float(encoder.fps) - spec.expected_fps) < 1e-6


if __name__ == "__main__":
    main_func("/data/yrb/musicarena/Haiwen/Autoregressive-Transcription/assets/audios/duo.mp3")
