import torch
import math
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from torch import Tensor
import pretty_midi


class Piano:
    def __init__(self, sr: int):
        self.sr = sr
        self.K = 20

    def __call__(self, f0: float, duration: float, velocity: float) -> Tensor:

        t = torch.arange(round(self.sr * duration), dtype=torch.float)
        audio = torch.zeros_like(t)

        for k in range(1, self.K + 1):
            fk = self.get_fk(f0, k)
            ak = self.get_ak(f0, k, t)

            mode = ak * torch.cos(2 * math.pi * fk * t / self.sr)
            mode *= (velocity / 128)
            audio += mode

        return audio

    def get_fk(self, f0: float, k: int) -> float:
        B = 0.0008
        return k * f0 * math.sqrt(1 + B * k**2)

    def get_ak(self, f0: float, k: int, t: Tensor) -> Tensor:
        alpha = 1.0 
        ak = 1.0 / (k**alpha)
        Hk = 1.0
        tau_k = 1.0 / (k ** 1.5)  # 1.0 - 1.5 for piano
        env = torch.exp(-t / self.sr / tau_k)
        return ak * Hk * env


def pitch_to_freq(pitch):
    return 440 * 2 ** ((pitch - 69) / 12)


def run_single_note():
    sr = 48000
    vst = Piano(sr)
    audio = vst(f0=440., duration=2.0, velocity=100)

    audio = audio / audio.abs().max()
    sf.write("_zz.wav", audio.numpy(), sr)

    if True:  # debug
        stft = librosa.feature.melspectrogram(y=audio.numpy(), sr=sr, n_fft=2048, hop_length=320, n_mels=256)
        plt.matshow(stft, origin='lower', aspect='auto', cmap='jet')
        plt.savefig("_zz.pdf")


def run_midi():

    sr = 48000
    vst = Piano(sr)

    midi_path = "./assets/MIDI-Unprocessed_Recital17-19_MID--AUDIO_18_R1_2018_wav--3.midi"
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    notes = midi_data.instruments[0].notes

    buffer = torch.zeros(math.ceil(notes[-1].end * sr))  # (l,)

    for i, note in enumerate(notes):

        start = note.start
        end = note.end
        velocity = note.velocity
        f0 = pitch_to_freq(note.pitch)
        # duration = end - start
        duration = 0.2
        audio = vst(f0, duration, velocity)

        bgn = round(start * sr)
        buffer[bgn : bgn + audio.shape[-1]] += audio
        
        if i == 200:
            break

    audio = audio / audio.abs().max() / 2
    sf.write("_zz.wav", buffer.numpy(), sr)
    

if __name__ == '__main__':
    run_single_note()
    run_midi()