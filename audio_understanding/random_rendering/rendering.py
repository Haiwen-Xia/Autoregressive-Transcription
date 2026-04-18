import os
import wave
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class MidiNote:
    pitch: int
    velocity: int      # 0~127
    program: int
    start: float
    dur: float


class Sampler:
    def __init__(self, time: float, sr: int, data_dir: Optional[str] = None):
        self.time = float(time)
        self.sr = int(sr)
        self.total_samples = int(round(self.time * self.sr))

        # notes[program][pitch] -> np.ndarray, shape [N], float32, mono, [-1, 1]
        self.standard_notes: Dict[int, Dict[int, np.ndarray]] = {}

        # event list
        self.notes: List[MidiNote] = []
        if data_dir is not None:
            self.read_data(data_dir)

    # ----------------------------
    # WAV loading
    # ----------------------------
    @staticmethod
    def _read_wav_mono_float32(path: str, target_sr: Optional[int] = None) -> np.ndarray:
        """
        Read PCM wav using stdlib wave.
        Supports:
          - int16
          - int32
          - uint8
        Converts to mono float32 in [-1, 1].
        If target_sr is provided and different, does a simple linear resample.
        """
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if sampwidth == 1:
            # unsigned 8-bit PCM: 0~255 -> [-1, 1]
            x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            x = (x - 128.0) / 128.0
        elif sampwidth == 2:
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            x = x / 32768.0
        elif sampwidth == 4:
            x = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
            x = x / 2147483648.0
        else:
            raise ValueError(f"Unsupported sampwidth={sampwidth} in {path}")

        if n_channels > 1:
            x = x.reshape(-1, n_channels).mean(axis=1)

        if target_sr is not None and framerate != target_sr:
            x = Sampler._linear_resample(x, framerate, target_sr)

        return np.asarray(x, dtype=np.float32)

    @staticmethod
    def _linear_resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr or x.size == 0:
            return np.asarray(x, dtype=np.float32)

        old_n = x.shape[0]
        new_n = int(round(old_n * dst_sr / src_sr))
        if new_n <= 1:
            return np.asarray(x[:1], dtype=np.float32)

        old_idx = np.arange(old_n, dtype=np.float32)
        new_idx = np.linspace(0, old_n - 1, new_n, dtype=np.float32)
        y = np.interp(new_idx, old_idx, x).astype(np.float32)
        return y

    def read_data(self, data_dir: str):
        """
        Load wavs from:
            data_dir/{program_num}/{pitch}.wav

        Example:
            data/0/60.wav
            data/0/61.wav
            data/24/64.wav
        """
        data_dir = os.path.abspath(data_dir)
        loaded = 0

        for program_name in os.listdir(data_dir):
            program_path = os.path.join(data_dir, program_name)
            if not os.path.isdir(program_path):
                continue

            try:
                program_num = int(program_name)
            except ValueError:
                continue

            if program_num not in self.standard_notes:
                self.standard_notes[program_num] = {}

            for filename in os.listdir(program_path):
                if not filename.lower().endswith(".wav"):
                    continue

                stem = os.path.splitext(filename)[0]
                try:
                    pitch = int(stem)
                except ValueError:
                    continue

                wav_path = os.path.join(program_path, filename)
                x = self._read_wav_mono_float32(wav_path, target_sr=self.sr)

                # truncate / keep as loaded
                self.standard_notes[program_num][pitch] = x
                loaded += 1

        if loaded == 0:
            raise RuntimeError(f"No wav files found under {data_dir}")

    # ----------------------------
    # MIDI note handling
    # ----------------------------
    @staticmethod
    def velocity_to_gain(velocity: int, vmin: float = 0.1, vmax: float = 1.0) -> float:
        """
        Map MIDI velocity 0~127 -> [vmin, vmax]
        Linear mapping.
        """
        v = int(np.clip(velocity, 0, 127))
        return float(vmin + (v / 127.0) * (vmax - vmin))

    @staticmethod
    def _apply_fade_out_inplace(x: np.ndarray, sr: int, fade_ms: float = 5.0):
        if x.size == 0:
            return

        m = int(round(sr * fade_ms / 1000.0))
        m = min(m, x.size)
        if m <= 1:
            x[-1] = 0.0
            return

        fade = np.linspace(1.0, 0.0, m, dtype=x.dtype)
        x[-m:] *= fade

    def set_midi_note(self, notes: List[Dict]):
        self.notes = [MidiNote(**note) for note in notes]
        
    def add_midi_note(self, pitch: int, velocity: int, program: int, start: float, dur: float):
        assert start >= 0.0
        assert dur > 0.0
        # assert start + dur < self.time

        if program not in self.standard_notes:
            raise KeyError(f"program {program} not loaded")
        if pitch not in self.standard_notes[program]:
            raise KeyError(f"program {program}, pitch {pitch} not loaded")

        self.notes.append(
            MidiNote(
                pitch=int(pitch),
                velocity=int(velocity),
                program=int(program),
                start=float(start),
                dur=float(dur),
            )
        )

    # ----------------------------
    # Rendering
    # ----------------------------
    def _max_base_len(self) -> int:
        if not self.standard_notes:
            return 0
        return max(
            arr.shape[0]
            for pitch_dict in self.standard_notes.values()
            for arr in pitch_dict.values()
        )

    def render(
        self,
        fade_ms: float = 20.0,
        clip: bool = True,
        velocity_min: float = 0.1,
        velocity_max: float = 1.0,
        time: Optional[float] = None,
    ) -> np.ndarray:
        if time is not None:
            self.time = float(time)
            self.total_samples = int(round(self.time * self.sr))
        out = np.zeros(self.total_samples, dtype=np.float32)
        if not self.notes:
            return out

        fade_samples = int(round(self.sr * fade_ms / 1000.0))
        fade_cache: Dict[int, np.ndarray] = {}
        gain_scale = (velocity_max - velocity_min) / 127.0
        scratch = np.empty(self._max_base_len(), dtype=np.float32)

        for note in self.notes:
            base = self.standard_notes[note.program][note.pitch]

            start_idx = int(round(note.start * self.sr))
            dur_samples = int(round(note.dur * self.sr))
            if dur_samples <= 0:
                continue

            end_idx = min(start_idx + dur_samples, self.total_samples)
            seg_len = min(end_idx - start_idx, base.shape[0])
            if seg_len <= 0:
                continue

            v = min(max(note.velocity, 0), 127)
            gain = velocity_min + v * gain_scale

            # multiply into pre-allocated scratch (zero per-note allocation)
            np.multiply(base[:seg_len], gain, out=scratch[:seg_len])

            # apply cached fade-out to tail
            fl = min(fade_samples, seg_len)
            if fl > 1:
                if fl not in fade_cache:
                    fade_cache[fl] = np.linspace(1.0, 0.0, fl, dtype=np.float32)
                scratch[seg_len - fl:seg_len] *= fade_cache[fl]
            elif fl == 1:
                scratch[seg_len - 1] = 0.0

            # mix
            out[start_idx:start_idx + seg_len] += scratch[:seg_len]

        if clip:
            np.clip(out, -1.0, 1.0, out=out)

        return out

    def render_notes(
        self,
        notes: List[Dict],
        fade_ms: float = 20.0,
        clip: bool = True,
        velocity_min: float = 0.1,
        velocity_max: float = 1.0,
        time: Optional[float] = None,
    ) -> np.ndarray:
        """Stateless render: takes note dicts from sampler, returns audio.
        Does not mutate self.notes. Safe for parallel / repeated calls.
        """
        total_time = float(time) if time is not None else self.time
        total_samples = int(round(total_time * self.sr))
        out = np.zeros(total_samples, dtype=np.float32)
        if not notes:
            return out

        fade_samples = int(round(self.sr * fade_ms / 1000.0))
        fade_cache: Dict[int, np.ndarray] = {}
        gain_scale = (velocity_max - velocity_min) / 127.0
        scratch = np.empty(self._max_base_len(), dtype=np.float32)

        for n in notes:
            base = self.standard_notes[n["program"]][n["pitch"]]

            start_idx = int(round(n["start"] * self.sr))
            dur_samples = int(round(n["dur"] * self.sr))
            if dur_samples <= 0:
                continue

            end_idx = min(start_idx + dur_samples, total_samples)
            seg_len = min(end_idx - start_idx, base.shape[0])
            if seg_len <= 0:
                continue

            v = min(max(n["velocity"], 0), 127)
            gain = velocity_min + v * gain_scale

            np.multiply(base[:seg_len], gain, out=scratch[:seg_len])

            fl = min(fade_samples, seg_len)
            if fl > 1:
                if fl not in fade_cache:
                    fade_cache[fl] = np.linspace(1.0, 0.0, fl, dtype=np.float32)
                scratch[seg_len - fl:seg_len] *= fade_cache[fl]
            elif fl == 1:
                scratch[seg_len - 1] = 0.0

            out[start_idx:start_idx + seg_len] += scratch[:seg_len]

        if clip:
            np.clip(out, -1.0, 1.0, out=out)

        return out
def load_plugin_state_or_preset(
    synth,
    preset_path: Optional[str] = None,
    state_path: Optional[str] = None,
):
    if state_path is not None:
        synth.load_state(state_path)

    if preset_path is None:
        return

    if preset_path.endswith(".vstpreset"):
        synth.load_vst3_preset(preset_path)
        return

    synth.load_preset(preset_path)


def build_engine_and_synth(
    vst_path: str,
    sample_rate: int,
    buffer_size: int,
    preset_path: Optional[str] = None,
    state_path: Optional[str] = None,
):
    """Create a dawdreamer RenderEngine and load a VST plugin.

    Returns (engine, synth) ready for rendering. Calling code is responsible
    for calling engine.load_graph([(synth, [])]) after any graph setup.

    Paths with spaces cause JUCE to emit a harmless "invalid URI" warning to
    stderr (audio still works).  We suppress it by redirecting fd 2 at the
    C-level for the duration of make_plugin_processor.
    """
    import dawdreamer

    engine = dawdreamer.RenderEngine(sample_rate, buffer_size)

    # Redirect C-level stderr to /dev/null to swallow the URI warning that
    # JUCE emits when the plugin path contains spaces.
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    try:
        synth = engine.make_plugin_processor("synth", vst_path)
    finally:
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)

    load_plugin_state_or_preset(synth, preset_path, state_path)
    engine.load_graph([(synth, [])])
    return engine, synth


class DawDreamerSampler:
    """VST-based renderer using dawdreamer.

    Drop-in replacement for Sampler: implements render_notes() with the same
    signature so it works transparently with OnlineRenderingBuffer.

    Each worker process must have its own engine instance. Use _daw_worker_init
    as the pool initializer so each forked worker creates a fresh engine.
    """

    def __init__(
        self,
        time: float,
        sr: int,
        vst_path: str,
        buffer_size: int = 512,
        preset_path: Optional[str] = None,
        state_path: Optional[str] = None,
    ):
        self.time = float(time)
        self.sr = int(sr)
        self.vst_path = vst_path
        self.buffer_size = int(buffer_size)
        self.engine, self.synth = build_engine_and_synth(
            vst_path,
            sr,
            buffer_size,
            preset_path=preset_path,
            state_path=state_path,
        )
        self._n_params: Optional[int] = None  # lazily cached

        # Pianoteq continuously monitors its own files and prints
        # "auto_reload_after_external_changes" to C-level stderr after every
        # preset scan.  Permanently redirect fd 2 → /dev/null to silence it;
        # Python-level sys.stderr is unaffected.
        _devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull, 2)
        os.close(_devnull)

    def _get_n_params(self) -> int:
        if self._n_params is None:
            self._n_params = len(self.synth.get_parameters_description())
        return self._n_params

    def randomize_parameters(
        self,
        num_params: Optional[int] = None,
        param_indices: Optional[List[int]] = None,
        skip_indices: Optional[List[int]] = None,
        rng: Optional[np.random.Generator] = None,
        search_steps: int = 100,
    ) -> dict:
        """Randomly set VST parameters, respecting each parameter's valid range.

        Uses synth.get_parameter_range() to discover whether each parameter is
        discrete (e.g. a boolean switch) or continuous, then samples accordingly:
          - Discrete params  (few distinct keys)  → pick one of the valid keys.
          - Continuous params (many keys / single range) → uniform in discovered bounds.

        get_parameter_range returns a dict whose keys are the valid VST [0,1]
        float values (individual floats for discrete params, or a dense set for
        continuous params). set_parameter always takes a [0,1] float.

        Args:
            num_params:    How many randomly-chosen parameters to randomize.
                           Ignored when param_indices is given.
                           If None, randomizes all parameters.
            param_indices: Explicit list of parameter indices to randomize.
            skip_indices:  Parameter indices to leave untouched (e.g. Volume).
            rng:           numpy Generator for reproducibility.
            search_steps:  Steps passed to get_parameter_range for discovery.

        Returns:
            {index: value} dict of what was set.
        """
        if rng is None:
            rng = np.random.default_rng()

        skip_set = set(skip_indices) if skip_indices else set()
        n_total = self._get_n_params()

        if param_indices is not None:
            indices = [i for i in param_indices if i not in skip_set]
        elif num_params is None:
            indices = [i for i in range(n_total) if i not in skip_set]
        else:
            pool = [i for i in range(n_total) if i not in skip_set]
            indices = rng.choice(pool, size=min(num_params, len(pool)), replace=False).tolist()

        set_map = {}
        for idx in indices:
            idx = int(idx)
            par_range = self.synth.get_parameter_range(idx, search_steps=search_steps, convert=True)
            keys = list(par_range.keys())

            # Keys may be plain floats (discrete) or (lo, hi) tuples (range).
            if len(keys) == 0:
                val = float(rng.random())
            elif isinstance(keys[0], tuple):
                # Pick a random range segment, then sample uniformly within it.
                lo, hi = keys[int(rng.integers(len(keys)))]
                val = float(rng.uniform(lo, hi))
            else:
                # Discrete or densely sampled float keys.
                # Treat as discrete if few unique values, else uniform in [min, max].
                keys_f = sorted(float(k) for k in keys)
                if len(keys_f) <= search_steps // 2:
                    # Discrete — pick one of the valid settings.
                    val = keys_f[int(rng.integers(len(keys_f)))]
                else:
                    # Continuous — sample uniformly between the discovered bounds.
                    val = float(rng.uniform(keys_f[0], keys_f[-1]))

            self.synth.set_parameter(idx, val)
            set_map[idx] = val
        return set_map

    def reset_parameters(self):
        """Reset all parameters to their default values (0.5 midpoint heuristic).

        Note: dawdreamer doesn't expose get_default_parameter_value; resetting to
        0.5 is a safe heuristic for most VSTs.  Pass explicit values if you need
        deterministic defaults.
        """
        n_total = self._get_n_params()
        for i in range(n_total):
            self.synth.set_parameter(i, 0.5)

    def render_notes(
        self,
        notes: List[Dict],
        time: Optional[float] = None,
        **kwargs,  # absorb fade_ms, clip, velocity_min/max — VST envelope handles these
    ) -> np.ndarray:
        """Render notes via VST plugin. Returns mono float32 array, shape [N]."""
        total_time = float(time) if time is not None else self.time
        total_samples = int(round(total_time * self.sr))

        self.synth.clear_midi()
        for n in notes:
            self.synth.add_midi_note(
                int(n["pitch"]),
                int(n["velocity"]),
                float(n["start"]),
                float(n["dur"]),
            )

        self.engine.render(total_time)
        audio = self.synth.get_audio()  # shape: [channels, samples]
        if audio.ndim == 2:
            audio = audio.mean(axis=0)

        # trim / zero-pad to exact length
        audio = np.asarray(audio, dtype=np.float32)
        if audio.shape[0] >= total_samples:
            return audio[:total_samples]
        out = np.zeros(total_samples, dtype=np.float32)
        out[:audio.shape[0]] = audio
        return out


if __name__ == "__main__":
    # Example usage
    sampler = Sampler(time=5, sr=16000)
    sampler.read_data(r"D:\local\codes\dsp\Rendered")
    import pretty_midi
    import soundfile as sf
    with open(r"D:\local\codes\dsp\rendering\Fugue1.mid", "rb") as f:
        pm = pretty_midi.PrettyMIDI(f)
    for track in pm.instruments:
        program = track.program
        for note in track.notes:
            sampler.add_midi_note(
                pitch=note.pitch,
                velocity=note.velocity,
                program=program,
                start=note.start,
                dur=note.end - note.start,
            )
    print(len(sampler.notes))
    audio = sampler.render(time=60.0)
    sf.write("test_render.wav", audio, sampler.sr, format="WAV")