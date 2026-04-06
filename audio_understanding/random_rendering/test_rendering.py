import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import dawdreamer as daw
import numpy as np
import pretty_midi
from scipy.io import wavfile


DEFAULT_SAMPLE_RATE = 48000
DEFAULT_BUFFER_SIZE = 128


def sanitize_name(value: str) -> str:
    """Make a filesystem-safe directory name."""
    cleaned = re.sub(r"[<>:\"/\\|?*]", "_", value.strip())
    cleaned = cleaned.replace(" ", "_")
    return cleaned or "unnamed"


def standard_midi_program_name(program_id: int) -> str:
    """Map standard MIDI program id to exact display name. 128 is reserved for drums."""
    if program_id == 128:
        return "Drum Kit"
    if not 0 <= program_id <= 127:
        raise ValueError("program_id must be in [0, 128], where 128 means drums")
    return pretty_midi.program_to_instrument_name(program_id)


def save_audio_mp3(audio: np.ndarray, sample_rate: int, out_mp3_path: Path) -> str:
    """
    Save output as MP3.
    Try soundfile first; fallback to pydub + ffmpeg.
    """
    stereo = audio.T.astype(np.float32)

    try:
        import soundfile as sf

        sf.write(str(out_mp3_path), stereo, sample_rate, format="MP3")
        return "soundfile"
    except Exception as soundfile_error:
        try:
            from pydub import AudioSegment

            tmp_wav = out_mp3_path.with_suffix(".tmp.wav")
            # Keep float32 path, no int16 conversion.
            wavfile.write(str(tmp_wav), sample_rate, stereo)
            AudioSegment.from_wav(str(tmp_wav)).export(
                str(out_mp3_path), format="mp3", bitrate="192k"
            )
            if tmp_wav.exists():
                tmp_wav.unlink()
            return "pydub"
        except Exception as pydub_error:
            raise RuntimeError(
                "Cannot export MP3. Install one of:"
                " (1) soundfile with MP3 support, or"
                " (2) pydub + ffmpeg in PATH."
                f" soundfile error={soundfile_error}; pydub error={pydub_error}"
            )


def build_engine_and_synth(vst_path: str, sample_rate: int, buffer_size: int):
    """Build engine, suppressing the harmless URI warning JUCE emits for paths with spaces."""
    import os

    engine = daw.RenderEngine(sample_rate, buffer_size)

    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    try:
        synth = engine.make_plugin_processor("synth", vst_path)
    finally:
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)

    engine.load_graph([(synth, [])])
    return engine, synth


def render_single_note_scale(vst_path: str, sample_rate: int, buffer_size: int) -> np.ndarray:
    """Render a simple ascending single-note scale."""
    engine, synth = build_engine_and_synth(vst_path, sample_rate, buffer_size)

    notes = range(60, 84)
    start = 0.0
    step = 0.10
    dur = 0.09
    velocity = 100

    for note in notes:
        synth.add_midi_note(note, velocity, start, dur)
        start += step

    total_seconds = start + dur
    engine.render(total_seconds)
    audio = engine.get_audio()

    return audio

def render_consecutive(vst_path: str, sample_rate: int, buffer_size: int) -> np.ndarray:
    """Render a simple ascending single-note scale."""
    engine, synth = build_engine_and_synth(vst_path, sample_rate, buffer_size)

    notes = [60]*10
    start = 0.0
    step = 0.10
    dur = 0.10
    velocity = 100

    for note in notes:
        synth.add_midi_note(note, velocity, start, dur)
        start += step

    total_seconds = start + dur
    engine.render(total_seconds)
    return engine.get_audio()
def render_continuous_chords(vst_path: str, sample_rate: int, buffer_size: int) -> np.ndarray:
    """Render a continuous chord progression."""
    engine, synth = build_engine_and_synth(vst_path, sample_rate, buffer_size)

    chords = [
        [60, 64, 67],  # C
        [65, 69, 72],  # F
        [67, 71, 74],  # G
        [60, 64, 67],  # C
    ]
    chord_len = 1.0
    velocity = 100

    for i, chord in enumerate(chords):
        start = i * chord_len
        for note in chord:
            synth.add_midi_note(note, velocity, start, chord_len)

    total_seconds = len(chords) * chord_len + 1.0
    engine.render(total_seconds)
    return engine.get_audio()


def render_fugue_polyphony(
    vst_path: str,
    midi_path: Path,
    sample_rate: int,
    buffer_size: int,
    start_sec: float = 15.0,
    end_sec: float = 25.0,
) -> np.ndarray:
    """Render polyphony from MIDI by rendering each track separately and summing."""
    if end_sec <= start_sec:
        raise ValueError("end_sec must be greater than start_sec")

    midi_obj = pretty_midi.PrettyMIDI(str(midi_path))
    window_dur = end_sec - start_sec
    rendered_tracks = []

    for instrument in midi_obj.instruments:
        if not instrument.notes:
            continue

        engine, synth = build_engine_and_synth(vst_path, sample_rate, buffer_size)

        for note in instrument.notes:
            note_start = max(float(note.start), start_sec)
            note_end = min(float(note.end), end_sec)
            if note_end <= note_start:
                continue

            synth.add_midi_note(
                int(note.pitch),
                int(note.velocity),
                note_start - start_sec,
                note_end - note_start,
            )

        engine.render(window_dur)
        rendered_tracks.append(engine.get_audio().astype(np.float32))

    if not rendered_tracks:
        n_samples = int(window_dur * sample_rate)
        return np.zeros((2, n_samples), dtype=np.float32)
    print(f"Rendered {len(rendered_tracks)} tracks for fugue polyphony.")
    max_len = max(track.shape[1] for track in rendered_tracks)
    mix = np.zeros((2, max_len), dtype=np.float32)
    for track in rendered_tracks:
        mix[:, : track.shape[1]] += track

    # Prevent clipping after summing multiple tracks.
    mix = np.clip(mix, -1.0, 1.0)
    return mix


def append_global_index(global_json_path: Path, entry: dict) -> None:
    if global_json_path.exists():
        try:
            data = json.loads(global_json_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    else:
        data = []

    data.append(entry)
    global_json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_render_pipeline(
    vst_path: str,
    program_id: int,
    midi_path: Path,
    output_root: Path,
    global_json_path: Path,
    sample_rate: int,
    buffer_size: int,
) -> None:
    program_name = f"{program_id}_" + standard_midi_program_name(program_id)
    program_dir = sanitize_name(program_name)
    vst_name = sanitize_name(Path(vst_path).name)
    out_dir = output_root / program_dir / vst_name
    out_dir.mkdir(parents=True, exist_ok=True)

    now_iso = datetime.now().isoformat(timespec="seconds")

    consecutive = render_consecutive(vst_path, sample_rate, buffer_size)
    consecutive_mp3 = out_dir / "consecutive.mp3"
    consecutive_backend = save_audio_mp3(consecutive, sample_rate, consecutive_mp3)
    
    single_audio = render_single_note_scale(vst_path, sample_rate, buffer_size)
    single_mp3 = out_dir / "single_note_scale.mp3"
    single_backend = save_audio_mp3(single_audio, sample_rate, single_mp3)

    chords_audio = render_continuous_chords(vst_path, sample_rate, buffer_size)
    chords_mp3 = out_dir / "continuous_chords.mp3"
    chords_backend = save_audio_mp3(chords_audio, sample_rate, chords_mp3)

    fugue_audio = render_fugue_polyphony(vst_path, midi_path, sample_rate, buffer_size)
    fugue_mp3 = out_dir / "fugue_polyphony.mp3"
    fugue_backend = save_audio_mp3(fugue_audio, sample_rate, fugue_mp3)

    midi_obj = pretty_midi.PrettyMIDI(str(midi_path))
    track_programs = []
    for idx, inst in enumerate(midi_obj.instruments):
        pid = 128 if inst.is_drum else int(inst.program)
        track_programs.append(
            {
                "track_index": idx,
                "program_id": pid,
                "program_name": standard_midi_program_name(pid),
                "is_drum": bool(inst.is_drum),
                "note_count": len(inst.notes),
            }
        )

    metadata = {
        "timestamp": now_iso,
        "program_id": program_id,
        "program_name": program_name,
        "vst_path": str(Path(vst_path)),
        "vst_file_name": Path(vst_path).name,
        "sample_rate": sample_rate,
        "buffer_size": buffer_size,
        "midi_path": str(midi_path),
        "midi_track_programs": track_programs,
        "fugue_window_seconds": {
            "start": 15.0,
            "end": 25.0,
        },
        "outputs": [
            {
                "name": "single_note_scale",
                "file": str(single_mp3),
                "export_backend": single_backend,
            },
            {
                "name": "continuous_chords",
                "file": str(chords_mp3),
                "export_backend": chords_backend,
            },
            {
                "name": "fugue_polyphony",
                "file": str(fugue_mp3),
                "export_backend": fugue_backend,
            },
            {
                "name": "consecutive",
                "file": str(consecutive_mp3),
                "export_backend": consecutive_backend,
            },
        ],
    }

    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # User requested global json fields including program name and vst path.
    global_entry = {
        "timestamp": now_iso,
        "program_id": program_id,
        "program_name": program_name,
        "vst_path": str(Path(vst_path)),
        "program_name_duplicate": program_name,
        "output_dir": str(out_dir),
        "metadata_file": str(metadata_path),
    }
    append_global_index(global_json_path, global_entry)

    print("Render done.")
    print(f"Output dir: {out_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Global json: {global_json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render 3 demo MIDI scenes with a VST using DawDreamer."
    )
    parser.add_argument("--vst-path", required=True, help="Path to .dll/.vst/.vst3 plugin")
    parser.add_argument(
        "--program-id",
        type=int,
        default=0,
        help="Standard MIDI program id: 0-127 for instruments, 128 for drums",
    )
    parser.add_argument(
        "--midi-path",
        default='/data/yrb/musicarena/Haiwen/Autoregressive-Transcription/Fugue1.mid',
        help="Path to polyphonic MIDI file (default: rendering/Fugue1.mid)",
    )
    parser.add_argument(
        "--output-root",
        default=".",
        help="Root output folder. Final folder is {program}/{VST_Path.name}/",
    )
    parser.add_argument(
        "--global-json",
        default=str(Path(__file__).with_name("global_render_index.json")),
        help="Global json index file path",
    )
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    midi_path = Path(args.midi_path)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    run_render_pipeline(
        vst_path=args.vst_path,
        program_id=args.program_id,
        midi_path=midi_path,
        output_root=Path(args.output_root),
        global_json_path=Path(args.global_json),
        sample_rate=args.sample_rate,
        buffer_size=args.buffer_size,
    )


if __name__ == "__main__":
    main()
#! error: attempt to map invalid URI 
#! Multi-core: could not acquire real-time scheduling, error 1 -- Operation not permitted
#! error: attempt to map invalid URI '/data/yrb/musicarena/Haiwen/Autoregressive-Transcription/Rendering_source/pianoteq_trial_v912/Pianoteq 9/x86-64bit/Pianoteq 9.vst3'