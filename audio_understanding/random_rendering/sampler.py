#没必要兼容那些连续分布，可以去掉了(保持p即可) ； 同时，_apply_no_overlap最好的duration采样

import numpy as np 
import random 
import json
from collections import defaultdict
from typing import List, Dict, Any

def _apply_fifo(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per (program, pitch) channel: eliminate containment while allowing overlap.

    Sort onsets and offsets independently, pair i-th onset with i-th offset (FIFO).
    """
    channels: dict = defaultdict(list)
    for n in notes:
        channels[(n["program"], n["pitch"])].append(n)

    result = []
    for ch_notes in channels.values():
        ch_notes_sorted = sorted(ch_notes, key=lambda x: x["start"])
        onsets  = sorted(n["start"] for n in ch_notes)
        offsets = sorted(n["start"] + n["dur"] for n in ch_notes)
        for note, onset, offset in zip(ch_notes_sorted, onsets, offsets):
            dur = round((offset - onset) / 0.01) * 0.01
            dur = max(0.01, dur)
            result.append({**note, "start": onset, "dur": dur})
    return result


def _apply_no_overlap(
    notes: List[Dict[str, Any]],
    time: float,
    dur_lo: float = 0.01,
    dur_hi: float = None,
) -> List[Dict[str, Any]]:
    """Per (program, pitch) channel: re-sample start+dur so no two notes overlap.

    Keeps pitch/program/velocity from input notes; discards original start/dur.
    Per channel:
      1. Sample n onsets uniformly in [0, time], sort them.
      2. For note i, sample dur in [dur_lo, min(dur_hi, gap_to_next_onset)].
    """
    if dur_hi is None:
        dur_hi = time

    channels: dict = defaultdict(list)
    for n in notes:
        channels[(n["program"], n["pitch"])].append(n)

    result = []
    for ch_notes in channels.values():
        n = len(ch_notes)
        onsets = np.sort(np.random.uniform(0, time, size=n))
        onsets = np.round(onsets / 0.01) * 0.01
        for j, note in enumerate(ch_notes):
            onset = float(onsets[j])
            max_dur = float(onsets[j + 1]) - onset if j + 1 < n else time - onset
            max_dur = max(max_dur, 0.01)
            hi = max(min(max_dur, dur_hi), dur_lo)
            dur = round(float(np.random.uniform(dur_lo, hi)) / 0.01) * 0.01
            dur = max(0.01, dur)
            result.append({**note, "start": onset, "dur": dur})
    return result


class BaseSampler: 
    """
    返回随机采样的音符用于渲染。
    """
    def __init__(self, time: float):
        self.time = time
        
    def sample(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
'''
1. 创建一个简单的随机采样方案：
program: 总是采样 0
pitch: 在 [21,108] 之间均匀分布
velocity: 总是采样 100
duration: 在 [0.1, 5.0] 之间均匀分布

'''
class MarginalRandomSampler(BaseSampler):
    """
    独立采样每个音符，不考虑联合分布。
    """
    
    def normalize_probs(self, distri):
        if "probs" not in distri:
            print("Warning: 分布缺少 'probs' 键，默认使用均匀分布")
            distri["probs"] = [1/len(distri["values"])] * len(distri["values"])
        total = sum(distri["probs"])
        if total > 0:
            distri["probs"] = [p / total for p in distri["probs"]]
        else:
            raise ValueError("概率总和必须大于0")
        # pre-convert to numpy arrays to avoid per-call list→array overhead
        distri["values"] = np.asarray(distri["values"])
        distri["probs"] = np.asarray(distri["probs"], dtype=np.float64)
        return distri
    def __init__(self, time: float, config_json: str, deduplication: str = "disabled"):
        super().__init__(time)
        assert deduplication in ("disabled", "FIFO", "no_overlap"), \
            f"deduplication must be 'disabled', 'FIFO', or 'no_overlap', got '{deduplication}'"
        self.deduplication = deduplication
        try:
            with open(config_json, "r", encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件 {config_json} 未找到")
        except json.JSONDecodeError:
            raise ValueError(f"配置文件 {config_json} 格式错误")
        
        # 检查必需的配置键
        required_keys = ["program", "pitch", "dur", "velocity", "num_notes"]
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"配置文件缺少键: {key}")
        
        self.program_distri = self.normalize_probs(self.config["program"])
        self.pitch_distri = self.normalize_probs(self.config["pitch"])
        self.dur_distri = self.normalize_probs(self.config["dur"]) #* might contain "type" for continuous distribution
        self.vel_distri = self.normalize_probs(self.config["velocity"])
        self.num_distri = self.normalize_probs(self.config["num_notes"])

        # dur bounds for _apply_no_overlap
        dur_cfg = self.config["dur"]
        if "type" in dur_cfg:
            self.dur_lo = float(dur_cfg.get("min", 0.01))
            self.dur_hi = float(dur_cfg.get("max", time))
        else:
            vals = dur_cfg["values"]
            self.dur_lo = max(0.01, float(min(vals)))
            self.dur_hi = float(max(vals))
        
    def sample(self, seed) -> List[Dict[str, Any]]:
        if seed is not None:
            super().sample(seed)
        notes = []
        number = int(np.random.choice(self.num_distri["values"], p=self.num_distri["probs"]))
        if number == 0:
            return notes

        # batch sample all attributes at once
        pitches = np.random.choice(self.pitch_distri["values"], size=number, p=self.pitch_distri["probs"])
        programs = np.random.choice(self.program_distri["values"], size=number, p=self.program_distri["probs"])
        vels = np.random.choice(self.vel_distri["values"], size=number, p=self.vel_distri["probs"])

        # 采样持续时间，支持连续或离散分布
        if "type" in self.dur_distri:
            if self.dur_distri["type"] == "uniform":
                durs = np.random.uniform(self.dur_distri["min"], self.dur_distri["max"], size=number)
            if self.dur_distri["type"] == "normal":
                durs = np.random.normal(self.dur_distri["mean"], self.dur_distri["std"], size=number)
        else:
            durs = np.random.choice(self.dur_distri["values"], size=number, p=self.dur_distri["probs"])

        # 确保持续时间合理
        durs = np.clip(durs, 0.01, self.time)
        # 离散到0.01秒分辨率
        durs = np.round(durs / 0.01) * 0.01

        # 采样起始时间
        max_starts = np.maximum(0, self.time - durs)
        starts = np.random.uniform(0, max_starts)
        starts = np.round(starts / 0.01) * 0.01

        for i in range(number):
            notes.append({"pitch": int(pitches[i]), "dur": float(durs[i]),
                          "velocity": int(vels[i]), "start": float(starts[i]),
                          "program": int(programs[i])})

        if self.deduplication == "FIFO":
            notes = _apply_fifo(notes)
        elif self.deduplication == "no_overlap":
            notes = _apply_no_overlap(notes, self.time, self.dur_lo, self.dur_hi)
        return notes


class UniformSampler(BaseSampler):
    """
    Uniform sampler: for keys *not* in keep_marginal_keys, replaces the config
    distribution with a uniform distribution bounded by [min(values), max(values)].
    For keys in keep_marginal_keys, keeps the original marginal distribution.

    keep_marginal_keys: subset of ["pitch", "dur", "velocity", "program"].
    """
    def __init__(self, time: float, config_json: str, keep_marginal_keys: List[str] = None,
                 deduplication: str = "disabled"):
        super().__init__(time)
        assert deduplication in ("disabled", "FIFO", "no_overlap"), \
            f"deduplication must be 'disabled', 'FIFO', or 'no_overlap', got '{deduplication}'"
        self.deduplication = deduplication
        with open(config_json, "r", encoding='utf-8') as f:
            self.config = json.load(f)

        required_keys = ["program", "pitch", "dur", "velocity", "num_notes"]
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"配置文件缺少键: {key}")

        keep = set(keep_marginal_keys or [])
        assert keep <= {"num_notes", "pitch", "dur", "velocity", "program"}, \
            f"keep_marginal_keys must be a subset of num_notes/pitch/dur/velocity/program, got {keep}"

        # Build final distribution dicts: marginal for kept keys, uniform-bounded for the rest.
        # num_notes always uses uniform-over-bounds.
        self.distris: Dict[str, dict] = {}
        for key in required_keys:
            raw = self.config[key]
            if key in keep:
                self.distris[key] = self._prepare_marginal(raw)
            else:
                lo, hi = self._bounds(raw)
                if key in ("pitch", "program", "velocity","num_notes"):
                    self.distris[key] = {"type": "uniform_int", "min": lo, "max": hi}
                else:  # dur
                    self.distris[key] = {"type": "uniform", "min": lo, "max": hi}

        self.dur_lo = max(0.01, float(self._bounds(self.config["dur"])[0]))
        self.dur_hi = float(self._bounds(self.config["dur"])[1])

    @staticmethod
    def _bounds(distri: dict):
        """Return (lo, hi) for a distribution dict."""
        if "type" in distri:
            if distri["type"] in ("uniform", "normal"):
                return distri.get("min", 0), distri.get("max", 1)
            raise ValueError(f"Unsupported distribution type: {distri['type']}")
        vals = distri["values"]
        return min(vals), max(vals)

    @staticmethod
    def _prepare_marginal(distri: dict) -> dict:
        """Prepare a distribution dict for marginal sampling."""
        distri = dict(distri)  # shallow copy
        if "type" in distri:
            return distri
        if "probs" not in distri:
            distri["probs"] = [1.0 / len(distri["values"])] * len(distri["values"])
        total = sum(distri["probs"])
        distri["probs"] = [p / total for p in distri["probs"]]
        distri["values"] = np.asarray(distri["values"])
        distri["probs"] = np.asarray(distri["probs"], dtype=np.float64)
        return distri

    @staticmethod
    def _sample_marginal(distri: dict, size: int) -> np.ndarray:
        """Sample `size` values from a distribution dict."""
        t = distri.get("type")
        if t == "uniform":
            return np.random.uniform(distri["min"], distri["max"], size=size)
        if t == "uniform_int":
            return np.random.uniform(distri["min"], distri["max"] + 1, size=size).astype(int)
        if t == "normal":
            return np.random.normal(distri["mean"], distri["std"], size=size)
        return np.random.choice(distri["values"], size=size, p=distri["probs"])

    def sample(self, seed: int = None) -> List[Dict[str, Any]]:
        if seed is not None:
            super().sample(seed)

        number = int(self._sample_marginal(self.distris["num_notes"], 1)[0])
        number = max(0, number)
        if number == 0:
            return []

        pitches  = self._sample_marginal(self.distris["pitch"],    number).astype(int)
        programs = self._sample_marginal(self.distris["program"],  number).astype(int)
        vels     = self._sample_marginal(self.distris["velocity"], number).astype(int)
        durs     = self._sample_marginal(self.distris["dur"],      number).astype(float)

        durs = np.clip(durs, 0.01, self.time)
        durs = np.round(durs / 0.01) * 0.01

        max_starts = np.maximum(0, self.time - durs)
        starts = np.round(np.random.uniform(0, max_starts) / 0.01) * 0.01

        notes = []
        for i in range(number):
            notes.append({
                "pitch":    int(np.clip(pitches[i],  0, 127)),
                "dur":      float(durs[i]),
                "velocity": int(np.clip(vels[i],     0, 127)),
                "start":    float(starts[i]),
                "program":  int(programs[i]),
            })

        if self.deduplication == "FIFO":
            notes = _apply_fifo(notes)
        elif self.deduplication == "no_overlap":
            notes = _apply_no_overlap(notes, self.time, self.dur_lo, self.dur_hi)
        return notes


if __name__ == "__main__":
    import os
#* todo: 一个特别随机的version
    config_path = r"D:\local\codes\dsp\rendering\piano_stats_config.json"
    # if not os.path.exists(config_path):
    #     # 生成示例配置文件
    #     example_config = {
    #         "program": {"values": [0], "probs": [1.0]},
    #         "pitch": {"values": list(range(21, 109)), "probs": [1/88] * 88},
    #         "dur": {"type": "uniform", "min": 0.1, "max": 5.0},  # 连续均匀分布
    #         "velocity": {"values": [100], "probs": [1.0]},
    #         "num_notes": {"values": [10, 2, 3, 4, 5], "probs": [0.1, 0.2, 0.4, 0.2, 0.1]}
    #     }
    #     with open(config_path, "w", encoding='utf-8') as f:
    #         json.dump(example_config, f, indent=4)
    #     print(f"已生成示例配置文件: {config_path}")

    sampler = MarginalRandomSampler(time=5.0, config_json=config_path)

    notes = sampler.sample(seed=42)
    from .rendering import Sampler
    render_sampler = Sampler(time=5.0, sr=16000,data_dir=r'D:\local\codes\dsp\Rendered')
    render_sampler.set_midi_note(notes)
    audio = render_sampler.render()
    import soundfile as sf
    sf.write(r"D:\local\codes\dsp\rendering\sampled_output.wav", audio, render_sampler.sr, format="WAV")