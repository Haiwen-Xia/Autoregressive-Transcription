"""
music_synth_gp.py

单音符音乐合成器，结合:
- formula_sed 的 GP (Exponential 核) 时域幅度调制
- synth_correct 风格的加性合成
- 可配置泛音序列 + 频率抖动

设计原则:
- generate_music_note 接受确定性参数（不含范围），不做随机采样
- 随机化由 sample_params / 预设函数完成，返回参数 dict
- 用 cumsum 累积相位实现时变频率合成（参考 formula_sed/ddsp/core.py）
"""

import numpy as np
import GPy
from functools import lru_cache


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def midi_to_hz(pitch: float) -> float:
    return 440.0 * (2.0 ** ((pitch - 69.0) / 12.0))


def _q(value: float, ndigits: int = 3) -> float:
    """Quantize float hyper-params for better cache hit rate."""
    return round(float(value), ndigits)


@lru_cache(maxsize=256)
def _cached_exp_cholesky(n_frames: int, variance: float, lengthscale: float) -> np.ndarray:
    """Cache Cholesky factor for Exponential kernel covariance."""
    x = np.linspace(0.0, 1.0, int(n_frames), dtype=np.float64)[:, None]
    kernel = GPy.kern.Exponential(
        input_dim=1,
        variance=float(variance),
        lengthscale=float(lengthscale),
    )
    K = kernel.K(x) + 1e-6 * np.eye(int(n_frames), dtype=np.float64)
    return np.linalg.cholesky(K)


@lru_cache(maxsize=256)
def _cached_periodic_cholesky(
    n_frames: int,
    variance: float,
    period: float,
    lengthscale: float,
) -> np.ndarray:
    """Cache Cholesky factor for StdPeriodic kernel covariance."""
    x = np.linspace(0.0, 1.0, int(n_frames), dtype=np.float64)[:, None]
    kernel = GPy.kern.StdPeriodic(
        input_dim=1,
        variance=float(variance),
        period=float(period),
        lengthscale=float(lengthscale),
    )
    K = kernel.K(x) + 1e-6 * np.eye(int(n_frames), dtype=np.float64)
    return np.linalg.cholesky(K)


@lru_cache(maxsize=256)
def _cached_rbf_cholesky(n: int, variance: float, lengthscale: float) -> np.ndarray:
    """Cache Cholesky factor for harmonic-index RBF covariance."""
    kk = np.arange(1, int(n) + 1, dtype=np.float64)
    x = (kk / max(int(n), 1))[:, None]
    kernel = GPy.kern.RBF(
        input_dim=1,
        variance=float(variance),
        lengthscale=float(lengthscale),
    )
    K = kernel.K(x) + 1e-6 * np.eye(int(n), dtype=np.float64)
    return np.linalg.cholesky(K)


def sample_overtone_series(
    rng: np.random.Generator,
    n_overtones: int,
    series_mode: str = "harmonic",
    custom_ratios: list | None = None,
    stretch_beta: float = 1.0,
    jitter_cents: float = 5.0,
) -> np.ndarray:
    """
    生成泛音序列（相对 f0 的频率倍数）。

    series_mode:
        "harmonic"  : 整数倍泛音 [1, 2, 3, ..., n_overtones]
        "stretched" : 拉伸泛音 k^beta，用于模拟钢琴非谐性
        "custom"    : 从 custom_ratios 中取前 n_overtones 个
    """
    if series_mode == "harmonic":
        k = np.arange(1, n_overtones + 1, dtype=np.float64)
        ratios = k
    elif series_mode == "stretched":
        k = np.arange(1, n_overtones + 1, dtype=np.float64)
        ratios = k ** stretch_beta
    elif series_mode == "custom":
        if custom_ratios is None:
            raise ValueError("series_mode='custom' 时需要提供 custom_ratios")
        ratios = np.array(custom_ratios[:n_overtones], dtype=np.float64)
    else:
        raise ValueError(f"未知的 series_mode: {series_mode}")

    # 频率抖动：freq_jittered = freq * 2^(cents/1200)
    jitter_semitones = rng.standard_normal(len(ratios)) * jitter_cents / 1200.0
    ratios_jittered = ratios * (2.0 ** jitter_semitones)
    return ratios_jittered





def compute_base_amps(
    rng: np.random.Generator,
    nominal_freqs: np.ndarray,
    nyquist: float,
    rolloff: float,
    amp_noise_mode: str = "none",
    amp_noise_std: float = 0.1,
    amp_noise_rbf_lengthscale: float = 0.3,
    odd_even_ratio: float = 1.0,
) -> np.ndarray:
    """
    计算各泛音的初始幅度 (静态)。

    基础形状: 多项式衰减 a[k] = 1 / k^rolloff  (保证 a[0]=1 最大)
    频率相关的衰减由时域 per-harmonic exp_decay 处理，此处不再做。

    amp_noise_mode:
        "none"     : 无噪声
        "gaussian" : 独立高斯 a[k] *= (1 + noise), noise ~ N(0, std)
        "rbf"      : RBF 核相关噪声 a[k] *= (1 + noise), noise ~ GP(0, K_rbf)

    odd_even_ratio:
        偶数次泛音 (k=2,4,6...) 的幅度缩放因子。
        1.0 = 不区分奇偶; 0.0 = 只保留奇数泛音 (clarinet 风格)
    """
    n = len(nominal_freqs)
    k = np.arange(1, n + 1, dtype=np.float64)

    # 多项式衰减: a[k] = 1/k^rolloff
    amps = 1.0 / (k ** rolloff)

    # 奇偶比: 偶数次泛音 (index 1,3,5... 对应 k=2,4,6) 乘以 odd_even_ratio
    even_mask = (k % 2 == 0)
    amps[even_mask] *= odd_even_ratio

    # 乘性噪声
    if amp_noise_mode == "gaussian":
        noise = rng.standard_normal(n) * amp_noise_std
        amps *= np.maximum(1.0 + noise, 0.01)  # 防止负数
    elif amp_noise_mode == "rbf":
        # RBF 核：泛音序号空间上的相关噪声（分解缓存后每次采样只做矩阵向量乘）
        L = _cached_rbf_cholesky(
            n,
            _q(amp_noise_std ** 2),
            _q(amp_noise_rbf_lengthscale),
        )
        noise = L @ rng.standard_normal(n)
        amps *= np.maximum(1.0 + noise, 0.01)

    # 约束: a[k] <= a[0]
    amps = np.minimum(amps, amps[0])

    return amps


def precompute_gp_cholesky(
    n_frames: int,
    gp_variance: float,
    gp_lengthscale: float,
    vibrato_variance: float = 1.0,
    vibrato_period: float = 0.8,
    vibrato_lengthscale: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """每个 clip 调用一次，预计算两个 GP 核的 Cholesky 分解。

    Returns:
        L_gp_exp:    (n_frames, n_frames) Exponential 核的下三角 Cholesky
        L_gp_vibrato:(n_frames, n_frames) StdPeriodic 核的下三角 Cholesky

    将返回值传入 generate_music_note(L_gp_exp=..., L_gp_vibrato=...)，
    同一 clip 内的所有 note 共用，避免重复 O(n^3) 分解。
    """
    L_gp_exp = _cached_exp_cholesky(n_frames, _q(gp_variance), _q(gp_lengthscale))
    L_gp_vibrato = _cached_periodic_cholesky(
        n_frames, _q(vibrato_variance), _q(vibrato_period), _q(vibrato_lengthscale)
    )
    return L_gp_exp, L_gp_vibrato


def sample_gp_exponential(
    rng: np.random.Generator,
    n_frames: int,
    variance: float = 1.0,
    lengthscale: float = 0.3,
    *,
    L: np.ndarray | None = None,  # 预计算的 Cholesky；传入则跳过 kernel 构建
) -> np.ndarray:
    """
    使用 Exponential (Matern-1/2) 核采样 GP 轨迹。
    返回经 softplus 映射后的正值包络。
    """
    if L is None:
        L = _cached_exp_cholesky(n_frames, _q(variance), _q(lengthscale))
    raw = L @ rng.standard_normal(n_frames)
    return np.log1p(np.exp(raw))


def sample_f0_trajectory(
    rng: np.random.Generator,
    f0: float,
    n_frames: int,
    f0_mode: str = "constant",
    vibrato_variance: float = 1.0,
    vibrato_lengthscale: float = 0.15,
    vibrato_period: float = 0.8,
    vibrato_depth_cents: float = 30.0,
    *,
    L_vibrato: np.ndarray | None = None,  # 预计算的 Cholesky；传入则跳过 kernel 构建
) -> np.ndarray:
    """
    生成时变 f0 轨迹（帧级），单位 Hz。

    f0_mode:
        "constant" : 恒定 f0
        "vibrato"  : StdPeriodic 核采样周期性偏移（模拟弦乐器揉弦）
    """
    if f0_mode == "constant":
        return np.full(n_frames, f0, dtype=np.float64)
    elif f0_mode == "vibrato":
        if L_vibrato is None:
            L_vibrato = _cached_periodic_cholesky(
                n_frames,
                _q(vibrato_variance),
                _q(vibrato_period),
                _q(vibrato_lengthscale),
            )
        raw = L_vibrato @ rng.standard_normal(n_frames)
        raw_norm = raw / (np.abs(raw).max() + 1e-9)
        cents_offset = raw_norm * vibrato_depth_cents
        return f0 * (2.0 ** (cents_offset / 1200.0))
    else:
        raise ValueError(f"未知的 f0_mode: {f0_mode}")


# ---------------------------------------------------------------------------
# 参数采样：从范围中随机采样出确定性参数
# ---------------------------------------------------------------------------

def sample_params(
    rng: np.random.Generator,
    # 泛音
    n_overtones: int = 20,
    series_mode: str = "harmonic",
    custom_ratios: list | None = None,
    stretch_beta: float = 1.0,
    jitter_cents: float = 5.0,
    # 幅度
    amp_mode: str = "exp_decay",
    rolloff_range: tuple = (1.2, 2.0),
    amp_noise_mode: str = "none",
    amp_noise_std_range: tuple = (0.05, 0.2),
    amp_noise_rbf_lengthscale_range: tuple = (0.2, 0.5),
    odd_even_ratio_range: tuple = (1.0, 1.0),
    # exp_decay 时域衰减
    decay_base_range: tuple = (1.2, 2.8),
    decay_highfreq_range: tuple = (10.0, 500.0),
    decay_shape_range: tuple = (1.2, 2.0),
    # GP
    n_frames: int = 100,
    gp_variance_range: tuple = (0.5, 2.0),
    gp_lengthscale_range: tuple = (0.15, 0.5),
    gp_modulation_depth_range: tuple = (0.05, 0.2),
    # f0
    f0_mode: str = "constant",
    vibrato_depth_cents_range: tuple = (15.0, 50.0),
    vibrato_period_range: tuple = (0.1, 0.3),
    vibrato_lengthscale_range: tuple = (0.05, 0.2),
    vibrato_variance_range: tuple = (0.5, 2.0),
    # 包络
    attack_time_range: tuple = (0.002, 0.02),
    release_time_range: tuple = (0.05, 0.3),
) -> dict:
    """从范围中随机采样，返回 generate_music_note 需要的确定性参数 dict。"""
    return {
        # 泛音
        "n_overtones": n_overtones,
        "series_mode": series_mode,
        "custom_ratios": custom_ratios,
        "stretch_beta": stretch_beta,
        "jitter_cents": jitter_cents,
        # 幅度
        "amp_mode": amp_mode,
        "rolloff": rng.uniform(*rolloff_range),
        "amp_noise_mode": amp_noise_mode,
        "amp_noise_std": rng.uniform(*amp_noise_std_range),
        "amp_noise_rbf_lengthscale": rng.uniform(*amp_noise_rbf_lengthscale_range),
        "odd_even_ratio": rng.uniform(*odd_even_ratio_range),
        # exp_decay 时域衰减
        "decay_base": rng.uniform(*decay_base_range),
        "decay_highfreq": rng.uniform(*decay_highfreq_range),
        "decay_shape": rng.uniform(*decay_shape_range),
        # GP
        "n_frames": n_frames,
        "gp_variance": rng.uniform(*gp_variance_range),
        "gp_lengthscale": rng.uniform(*gp_lengthscale_range),
        "gp_modulation_depth": rng.uniform(*gp_modulation_depth_range),
        # f0
        "f0_mode": f0_mode,
        "vibrato_depth_cents": rng.uniform(*vibrato_depth_cents_range),
        "vibrato_period": rng.uniform(*vibrato_period_range),
        "vibrato_lengthscale": rng.uniform(*vibrato_lengthscale_range),
        "vibrato_variance": rng.uniform(*vibrato_variance_range),
        # 包络
        "attack_time": rng.uniform(*attack_time_range),
        "release_time": rng.uniform(*release_time_range),
    }


def piano_params(rng: np.random.Generator) -> dict:
    """钢琴风格预设：大衰减率、固定 f0、拉伸泛音。"""
    return sample_params(
        rng,
        n_overtones=30,
        series_mode="stretched",
        stretch_beta=1.0 + rng.uniform(0.001, 0.005),
        jitter_cents=3.0,
        amp_mode="exp_decay",
        # 钢琴衰减大得多：基频衰减率 ~2-6/s，高频额外衰减很大
        decay_base_range=(2.0, 6.0),
        decay_highfreq_range=(200.0, 2000.0),
        decay_shape_range=(1.0, 1.8),
        gp_modulation_depth_range=(0.03, 0.10),
        gp_lengthscale_range=(0.2, 0.5),
        f0_mode="constant",
        attack_time_range=(0.002, 0.008),
        release_time_range=(0.05, 0.15),
    )


def string_params(rng: np.random.Generator) -> dict:
    """弦乐器风格预设：GP 主包络、vibrato f0。"""
    return sample_params(
        rng,
        n_overtones=20,
        series_mode="harmonic",
        jitter_cents=4.0,
        amp_mode="constant",
        rolloff_range=(1.2, 2.0),
        n_frames=120,
        gp_variance_range=(1.0, 2.5),
        gp_lengthscale_range=(0.3, 0.6),
        f0_mode="vibrato",
        vibrato_depth_cents_range=(20.0, 50.0),
        vibrato_period_range=(0.10, 0.25),
        vibrato_lengthscale_range=(0.05, 0.15),
        attack_time_range=(0.01, 0.05),
        release_time_range=(0.1, 0.4),
    )


def bell_params(rng: np.random.Generator, custom_ratios: list | None = None) -> dict:
    """钟声风格预设：自定义非整数泛音比、慢衰减。"""
    if custom_ratios is None:
        custom_ratios = [1.0, 1.5, 2.0, 2.67, 3.2, 4.0, 5.33, 6.0, 6.67]
    return sample_params(
        rng,
        n_overtones=len(custom_ratios),
        series_mode="custom",
        custom_ratios=custom_ratios,
        jitter_cents=8.0,
        amp_mode="exp_decay",
        decay_base_range=(0.3, 1.0),
        decay_highfreq_range=(5.0, 50.0),
        decay_shape_range=(1.0, 1.5),
        gp_modulation_depth_range=(0.02, 0.08),
        gp_lengthscale_range=(0.3, 0.7),
        f0_mode="constant",
        attack_time_range=(0.001, 0.005),
        release_time_range=(0.3, 0.8),
    )


def clarinet_params(rng: np.random.Generator) -> dict:
    """
    单簧管风格预设：
    - 奇数泛音为主 (odd_even_ratio ≈ 0.0~0.15)
    - GP 主包络 (constant 模式，持续音)
    - 轻微 vibrato
    - RBF 相关噪声让泛音幅度更自然
    """
    return sample_params(
        rng,
        n_overtones=20,
        series_mode="harmonic",
        jitter_cents=2.0,
        amp_mode="constant",
        rolloff_range=(0.8, 1.2),
        amp_noise_mode="rbf",
        amp_noise_std_range=(0.08, 0.15),
        amp_noise_rbf_lengthscale_range=(0.3, 0.6),
        odd_even_ratio_range=(0.0, 0.15),   # 几乎只有奇数泛音
        n_frames=120,
        gp_variance_range=(0.8, 1.5),
        gp_lengthscale_range=(0.3, 0.6),
        f0_mode="vibrato",
        vibrato_depth_cents_range=(8.0, 20.0),
        vibrato_period_range=(0.12, 0.25),
        vibrato_lengthscale_range=(0.05, 0.15),
        vibrato_variance_range=(0.5, 1.5),
        attack_time_range=(0.02, 0.06),
        release_time_range=(0.05, 0.15),
    )


def trumpet_params(rng: np.random.Generator) -> dict:
    """
    小号风格预设：
    - 丰富的泛音（高次泛音强），rolloff 小
    - constant (GP 主包络) 模式
    - 独立高斯噪声让各泛音幅度有微小差异
    - 明显 vibrato
    """
    return sample_params(
        rng,
        n_overtones=25,
        series_mode="harmonic",
        jitter_cents=3.0,
        amp_mode="constant",
        rolloff_range=(0.5, 1.0),         # 泛音衰减慢，高频丰富（铜管特征）
        amp_noise_mode="gaussian",
        amp_noise_std_range=(0.05, 0.15),
        odd_even_ratio_range=(1.0, 1.0),  # 奇偶泛音都有
        n_frames=120,
        gp_variance_range=(1.0, 2.0),
        gp_lengthscale_range=(0.25, 0.5),
        f0_mode="vibrato",
        vibrato_depth_cents_range=(15.0, 40.0),
        vibrato_period_range=(0.10, 0.20),
        vibrato_lengthscale_range=(0.05, 0.12),
        vibrato_variance_range=(0.8, 1.8),
        attack_time_range=(0.01, 0.04),
        release_time_range=(0.05, 0.2),
    )


def random_sample_params(rng: np.random.Generator) -> dict:
    """
    通用随机采样方案，覆盖多种音色风格。

    分布设计:
    - amp_mode: constant:exp_decay = 1:9
    - series_mode: uniform {harmonic, stretched}
    - n_overtones: uniform [9, 20]
    - rolloff: uniform [1.1, 2.0]
    - f0_mode: constant:vibrato = 1:1
    - gp_modulation_depth: 根据 amp_mode 自适应
        constant 模式不使用 gp_modulation_depth (GP 是主包络)
        exp_decay 模式使用小值微调制 [0.03, 0.15]
    - amp_noise_mode: uniform {none, gaussian, rbf}
    - odd_even_ratio: 大部分 1.0，偶尔 clarinet 风格
    """
    # amp_mode: 10% constant, 90% exp_decay
    amp_mode = "constant" if rng.random() < 0.1 else "exp_decay"

    # series_mode: uniform
    series_mode = rng.choice(["harmonic", "stretched"])
    stretch_beta = 1.0 + rng.uniform(0.001, 0.01) if series_mode == "stretched" else 1.0

    # n_overtones: uniform 9~20
    n_overtones = int(rng.integers(9, 21))

    # f0_mode: 50% constant, 50% vibrato
    f0_mode = rng.choice(["constant", "vibrato"])

    # amp_noise_mode: uniform {none, gaussian, rbf}
    amp_noise_mode = rng.choice(["none", "gaussian", "rbf"])

    # odd_even_ratio: 80% 正常 (0.8~1.0), 20% 奇数为主 (0.0~0.3)
    if rng.random() < 0.2:
        odd_even_ratio = rng.uniform(0.0, 0.3)
    else:
        odd_even_ratio = rng.uniform(0.8, 1.0)

    # gp_modulation_depth: constant 模式不用 (设为 0), exp_decay 模式小值微调
    if amp_mode == "constant":
        gp_modulation_depth = 0.0  # GP 是主包络，不需要额外微调制
    else:
        gp_modulation_depth = rng.uniform(0.03, 0.15)

    return {
        # 泛音
        "n_overtones": n_overtones,
        "series_mode": series_mode,
        "custom_ratios": None,
        "stretch_beta": stretch_beta,
        "jitter_cents": rng.uniform(2.0, 8.0),
        # 幅度
        "amp_mode": amp_mode,
        "rolloff": rng.uniform(1.1, 2.0),
        "amp_noise_mode": amp_noise_mode,
        "amp_noise_std": rng.uniform(0.05, 0.2),
        "amp_noise_rbf_lengthscale": rng.uniform(0.2, 0.5),
        "odd_even_ratio": odd_even_ratio,
        # exp_decay 时域衰减
        "decay_base": rng.uniform(1.0, 4.0),
        "decay_highfreq": rng.uniform(10.0, 500.0),
        "decay_shape": rng.uniform(1.0, 2.0),
        # GP
        "n_frames": 100,
        "gp_variance": rng.uniform(0.5, 2.5),
        "gp_lengthscale": rng.uniform(0.15, 0.6),
        "gp_modulation_depth": gp_modulation_depth,
        # f0
        "f0_mode": f0_mode,
        "vibrato_depth_cents": rng.uniform(10.0, 50.0),
        "vibrato_period": rng.uniform(0.08, 0.3),
        "vibrato_lengthscale": rng.uniform(0.05, 0.2),
        "vibrato_variance": rng.uniform(0.5, 2.0),
        # 包络
        "attack_time": rng.uniform(0.002, 0.05),
        "release_time": rng.uniform(0.05, 0.4),
    }


# ---------------------------------------------------------------------------
# 主合成函数（确定性参数，不含范围/随机采样）
# ---------------------------------------------------------------------------

def generate_music_note(
    pitch: float = 60,
    duration: float = 2.0,
    velocity: float = 0.8,
    sr: int = 44100,
    seed: int | None = None,
    # --- 泛音序列 ---
    n_overtones: int = 20,
    series_mode: str = "harmonic",
    custom_ratios: list | None = None,
    stretch_beta: float = 1.0,
    jitter_cents: float = 5.0,
    # --- 幅度模式 ---
    amp_mode: str = "exp_decay",         # "constant" | "exp_decay"
    rolloff: float = 1.5,
    amp_noise_mode: str = "none",         # "none" | "gaussian" | "rbf"
    amp_noise_std: float = 0.1,           # 噪声标准差
    amp_noise_rbf_lengthscale: float = 0.3,  # RBF 噪声的 lengthscale
    odd_even_ratio: float = 1.0,          # 偶数泛音缩放 (0=仅奇数)
    # --- exp_decay 时域衰减（每个泛音独立） ---
    decay_base: float = 2.0,             # 基础衰减率 (1/s)
    decay_highfreq: float = 100.0,       # 高频额外衰减
    decay_shape: float = 1.5,            # 衰减的频率依赖指数
    # --- GP 时域调制 ---
    n_frames: int = 100,
    gp_variance: float = 1.0,
    gp_lengthscale: float = 0.25,
    gp_modulation_depth: float = 0.15,   # exp_decay 下 GP 微调制深度
    # --- f0 模式 ---
    f0_mode: str = "constant",
    vibrato_depth_cents: float = 30.0,
    vibrato_period: float = 0.8,
    vibrato_lengthscale: float = 0.15,
    vibrato_variance: float = 1.0,
    # --- 包络 ---
    attack_time: float = 0.01,
    release_time: float = 0.1,
    # --- 输出 ---
    normalize: bool = True,
    # --- 预计算协方差（每个 clip 共享，避免重复 O(n^3) 分解）---
    L_gp_exp: np.ndarray | None = None,     # Exponential 核 Cholesky (n_frames, n_frames)
    L_gp_vibrato: np.ndarray | None = None, # StdPeriodic 核 Cholesky (n_frames, n_frames)
) -> tuple[np.ndarray, dict]:
    """
    生成单音符音频。所有参数都是确定性的值（不含范围）。
    随机性仅来自 seed 控制的 rng（影响泛音 jitter、GP 采样、相位等）。

    amp_mode 对包络的影响:
        "constant" : GP 是主包络
        "exp_decay": 每个泛音独立指数衰减（高频更快），GP 仅微调制

    f0_mode 对频率的影响:
        "constant" : 固定 f0
        "vibrato"  : StdPeriodic 核时变 f0，用 cumsum 累积相位

    实际用到的 GP 控制帧数 actual_frames:
        从 n_frames 中随机取 actual_frames ∈ [2, n_frames * min(1, duration/5)] 个控制点，
        其余位置线性插值 → 短音符包络更粗糙/不连续，长音符可达到全分辨率。
    """
    rng = np.random.default_rng(seed)
    vel = float(np.clip(velocity, 0.0, 1.0))

    f0 = midi_to_hz(pitch)
    nyquist = 0.5 * sr
    total_duration = duration + release_time
    n_samples = int(np.round(total_duration * sr))
    t = np.arange(n_samples, dtype=np.float64) / sr

    # ------------------------------------------------------------------
    # 1. 泛音序列
    # ------------------------------------------------------------------
    ratios = sample_overtone_series(
        rng, n_overtones, series_mode, custom_ratios, stretch_beta, jitter_cents
    )
    nominal_freqs = f0 * ratios
    # 保留所有低于 Nyquist 的泛音；至少保留基频（ratio 最小的那个，即 ratios[0]≈1.0）
    valid_mask = nominal_freqs < nyquist
    valid_mask[0] = True  # 保证 f0 本身始终进入合成
    ratios = ratios[valid_mask]
    nominal_freqs = nominal_freqs[valid_mask]
    n_valid = len(nominal_freqs)

    # ------------------------------------------------------------------
    # 2. 静态初始幅度
    # ------------------------------------------------------------------
    base_amps = compute_base_amps(
        rng, nominal_freqs, nyquist, rolloff,
        amp_noise_mode, amp_noise_std, amp_noise_rbf_lengthscale, odd_even_ratio,
    )
    vel_filter = np.exp(-nominal_freqs / (3500.0 + 2500.0 * vel))
    base_amps = base_amps * vel_filter

    # ------------------------------------------------------------------
    # 3. 时变 f0 & GP：先在 n_frames 网格上采样，再随机子采样控制点插值
    # actual_frames ∈ [2, n_frames * min(1, duration/5)]，越少越粗糙
    # ------------------------------------------------------------------
    frac_upper = min(1.0, duration / 5.0)
    frac_upper = max(2.0 / n_frames, frac_upper)       # 至少保证 2 个控制点
    actual_frames = max(2, int(n_frames * rng.uniform(1.0 / n_frames, frac_upper)))

    dense_times = np.linspace(0, total_duration, n_frames)
    sparse_idx = np.round(np.linspace(0, n_frames - 1, actual_frames)).astype(int)
    sparse_times = dense_times[sparse_idx]

    f0_full = sample_f0_trajectory(
        rng, f0, n_frames, f0_mode,
        vibrato_variance, vibrato_lengthscale, vibrato_period, vibrato_depth_cents,
        L_vibrato=L_gp_vibrato,
    )
    f0_samples = np.interp(t, sparse_times, f0_full[sparse_idx])  # 子采样插值

    # ------------------------------------------------------------------
    # 4. GP 时域包络（子采样后线性插值 → 粗糙包络）
    # ------------------------------------------------------------------
    gp_full = sample_gp_exponential(rng, n_frames, gp_variance, gp_lengthscale, L=L_gp_exp)
    gp_env = np.interp(t, sparse_times, gp_full[sparse_idx])
    gp_env = gp_env / (gp_env.max() + 1e-9)

    # ------------------------------------------------------------------
    # 5. Attack / Release
    # ------------------------------------------------------------------
    ar_env = np.ones(n_samples, dtype=np.float64)
    n_attack = int(attack_time * sr)
    if n_attack > 0:
        ar_env[:n_attack] *= np.linspace(0.0, 1.0, n_attack) ** 2
    n_release = int(release_time * sr)
    if n_release > 0:
        ar_env[-n_release:] *= np.linspace(1.0, 0.0, n_release) ** 2

    # ------------------------------------------------------------------
    # 6. 加性合成：cumsum 累积相位（参考 formula_sed/ddsp/core.py）
    # ------------------------------------------------------------------
    y = np.zeros(n_samples, dtype=np.float64)

    for i, (ratio_k, ak) in enumerate(zip(ratios, base_amps)):
        phase0 = 2.0 * np.pi * rng.random()

        # 瞬时频率 = f0(t) * ratio_k -> rad/sample -> cumsum
        omega = f0_samples * ratio_k * (2.0 * np.pi / sr)
        phase = np.cumsum(omega) + phase0
        partial = np.sin(phase)

        if amp_mode == "exp_decay":
            # 每个泛音的频率相关指数衰减
            freq_norm = nominal_freqs[i] / max(nyquist, 1.0)
            decay_rate = decay_base + decay_highfreq * (freq_norm ** decay_shape)
            per_harmonic_env = np.exp(-decay_rate * t)
            # GP 微调制
            gp_mod = 1.0 + gp_modulation_depth * (gp_env - gp_env.mean())
            per_harmonic_env = per_harmonic_env * gp_mod
            y += ak * partial * per_harmonic_env
        elif amp_mode == "constant":
            y += ak * partial

    # ------------------------------------------------------------------
    # 7. 总包络
    # ------------------------------------------------------------------
    if amp_mode == "constant":
        final_env = gp_env * ar_env
        y *= final_env
    else:
        # 用基频的衰减曲线作为代表性包络（用于可视化）
        representative_decay = np.exp(-decay_base * t)
        gp_mod = 1.0 + gp_modulation_depth * (gp_env - gp_env.mean())
        final_env = representative_decay * gp_mod * ar_env
        y *= ar_env

    # ------------------------------------------------------------------
    # 8. 归一化
    # ------------------------------------------------------------------
    if normalize:
        peak = np.max(np.abs(y)) + 1e-9
        y = 0.95 * y / peak

    info = {
        "f0": f0,
        "f0_frames": f0_full,
        "f0_samples": f0_samples,
        "nominal_freqs": nominal_freqs,
        "base_amps": base_amps,
        "n_valid_overtones": n_valid,
        "gp_env_frames": gp_full,
        "gp_env": gp_env,
        "final_env": final_env,
        "ar_env": ar_env,
        "amp_mode": amp_mode,
        "f0_mode": f0_mode,
    }
    return y.astype(np.float32), info


# ---------------------------------------------------------------------------
# 示例入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import soundfile as sf
    import matplotlib.pyplot as plt

    sr = 44100

    # --- 示例1: 钢琴 (exp_decay, 大衰减) ---
    rng1 = np.random.default_rng(42)
    p1 = piano_params(rng1)
    y1, info1 = generate_music_note(pitch=60, duration=2.0, velocity=0.8, sr=sr, seed=42, **p1)
    sf.write("music_note_piano.wav", y1, sr)
    print(f"示例1 piano: {y1.shape}, 泛音={info1['n_valid_overtones']}, "
          f"decay_base={p1['decay_base']:.1f}, decay_hf={p1['decay_highfreq']:.0f}")

    # --- 示例2: 弦乐器 (GP 主包络 + vibrato) ---
    rng2 = np.random.default_rng(7)
    p2 = string_params(rng2)
    y2, info2 = generate_music_note(pitch=60, duration=3.0, velocity=0.6, sr=sr, seed=7, **p2)
    sf.write("music_note_string.wav", y2, sr)
    print(f"示例2 string: {y2.shape}, 泛音={info2['n_valid_overtones']}, "
          f"vibrato_depth={p2['vibrato_depth_cents']:.1f}cents")

    # --- 示例3: 钟声 (自定义泛音 + 慢衰减) ---
    rng3 = np.random.default_rng(99)
    p3 = bell_params(rng3)
    y3, info3 = generate_music_note(pitch=69, duration=3.0, velocity=0.9, sr=sr, seed=99, **p3)
    sf.write("music_note_bell.wav", y3, sr)
    print(f"示例3 bell: {y3.shape}, 泛音={info3['n_valid_overtones']}, "
          f"decay_base={p3['decay_base']:.2f}")

    # --- 示例4: 单簧管 (奇数泛音 + RBF 噪声) ---
    rng4 = np.random.default_rng(13)
    p4 = clarinet_params(rng4)
    y4, info4 = generate_music_note(pitch=62, duration=2.5, velocity=0.7, sr=sr, seed=13, **p4)
    sf.write("music_note_clarinet.wav", y4, sr)
    print(f"示例4 clarinet: {y4.shape}, 泛音={info4['n_valid_overtones']}, "
          f"odd_even_ratio={p4['odd_even_ratio']:.3f}")

    # --- 示例5: 小号 (丰富泛音 + gaussian 噪声 + vibrato) ---
    rng5 = np.random.default_rng(21)
    p5 = trumpet_params(rng5)
    y5, info5 = generate_music_note(pitch=67, duration=2.0, velocity=0.85, sr=sr, seed=21, **p5)
    sf.write("music_note_trumpet.wav", y5, sr)
    print(f"示例5 trumpet: {y5.shape}, 泛音={info5['n_valid_overtones']}, "
          f"rolloff={p5['rolloff']:.2f}")

    # --- 可视化 ---
    fig, axes = plt.subplots(5, 3, figsize=(18, 17))
    titles = ["Piano (exp_decay)", "String (GP + vibrato)", "Bell (custom overtones)",
              "Clarinet (odd harmonics)", "Trumpet (rich harmonics)"]

    for idx, (y, info, title) in enumerate(zip(
        [y1, y2, y3, y4, y5], [info1, info2, info3, info4, info5], titles
    )):
        t_samples = np.arange(len(y)) / sr

        axes[idx, 0].specgram(y, Fs=sr, scale="dB", NFFT=2048, noverlap=1024)
        axes[idx, 0].set_title(f"{title} - spectrogram")
        axes[idx, 0].set_ylabel("Frequency (Hz)")
        axes[idx, 0].set_xlabel("Time (s)")

        env_label = "exp_decay × GP (f0)" if info["amp_mode"] == "exp_decay" else "GP × AR"
        axes[idx, 1].plot(t_samples, info["final_env"], label=env_label, alpha=0.7)
        t_frames = np.linspace(0, t_samples[-1], len(info["gp_env_frames"]))
        gp_norm = info["gp_env_frames"] / (info["gp_env_frames"].max() + 1e-9)
        axes[idx, 1].plot(t_frames, gp_norm, label="GP (norm)", linestyle="--", alpha=0.7)
        axes[idx, 1].plot(t_samples, info["ar_env"], label="AR env", linestyle=":", alpha=0.5)
        axes[idx, 1].set_title(f"{title} - envelope")
        axes[idx, 1].legend(fontsize=8)
        axes[idx, 1].set_xlabel("Time (s)")

        t_f0 = np.linspace(0, t_samples[-1], len(info["f0_frames"]))
        axes[idx, 2].plot(t_f0, info["f0_frames"])
        axes[idx, 2].set_title(f"{title} - f0 trajectory")
        axes[idx, 2].set_ylabel("f0 (Hz)")
        axes[idx, 2].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig("music_synth_gp_demo.png", dpi=120)
    plt.show()

    # --- random_sample_params: 10 notes ---
    import time
    print("\n=== random_sample_params: 10 notes ===")
    rng_bench = np.random.default_rng(0)
    durations = rng_bench.uniform(0.5, 3.0, size=10)
    pitches = rng_bench.integers(48, 84, size=10)  # C3 ~ B5

    for i in range(10):
        params = random_sample_params(np.random.default_rng(i))
        t0 = time.perf_counter()
        y_b, info_b = generate_music_note(
            pitch=float(pitches[i]), duration=float(durations[i]),
            velocity=rng_bench.uniform(0.3, 1.0), sr=sr, seed=i, **params,
        )
        elapsed = time.perf_counter() - t0
        fname = f"random_note_{i:02d}.wav"
        sf.write(fname, y_b, sr)
        print(f"  [{i:02d}] pitch={pitches[i]}, dur={durations[i]:.2f}s, "
              f"amp_mode={params['amp_mode']}, f0_mode={params['f0_mode']}, "
              f"overtones={info_b['n_valid_overtones']}, "
              f"wall={elapsed*1000:.1f}ms -> {fname}")
