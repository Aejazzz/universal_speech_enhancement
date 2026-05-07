"""
Spectral subtraction (Boll, 1979) with over-subtraction and a spectral floor to suppress
musical noise — classical SE textbook method (Loizou, 2013, Ch. 5).

  S_hat[k, t] = max(|Y[k, t]| - alpha * |N_hat[k]|, beta * |Y[k, t]|)

Strength = mix between processed (1.0) and raw (0.0).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import librosa
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"spectral_subtraction requires librosa: {exc}") from exc

from enhancement_experts.base import EnhancementExpert


@dataclass
class SpecSubConfig:
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    over_subtract: float = 2.0   # alpha — how aggressively we subtract noise
    spectral_floor: float = 0.02  # beta — keep this fraction of original magnitude
    noise_init_sec: float = 0.3


class SpectralSubtractionExpert(EnhancementExpert):
    name = "SpectralSubtraction"

    def __init__(self, config: SpecSubConfig | None = None) -> None:
        self.cfg = config or SpecSubConfig()

    def _denoise(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        c = self.cfg
        if waveform.size < c.n_fft:
            return waveform
        stft = librosa.stft(
            waveform.astype(np.float32),
            n_fft=c.n_fft,
            hop_length=c.hop_length,
            win_length=c.win_length,
        )
        mag = np.abs(stft)
        phase = np.angle(stft)
        n_init = max(2, int(c.noise_init_sec * sr / c.hop_length))
        n_init = min(n_init, mag.shape[1])
        noise_mag = np.mean(mag[:, :n_init], axis=1, keepdims=True)
        new_mag = np.maximum(mag - c.over_subtract * noise_mag, c.spectral_floor * mag)
        out = librosa.istft(new_mag * np.exp(1j * phase), hop_length=c.hop_length, length=len(waveform))
        return out.astype(np.float32)

    def enhance(self, waveform: np.ndarray, sr: int, strength: float) -> np.ndarray:
        if waveform.size == 0:
            return waveform
        denoised = self._denoise(waveform.astype(np.float32), sr)
        s = float(np.clip(strength, 0.0, 1.0))
        return (s * denoised + (1.0 - s) * waveform.astype(np.float32)).astype(np.float32)
