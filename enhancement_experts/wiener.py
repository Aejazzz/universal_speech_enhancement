"""
Frequency-domain Wiener filter with the **decision-directed** a-priori SNR estimator
(Ephraim & Malah 1984), i.e. the classical reference for parametric SE since the 1980s.

Implementation notes:
  * Noise PSD is estimated from the first ``noise_init_sec`` of audio.
  * The decision-directed update (``alpha`` smoothing) prevents musical noise.
  * Output is a soft Wiener gain mask applied to the noisy magnitude spectrum.
  * Strength 1.0 == fully filtered, strength 0.0 == passthrough.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import librosa
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"wiener requires librosa: {exc}") from exc

from enhancement_experts.base import EnhancementExpert


@dataclass
class WienerConfig:
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    alpha: float = 0.98       # decision-directed smoothing (Ephraim-Malah)
    noise_init_sec: float = 0.3
    snr_floor_db: float = -15.0


class WienerFilterExpert(EnhancementExpert):
    name = "WienerFilter"

    def __init__(self, config: WienerConfig | None = None) -> None:
        self.cfg = config or WienerConfig()

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
        psd = mag ** 2 + 1e-12

        # Initial noise PSD from the first ``noise_init_sec`` (assumes audio starts in silence/noise).
        n_init_frames = max(2, int(c.noise_init_sec * sr / c.hop_length))
        n_init_frames = min(n_init_frames, mag.shape[1])
        noise_psd = np.mean(psd[:, :n_init_frames], axis=1, keepdims=True) + 1e-12

        # Posterior SNR: power ratio of observation to noise.
        snr_post = np.maximum(psd / noise_psd - 1.0, 0.0)

        # Decision-directed a-priori SNR: smooth using previous gain.
        snr_prio = np.zeros_like(psd)
        gain = np.zeros_like(psd)
        snr_floor = 10.0 ** (c.snr_floor_db / 10.0)
        prev_gain = np.zeros(psd.shape[0])
        for k in range(psd.shape[1]):
            if k == 0:
                snr_prio[:, k] = snr_post[:, k]
            else:
                snr_prio[:, k] = (
                    c.alpha * (prev_gain ** 2) * psd[:, k - 1] / noise_psd[:, 0]
                    + (1.0 - c.alpha) * snr_post[:, k]
                )
            snr_prio[:, k] = np.maximum(snr_prio[:, k], snr_floor)
            gain[:, k] = snr_prio[:, k] / (1.0 + snr_prio[:, k])
            prev_gain = gain[:, k]

        new_stft = gain * mag * np.exp(1j * phase)
        out = librosa.istft(new_stft, hop_length=c.hop_length, length=len(waveform))
        return out.astype(np.float32)

    def enhance(self, waveform: np.ndarray, sr: int, strength: float) -> np.ndarray:
        if waveform.size == 0:
            return waveform
        denoised = self._denoise(waveform.astype(np.float32), sr)
        s = float(np.clip(strength, 0.0, 1.0))
        return (s * denoised + (1.0 - s) * waveform.astype(np.float32)).astype(np.float32)
