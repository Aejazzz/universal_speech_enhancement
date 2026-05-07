"""
Spectral-gating noise reduction (a.k.a. ``noisereduce``-style) implemented from scratch.

Reference: T. Sainburg, "Spectral gating for noise reduction" (timsainb/noisereduce, 2019),
which itself follows the classical approach of computing a stationary noise profile from
the quietest frames and applying a soft mask in the time-frequency domain.

This expert needs no checkpoints — it works on any 16 kHz waveform and is genuinely useful
on broadband stationary noise (HVAC, hum, hiss, low-level white noise).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import librosa
    from scipy.ndimage import uniform_filter
except Exception as exc:  # pragma: no cover - librosa/scipy are required by other modules
    raise RuntimeError(f"spectral_gate requires librosa + scipy: {exc}") from exc

from enhancement_experts.base import EnhancementExpert


@dataclass
class SpectralGateConfig:
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_std_thresh: float = 1.5  # bins above (mean + n_std * std) of noise pass through
    mask_gain_db: float = 20.0  # gain reduction applied to noise bins
    smooth_freq: int = 3
    smooth_time: int = 5
    quiet_quantile: float = 0.10  # use the quietest 10% of frames as the noise model


class SpectralGateExpert(EnhancementExpert):
    name = "SpectralGate"

    def __init__(self, config: SpectralGateConfig | None = None) -> None:
        self.cfg = config or SpectralGateConfig()

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

        # Estimate noise spectrum from the quietest frames (no separate noise clip required).
        frame_energy = np.sum(mag ** 2, axis=0)
        threshold = np.quantile(frame_energy, c.quiet_quantile)
        mask_quiet = frame_energy <= threshold + 1e-8
        if mask_quiet.sum() < 5:
            mask_quiet = np.argsort(frame_energy)[: max(5, len(frame_energy) // 10)]
            noise_frames = mag[:, mask_quiet]
        else:
            noise_frames = mag[:, mask_quiet]
        noise_mean = noise_frames.mean(axis=1)
        noise_std = noise_frames.std(axis=1) + 1e-8
        thresh_per_bin = noise_mean + c.n_std_thresh * noise_std

        soft_mask = mag > thresh_per_bin[:, None]
        soft_mask = uniform_filter(soft_mask.astype(np.float32), size=(c.smooth_freq, c.smooth_time))

        # Convert mask to dB attenuation: 0 dB where mask=1 (speech), -mask_gain_db where mask=0 (noise).
        gain_db = -c.mask_gain_db * (1.0 - soft_mask)
        gain = np.power(10.0, gain_db / 20.0)
        new_mag = mag * gain

        out = librosa.istft(new_mag * np.exp(1j * phase), hop_length=c.hop_length, length=len(waveform))
        return out.astype(np.float32)

    def enhance(self, waveform: np.ndarray, sr: int, strength: float) -> np.ndarray:
        if waveform.size == 0:
            return waveform
        denoised = self._denoise(waveform.astype(np.float32), sr)
        s = float(np.clip(strength, 0.0, 1.0))
        mixed = s * denoised + (1.0 - s) * waveform.astype(np.float32)
        return mixed.astype(np.float32)
