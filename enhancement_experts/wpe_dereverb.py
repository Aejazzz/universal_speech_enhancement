"""
Single-channel WPE dereverberation (Nakatani et al., 2010, "Speech dereverberation based
on variance-normalized delayed linear prediction"), using the well-tested ``nara_wpe``
implementation.

WPE estimates the room reverberation as a delayed linear prediction of past STFT frames
and subtracts it. Iterative variant ``wpe_v8`` typically converges in 3 iterations.

Uses no external checkpoints — pure DSP. Genuinely targets reverberation, which none of
our other experts handle directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from enhancement_experts.base import EnhancementExpert


@dataclass
class WPEConfig:
    n_fft: int = 512
    hop_length: int = 128
    iterations: int = 3
    taps: int = 10
    delay: int = 3


class WPEDereverbExpert(EnhancementExpert):
    name = "WPEDereverb"

    def __init__(self, config: WPEConfig | None = None) -> None:
        self.cfg = config or WPEConfig()
        try:
            from nara_wpe.wpe import wpe_v8  # type: ignore
            from nara_wpe.utils import istft, stft  # type: ignore

            self._wpe = wpe_v8
            self._stft = stft
            self._istft = istft
            self.available = True
        except Exception as exc:  # noqa: BLE001
            import logging

            logging.getLogger(__name__).warning("nara_wpe unavailable: %s", exc)
            self.available = False

    def _denoise(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        if not self.available or waveform.size < self.cfg.n_fft:
            return waveform
        c = self.cfg
        x = waveform.astype(np.float32)[None, :]  # (D=1, T)
        Y = self._stft(x, size=c.n_fft, shift=c.hop_length)  # (D, T_frames, F)
        Y_fdt = Y.transpose(2, 0, 1)  # (F, D, T_frames) — what wpe_v8 expects
        try:
            Z_fdt = self._wpe(
                Y_fdt,
                taps=c.taps,
                delay=c.delay,
                iterations=c.iterations,
                statistics_mode="full",
            )
        except Exception:
            return waveform
        Z = Z_fdt.transpose(1, 2, 0)  # (D, T_frames, F)
        out = self._istft(Z, size=c.n_fft, shift=c.hop_length)[0]
        out = out[: len(waveform)] if len(out) >= len(waveform) else np.pad(out, (0, len(waveform) - len(out)))
        return out.astype(np.float32)

    def enhance(self, waveform: np.ndarray, sr: int, strength: float) -> np.ndarray:
        if waveform.size == 0:
            return waveform
        denoised = self._denoise(waveform.astype(np.float32), sr)
        s = float(np.clip(strength, 0.0, 1.0))
        return (s * denoised + (1.0 - s) * waveform.astype(np.float32)).astype(np.float32)
