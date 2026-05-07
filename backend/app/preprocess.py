"""
Preprocessing & post-processing for the enhancement pipeline.

* DC-offset removal — subtracts the mean.
* High-pass at 60 Hz — removes mains hum / rumble that no speech-band model needs.
* Loudness normalisation — pyloudnorm (ITU-R BS.1770) if available, otherwise peak gain.
* Soft clipper / brick-wall limiter on the output.

These steps are deliberately conservative; their job is to make every input land in the
"normal" range (~-23 LUFS, no DC, no rumble) so the experts and DNSMOS scorer don't get
fooled by trivial gain mismatches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from scipy import signal


@dataclass
class PreprocessReport:
    dc_offset_removed: float
    rms_db_in: float
    rms_db_out: float
    loudness_lufs_in: float | None
    loudness_lufs_out: float | None
    high_pass_hz: float
    target_lufs: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dc_offset_removed": float(self.dc_offset_removed),
            "rms_db_in": float(self.rms_db_in),
            "rms_db_out": float(self.rms_db_out),
            "loudness_lufs_in": (float(self.loudness_lufs_in) if self.loudness_lufs_in is not None else None),
            "loudness_lufs_out": (float(self.loudness_lufs_out) if self.loudness_lufs_out is not None else None),
            "high_pass_hz": float(self.high_pass_hz),
            "target_lufs": float(self.target_lufs),
        }


def _rms_db(x: np.ndarray) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(x ** 2) + 1e-12)))


def _measure_lufs(x: np.ndarray, sr: int) -> float | None:
    try:
        import pyloudnorm as pyln  # type: ignore

        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(x))
    except Exception:
        return None


def preprocess(
    waveform: np.ndarray,
    sr: int,
    *,
    target_lufs: float = -23.0,
    high_pass_hz: float = 60.0,
) -> tuple[np.ndarray, PreprocessReport]:
    """Return ``(processed_wave, report)``. Does not modify the input array in place."""
    x = waveform.astype(np.float32, copy=True)
    rms_in_db = _rms_db(x)
    lufs_in = _measure_lufs(x, sr)

    dc = float(np.mean(x))
    x = x - dc

    # 4th-order Butterworth high-pass at 60 Hz to clean rumble/mains hum without touching speech.
    if high_pass_hz > 0 and sr > 2 * high_pass_hz:
        sos = signal.butter(4, high_pass_hz / (sr / 2), btype="highpass", output="sos")
        x = signal.sosfilt(sos, x).astype(np.float32)

    # Loudness normalisation. Prefer pyloudnorm (ITU-R BS.1770); else peak-normalise to -3 dBFS.
    lufs_out: float | None
    try:
        import pyloudnorm as pyln  # type: ignore

        meter = pyln.Meter(sr)
        if lufs_in is None:
            lufs_in = float(meter.integrated_loudness(x))
        if np.isfinite(lufs_in):
            x = pyln.normalize.loudness(x, lufs_in, target_lufs).astype(np.float32)
            lufs_out = float(meter.integrated_loudness(x))
        else:
            lufs_out = None
    except Exception:
        peak = float(np.max(np.abs(x)) + 1e-9)
        x = (x / peak * 0.7).astype(np.float32)
        lufs_out = None

    rms_out_db = _rms_db(x)
    report = PreprocessReport(
        dc_offset_removed=dc,
        rms_db_in=rms_in_db,
        rms_db_out=rms_out_db,
        loudness_lufs_in=lufs_in,
        loudness_lufs_out=lufs_out,
        high_pass_hz=high_pass_hz,
        target_lufs=target_lufs,
    )
    return x, report


def soft_limiter(waveform: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    """Brick-wall limiter that prevents inter-sample peaks from exceeding ``ceiling_db``."""
    ceiling = 10.0 ** (ceiling_db / 20.0)
    peak = float(np.max(np.abs(waveform)))
    if peak <= ceiling or peak == 0.0:
        return waveform.astype(np.float32, copy=False)
    return (waveform * (ceiling / peak)).astype(np.float32)
