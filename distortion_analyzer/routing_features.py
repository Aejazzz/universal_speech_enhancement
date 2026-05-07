"""Acoustic features for dashboard explanations (interpretable, not the CRNN head)."""

from __future__ import annotations

import numpy as np
import librosa


def summarize_for_dashboard(waveform: np.ndarray, sr: int) -> dict[str, float]:
    """Return human-readable distortion summary aligned with routing heuristics."""
    rms = float(np.sqrt(np.mean(waveform**2) + 1e-8))
    snr_db = float(np.clip(20.0 * np.log10(1.0 / (rms + 1e-8)), -5.0, 40.0))
    clip = float(np.mean(np.abs(waveform) > 0.99))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=waveform)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=waveform)))
    reverb = float(np.clip(1.0 - zcr, 0.0, 1.0))
    intelligibility = float(np.clip(1.0 - flatness, 0.0, 1.0))
    codec = float(np.clip(flatness, 0.0, 1.0))
    noise_level = float(np.clip(1.0 - (snr_db / 40.0), 0.0, 1.0))
    return {
        "snr_db": snr_db,
        "reverb": reverb,
        "clip": clip,
        "noise_level": noise_level,
        "codec": codec,
        "intelligibility": intelligibility,
    }


def format_reason(summary: dict[str, float], predicted_expert: str) -> str:
    return (
        f"snr={summary['snr_db']:.2f}, reverb={summary['reverb']:.2f}, "
        f"clip={summary['clip']:.2f}, predicted={predicted_expert}"
    )
