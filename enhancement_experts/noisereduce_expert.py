"""
Noise reduction via the ``noisereduce`` package (Sainburg, Thielk & Gentner, 2020 — used in
*A finite state automaton model of canonical phonological…*; the same library powers many
production audio pipelines).

Default uses ``stationary=False`` — adapts to non-stationary noise frame by frame, which
suits real-world recordings (HVAC + traffic + occasional clicks). Strength is mapped to
``prop_decrease`` (the dB attenuation applied to noise bins).
"""
from __future__ import annotations

import logging

import numpy as np

from enhancement_experts.base import EnhancementExpert

logger = logging.getLogger(__name__)


class NoiseReduceExpert(EnhancementExpert):
    name = "NoiseReduce"

    def __init__(self, stationary: bool = False) -> None:
        self.stationary = stationary
        try:
            import noisereduce as nr  # type: ignore

            self._nr = nr
            self.available = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("noisereduce unavailable: %s", exc)
            self.available = False

    def enhance(self, waveform: np.ndarray, sr: int, strength: float) -> np.ndarray:
        if not self.available or waveform.size == 0:
            return waveform
        s = float(np.clip(strength, 0.0, 1.0))
        try:
            out = self._nr.reduce_noise(
                y=waveform.astype(np.float32),
                sr=int(sr),
                stationary=self.stationary,
                prop_decrease=s,
            )
            out = np.asarray(out, dtype=np.float32)
            if out.size != waveform.size:
                out = out[: waveform.size] if out.size > waveform.size else np.pad(out, (0, waveform.size - out.size))
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("noisereduce.reduce_noise failed: %s", exc)
            return waveform.astype(np.float32)
