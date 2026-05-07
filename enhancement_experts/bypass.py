from __future__ import annotations

import numpy as np

from enhancement_experts.base import EnhancementExpert


class BypassExpert(EnhancementExpert):
    name = "BYPASS"

    def enhance(self, waveform: np.ndarray, sr: int, strength: float) -> np.ndarray:
        return waveform
