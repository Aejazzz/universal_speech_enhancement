from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EnhancementExpert(ABC):
    name: str

    @abstractmethod
    def enhance(self, waveform: np.ndarray, sr: int, strength: float) -> np.ndarray:
        ...
