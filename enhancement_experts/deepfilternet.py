from __future__ import annotations

import numpy as np
import torch
from df.enhance import enhance
from df.enhance import init_df

from enhancement_experts.base import EnhancementExpert


class DeepFilterNet3Expert(EnhancementExpert):
    name = "DeepFilterNet3"

    def __init__(self, device: str = "cuda") -> None:
        self.model, self.df_state, _ = init_df()
        self.device = device

    def enhance(self, waveform: np.ndarray, sr: int, strength: float) -> np.ndarray:
        audio = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            enhanced = enhance(self.model, self.df_state, audio)
        enhanced_np = enhanced.squeeze(0).cpu().numpy()
        blended = strength * enhanced_np + (1.0 - strength) * waveform
        return blended.astype(np.float32)
