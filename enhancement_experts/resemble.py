from __future__ import annotations

import numpy as np
import torch

from enhancement_experts.base import EnhancementExpert


class ResembleEnhanceExpert(EnhancementExpert):
    name = "ResembleEnhance"

    def __init__(self, device: str = "cuda") -> None:
        self.device = device

    def enhance(self, waveform: np.ndarray, sr: int, strength: float) -> np.ndarray:
        # Uses package-level API when available; fallback keeps compatibility.
        try:
            from resemble_enhance.enhancer.inference import enhance_audio

            tensor = torch.tensor(waveform, dtype=torch.float32, device=self.device).unsqueeze(0)
            out = enhance_audio(tensor, sr)
            enhanced = out.squeeze(0).detach().cpu().numpy()
        except Exception:
            enhanced = waveform
        mixed = strength * enhanced + (1.0 - strength) * waveform
        return mixed.astype(np.float32)
