from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio

from enhancement_experts.base import EnhancementExpert


class MossFormer2Expert(EnhancementExpert):
    name = "MossFormer2"

    def __init__(self, checkpoint_path: str = "checkpoints/mossformer2.pt", device: str = "cuda") -> None:
        self.device = device
        self.model = None
        # Empty string becomes Path("") which is "." — must not treat as a checkpoint.
        raw = (checkpoint_path or "").strip()
        self.checkpoint_path: Path | None = Path(raw).expanduser() if raw else None
        self._load_model()

    def _load_model(self) -> None:
        # Adapter point for ClearerVoice-Studio checkpoint/model class.
        cp = self.checkpoint_path
        if cp is None or not cp.is_file():
            return
        self.model = torch.jit.load(str(cp), map_location=self.device)
        self.model.eval()

    def enhance(self, waveform: np.ndarray, sr: int, strength: float) -> np.ndarray:
        if self.model is None:
            return waveform
        with torch.no_grad():
            audio = torch.tensor(waveform, dtype=torch.float32, device=self.device).unsqueeze(0)
            out = self.model(audio).squeeze(0).detach().cpu().numpy()
        mixed = strength * out + (1.0 - strength) * waveform
        return mixed.astype(np.float32)
