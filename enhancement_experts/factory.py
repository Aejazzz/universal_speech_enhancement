from __future__ import annotations

from typing import Dict

from enhancement_experts.base import EnhancementExpert
from enhancement_experts.bypass import BypassExpert
from enhancement_experts.deepfilternet import DeepFilterNet3Expert
from enhancement_experts.mossformer import MossFormer2Expert
from enhancement_experts.resemble import ResembleEnhanceExpert


def build_experts(device: str, mossformer_checkpoint: str = "checkpoints/mossformer2.pt") -> Dict[str, EnhancementExpert]:
    return {
        "DeepFilterNet3": DeepFilterNet3Expert(device=device),
        "ResembleEnhance": ResembleEnhanceExpert(device=device),
        "MossFormer2": MossFormer2Expert(checkpoint_path=mossformer_checkpoint, device=device),
        "BYPASS": BypassExpert(),
    }
