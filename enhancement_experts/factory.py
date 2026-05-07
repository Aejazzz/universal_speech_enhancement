from __future__ import annotations

import logging
from typing import Dict

from enhancement_experts.base import EnhancementExpert
from enhancement_experts.bypass import BypassExpert
from enhancement_experts.deepfilternet import DeepFilterNet3Expert
from enhancement_experts.mossformer import MossFormer2Expert
from enhancement_experts.resemble import ResembleEnhanceExpert
from enhancement_experts.noisereduce_expert import NoiseReduceExpert
from enhancement_experts.spectral_gate import SpectralGateExpert
from enhancement_experts.spectral_subtraction import SpectralSubtractionExpert
from enhancement_experts.wiener import WienerFilterExpert
from enhancement_experts.wpe_dereverb import WPEDereverbExpert

logger = logging.getLogger(__name__)


def build_experts(device: str, mossformer_checkpoint: str = "checkpoints/mossformer2.pt") -> Dict[str, EnhancementExpert]:
    """Return the expert registry. Includes neural experts (DeepFilterNet3 etc.) and
    classical FFT-based experts (spectral gating, Wiener, spectral subtraction).

    The classical experts have **zero checkpoint requirements** and are always active —
    they give the dynamic router real options even when the neural-net checkpoints
    are missing.
    """
    experts: Dict[str, EnhancementExpert] = {
        "DeepFilterNet3": DeepFilterNet3Expert(device=device),
        "ResembleEnhance": ResembleEnhanceExpert(device=device),
        "MossFormer2": MossFormer2Expert(checkpoint_path=mossformer_checkpoint, device=device),
        "WPEDereverb": WPEDereverbExpert(),
        "NoiseReduce": NoiseReduceExpert(stationary=False),
        "SpectralGate": SpectralGateExpert(),
        "WienerFilter": WienerFilterExpert(),
        "SpectralSubtraction": SpectralSubtractionExpert(),
        "BYPASS": BypassExpert(),
    }
    logger.info("Built experts: %s", list(experts.keys()))
    return experts
