from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch
import torch.nn as nn
from transformers import WavLMModel, Wav2Vec2FeatureExtractor


ACTIONS = ["DeepFilterNet3", "ResembleEnhance", "MossFormer2", "BYPASS"]


@dataclass
class PolicyOutput:
    expert_idx: int
    probabilities: Dict[str, float]
    strength: float
    refine: bool
    confidence: float


class TransformerPolicyAgent(nn.Module):
    """
    Routing policy on top of a *frozen* WavLM encoder.

    Design notes (why this is better than a 2-token transformer):
      * WavLM already provides rich phonetic/acoustic features — there is far more signal in
        per-layer embeddings than a single mean-pooled vector. We mean-pool over time at
        several layers (default {6, 9, 12}) and concatenate. This mirrors common practice
        in WavLM/HuBERT downstream tasks (SUPERB benchmark, NISQA-style heads).
      * The encoder is frozen (params with ``requires_grad=False``). The trainable head is
        a small MLP over the pooled features + the 6-D distortion vector, ending in three
        heads (action / strength / refine). With <2M trainable params this trains fast on
        a laptop GPU and avoids overfitting on a few-thousand-clip dataset.

    Backwards-compatible constructor: same kwargs the rest of the codebase already passes.
    Optional new kwargs (``freeze_wavlm``, ``pool_layers``) have safe defaults.
    """

    def __init__(
        self,
        wavlm_name: str,
        distortion_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        freeze_wavlm: bool = True,
        pool_layers: Iterable[int] | None = None,
    ) -> None:
        super().__init__()
        # Wav2Vec2FeatureExtractor is exposed for callers that may need preprocessing
        # statistics; the model itself accepts already-normalized waveforms in [-1, 1].
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wavlm_name)
        self.wavlm = WavLMModel.from_pretrained(wavlm_name)

        # Force hidden-state output without changing transformers' default config too aggressively.
        self.wavlm.config.output_hidden_states = True

        if freeze_wavlm:
            for p in self.wavlm.parameters():
                p.requires_grad = False
            self.wavlm.eval()

        wavlm_hidden = self.wavlm.config.hidden_size  # 768 for base-plus
        n_total_layers = self.wavlm.config.num_hidden_layers  # 12 for base-plus
        if pool_layers is None:
            # Sensible defaults: mid + late + final.
            pool_layers = (max(1, n_total_layers // 2), max(1, (3 * n_total_layers) // 4), n_total_layers)
        self.pool_layers: list[int] = sorted({int(i) for i in pool_layers if 0 <= int(i) <= n_total_layers})
        if not self.pool_layers:
            self.pool_layers = [n_total_layers]

        feat_dim = wavlm_hidden * len(self.pool_layers)

        # Trainable routing trunk (small).
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.distortion_proj = nn.Sequential(
            nn.LayerNorm(distortion_dim),
            nn.Linear(distortion_dim, hidden_dim),
            nn.GELU(),
        )
        # ``num_heads`` / ``num_layers`` parameters from the legacy config control the trunk
        # depth; we ignore num_heads (we no longer use multi-head attention here) but honor
        # num_layers as the depth of an MLP residual stack.
        trunk_layers: list[nn.Module] = []
        for _ in range(max(1, num_layers)):
            trunk_layers.append(
                nn.Sequential(
                    nn.LayerNorm(2 * hidden_dim),
                    nn.Linear(2 * hidden_dim, 2 * hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.trunk = nn.ModuleList(trunk_layers)
        self._unused_num_heads = int(num_heads)  # kept for forward-compat with old configs

        self.action_head = nn.Linear(2 * hidden_dim, len(ACTIONS))
        self.strength_head = nn.Sequential(nn.Linear(2 * hidden_dim, 1), nn.Sigmoid())
        self.refine_head = nn.Sequential(nn.Linear(2 * hidden_dim, 1), nn.Sigmoid())

    def _pool_wavlm(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Frozen WavLM forward + multi-layer mean-pool over time."""
        with torch.no_grad():
            outputs = self.wavlm(waveforms, output_hidden_states=True)
        # outputs.hidden_states is a tuple of (B, T, H), length = num_hidden_layers + 1
        # (entry 0 is the conv-feature output before transformer).
        hidden_states = outputs.hidden_states
        layer_pools: list[torch.Tensor] = []
        for layer_idx in self.pool_layers:
            # Clamp into range; transformers exposes layer 0 as conv features and layers 1..N as transformer outputs.
            li = max(0, min(layer_idx, len(hidden_states) - 1))
            layer_pools.append(hidden_states[li].mean(dim=1))
        feat = torch.cat(layer_pools, dim=-1)
        return feat

    def forward_logits(
        self, waveforms: torch.Tensor, distortion_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_feat = self._pool_wavlm(waveforms)
        a = self.audio_proj(audio_feat)
        d = self.distortion_proj(distortion_features)
        x = torch.cat([a, d], dim=-1)
        for block in self.trunk:
            x = x + block(x)
        action_logits = self.action_head(x)
        strength = self.strength_head(x).squeeze(-1)
        refine_prob = self.refine_head(x).squeeze(-1)
        return action_logits, strength, refine_prob

    def forward(self, waveforms: torch.Tensor, distortion_features: torch.Tensor) -> PolicyOutput:
        action_logits, strength, refine_prob = self.forward_logits(waveforms, distortion_features)
        action_probs = torch.softmax(action_logits, dim=-1)
        expert_idx = int(torch.argmax(action_probs[0]).item())
        probs = {name: float(action_probs[0, idx].item()) for idx, name in enumerate(ACTIONS)}
        return PolicyOutput(
            expert_idx=expert_idx,
            probabilities=probs,
            strength=float(strength[0].item()),
            refine=bool(refine_prob[0].item() > 0.5),
            confidence=float(action_probs[0].max().item()),
        )
