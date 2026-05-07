from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from backend.app.audio import load_audio, save_audio
from backend.app.config import AppConfig

logger = logging.getLogger(__name__)
from distortion_analyzer.model import DistortionAnalyzer
from distortion_analyzer.routing_features import format_reason, summarize_for_dashboard
from enhancement_experts.factory import build_experts
from evaluation.metrics import compute_metrics, dnsmos_score
from policy_agent.model import ACTIONS, TransformerPolicyAgent
from visualizations.plots import save_policy_plot, save_spectrogram_plot, save_waveform_plot

# Margin (DNSMOS OVRL) by which a non-BYPASS candidate must beat BYPASS to be picked.
# Mirrors the "do no harm" margin used in scripts/rederive_oracle_labels.py.
_DYNAMIC_BYPASS_MARGIN = 0.02


class EnhancementPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = "cuda" if config.system.device == "cuda" and torch.cuda.is_available() else "cpu"
        mossformer_checkpoint = self._resolve_mossformer_checkpoint(config.system.mossformer_checkpoint)
        self.distortion_model = DistortionAnalyzer().to(self.device).eval()
        self.policy = TransformerPolicyAgent(
            wavlm_name=config.policy.wavlm_name,
            distortion_dim=6,
            hidden_dim=config.policy.hidden_dim,
            num_heads=config.policy.num_heads,
            num_layers=config.policy.num_layers,
            dropout=config.policy.dropout,
        ).to(self.device)
        self._load_policy_weights(config.system.policy_checkpoint)
        self.policy.eval()
        self.experts = build_experts(self.device, mossformer_checkpoint=mossformer_checkpoint)

    def _load_policy_weights(self, ckpt_path: str) -> None:
        path = Path(ckpt_path)
        if not ckpt_path or not path.exists():
            logger.warning(
                "Policy checkpoint missing (%s): using random-init router. "
                "Train with scripts/train_policy.py or set system.policy_checkpoint.",
                path,
            )
            return
        payload = torch.load(path, map_location=self.device)
        state = payload.get("state_dict", payload)
        first_key = next(iter(state.keys()))
        if isinstance(first_key, str) and first_key.startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        missing, unexpected = self.policy.load_state_dict(state, strict=False)
        logger.info(
            "Loaded policy weights from %s (missing=%s unexpected=%s)",
            path,
            len(missing),
            len(unexpected),
        )

    def _expert_is_active(self, name: str) -> bool:
        """Whether a given expert can produce a meaningfully different output.

        Resemble Enhance / MossFormer2 silently fall back to identity when their
        checkpoints are not installed; including them in the dynamic candidate set
        would only inflate latency without giving the router useful options.
        """
        if name == "BYPASS":
            return True
        if name == "DeepFilterNet3":
            return True
        if name == "MossFormer2":
            expert = self.experts.get("MossFormer2")
            return getattr(expert, "model", None) is not None
        if name == "ResembleEnhance":
            try:
                import resemble_enhance  # type: ignore  # noqa: F401
                return True
            except Exception:
                return False
        return False

    def _dynamic_select(
        self,
        waveform: np.ndarray,
        sr: int,
        policy_strength: float,
    ) -> tuple[str, float, np.ndarray, list[dict[str, Any]], str]:
        """
        Speculate-and-measure: enhance with every active non-BYPASS expert at a small
        strength sweep, score every candidate (including BYPASS) with DNSMOS OVRL,
        and pick the maximum. A small bias is given to BYPASS so we don't swap in a
        barely-better enhancement that may introduce audible artifacts.

        Returns ``(expert_name, strength, enhanced_waveform, candidates, reason)``.
        """
        dnsmos_input = float(dnsmos_score(waveform, sr))
        candidates: list[dict[str, Any]] = [
            {"expert": "BYPASS", "strength": 0.0, "dnsmos": dnsmos_input}
        ]
        cached: dict[tuple[str, float], np.ndarray] = {("BYPASS", 0.0): waveform}

        # Strength sweep: include the policy's recommended strength (clamped) plus two
        # standard checkpoints. Using a set deduplicates collisions.
        sweep = sorted({
            float(np.clip(policy_strength, 0.4, 0.95)),
            0.7,
            0.95,
        })

        for name in ("DeepFilterNet3", "ResembleEnhance", "MossFormer2"):
            if not self._expert_is_active(name):
                continue
            expert = self.experts[name]
            for s in sweep:
                try:
                    enh = expert.enhance(waveform, sr, float(s))
                    enh = np.asarray(enh, dtype=np.float32)
                    if len(enh) != len(waveform):
                        if len(enh) > len(waveform):
                            enh = enh[: len(waveform)].copy()
                        else:
                            enh = np.pad(enh, (0, len(waveform) - len(enh)))
                    score = float(dnsmos_score(enh, sr))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("dynamic candidate %s@%.2f failed: %s", name, s, exc)
                    continue
                candidates.append({"expert": name, "strength": float(s), "dnsmos": score})
                cached[(name, float(s))] = enh

        # Pick the maximum-DNSMOS candidate, but penalize non-BYPASS by the margin so
        # BYPASS is preferred on ties / near-ties (do-no-harm).
        def _key(c: dict[str, Any]) -> float:
            penalty = 0.0 if c["expert"] == "BYPASS" else _DYNAMIC_BYPASS_MARGIN
            return float(c["dnsmos"]) - penalty

        best = max(candidates, key=_key)
        chosen_audio = cached[(best["expert"], float(best["strength"]))]
        bypass_score = candidates[0]["dnsmos"]
        if best["expert"] == "BYPASS":
            reason = (
                f"BYPASS OVRL={bypass_score:.3f} - input is already best of "
                f"{len(candidates)} candidates"
            )
        else:
            reason = (
                f"{best['expert']}@{best['strength']:.2f} OVRL={best['dnsmos']:.3f} "
                f"beat BYPASS OVRL={bypass_score:.3f} (margin={_DYNAMIC_BYPASS_MARGIN:.3f})"
            )
        return best["expert"], float(best["strength"]), chosen_audio, candidates, reason

    def _resolve_mossformer_checkpoint(self, configured_path: str) -> str:
        if configured_path:
            return configured_path
        search_roots = [Path("checkpoints"), Path("models")]
        patterns = ["*moss*former*.pt", "*moss*former*.pth", "*moss*former*.ckpt"]
        for root in search_roots:
            if not root.exists():
                continue
            for pattern in patterns:
                matches = sorted(root.rglob(pattern))
                if matches:
                    return str(matches[0])
        return "checkpoints/mossformer2.pt"

    def run(self, input_path: str, reference_path: str | None = None) -> Dict[str, Any]:
        run_id = str(uuid.uuid4())[:8]
        out_root = Path(self.config.system.output_root) / run_id
        out_root.mkdir(parents=True, exist_ok=True)

        waveform, sr = load_audio(input_path, self.config.system.sample_rate)
        distortion_summary = summarize_for_dashboard(waveform, sr)
        distortion = self.distortion_model.predict(waveform, sr).to(self.device).unsqueeze(0)
        wav_tensor = torch.tensor(waveform, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.amp.autocast(device_type="cuda", enabled=self.config.system.mixed_precision and self.device == "cuda"):
            policy_output = self.policy(wav_tensor, distortion)

        policy_advice = {
            "expert": ACTIONS[policy_output.expert_idx],
            "strength": float(policy_output.strength),
            "refine": bool(policy_output.refine),
            "confidence": float(policy_output.confidence),
        }

        dynamic_candidates: list[dict[str, Any]] = []
        decision_reason: str
        if self.config.system.dynamic_routing:
            expert_name, chosen_strength, enhanced, dynamic_candidates, decision_reason = (
                self._dynamic_select(waveform, sr, float(policy_output.strength))
            )
        else:
            expert_name = policy_advice["expert"]
            chosen_strength = float(policy_output.strength)
            enhanced = self.experts[expert_name].enhance(waveform, sr, chosen_strength)
            decision_reason = "dynamic_routing disabled - using trained policy directly"

        if policy_output.refine and expert_name != "BYPASS":
            enhanced = self.experts[expert_name].enhance(enhanced, sr, min(1.0, chosen_strength + 0.1))

        reference = None
        if reference_path:
            reference, _ = load_audio(reference_path, sr)
        metrics = compute_metrics(waveform, enhanced, sr, reference)

        output_audio_path = str(out_root / "enhanced.wav")
        metrics_path = str(out_root / "metrics.json")
        routing_path = str(out_root / "routing.json")
        csv_path = str(out_root / "summary.csv")
        waveform_plot = str(out_root / "waveform.png")
        spectrogram_plot = str(out_root / "spectrogram.png")
        policy_plot = str(out_root / "policy_probs.png")

        save_audio(output_audio_path, enhanced, sr)
        save_waveform_plot(waveform, enhanced, sr, waveform_plot)
        save_spectrogram_plot(waveform, enhanced, sr, spectrogram_plot)
        save_policy_plot(policy_output.probabilities, policy_plot)

        with Path(metrics_path).open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        routing = {
            "expert": expert_name,
            "strength": float(chosen_strength),
            "refine": bool(policy_output.refine),
            "confidence": float(policy_output.confidence),
            "probabilities": policy_output.probabilities,
            "distortion_summary": distortion_summary,
            "reason": format_reason(distortion_summary, expert_name),
            "policy_advice": policy_advice,
            "dynamic_candidates": dynamic_candidates,
            "decision_reason": decision_reason,
            "dynamic_routing": bool(self.config.system.dynamic_routing),
        }
        with Path(routing_path).open("w", encoding="utf-8") as handle:
            json.dump(routing, handle, indent=2)

        pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "expert": expert_name,
                    "strength": float(chosen_strength),
                    "refine": bool(policy_output.refine),
                    "policy_advice_expert": policy_advice["expert"],
                    "decision_reason": decision_reason,
                    **{f"improv_{k}": v for k, v in metrics["improvement"].items()},
                }
            ]
        ).to_csv(csv_path, index=False)

        return {
            "id": run_id,
            "input_path": input_path,
            "output_audio_path": output_audio_path,
            "metrics_path": metrics_path,
            "routing_log_path": routing_path,
            "csv_report_path": csv_path,
            "plots": [waveform_plot, spectrogram_plot, policy_plot],
            "routing": routing,
            "metrics": metrics,
        }
