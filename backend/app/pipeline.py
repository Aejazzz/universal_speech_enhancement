from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from backend.app.audio import load_audio, save_audio
from backend.app.config import AppConfig
from backend.app.preprocess import preprocess, soft_limiter

logger = logging.getLogger(__name__)
from distortion_analyzer.model import DistortionAnalyzer
from distortion_analyzer.routing_features import format_reason, summarize_for_dashboard
from enhancement_experts.factory import build_experts
from evaluation.metrics import compute_metrics, dnsmos_full, dnsmos_score, utmos_score
from policy_agent.model import ACTIONS, TransformerPolicyAgent
from visualizations.plots import save_policy_plot, save_spectrogram_plot, save_waveform_plot

# Margin (DNSMOS OVRL) by which a non-BYPASS candidate must beat BYPASS to be picked.
# Mirrors the "do no harm" margin used in scripts/rederive_oracle_labels.py.
_DYNAMIC_BYPASS_MARGIN = 0.02

# All experts the dynamic router is allowed to consider, in priority order. Classical experts
# are FFT-only and always available; neural experts depend on checkpoints / external installs.
_DYNAMIC_CANDIDATE_ORDER = (
    "DeepFilterNet3",
    "WPEDereverb",
    "NoiseReduce",
    "SpectralGate",
    "WienerFilter",
    "SpectralSubtraction",
    "ResembleEnhance",
    "MossFormer2",
)


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

        Classical FFT experts (SpectralGate, WienerFilter, SpectralSubtraction) are always
        active — they need no checkpoints. Neural experts only count if they actually have
        usable weights / packages installed; otherwise including them in the dynamic
        candidate set would only inflate latency.
        """
        if name == "BYPASS":
            return True
        if name in {"SpectralGate", "WienerFilter", "SpectralSubtraction"}:
            return True
        if name == "DeepFilterNet3":
            return True  # public model is always downloadable on first call
        if name in {"WPEDereverb", "NoiseReduce"}:
            expert = self.experts.get(name)
            return bool(getattr(expert, "available", False))
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
        Speculate-and-measure across every active expert at a strength sweep.

        Each candidate gets:
          * **DNSMOS P.835** OVRL/SIG/BAK (Microsoft) — measures perceived signal/background quality
          * **UTMOS22 strong** (Saeki 2022, sarulab) — orthogonal artifact detector

        Combined ``rank_score = 0.45·OVRL + 0.20·SIG + 0.15·BAK + 0.20·UTMOS``. UTMOS is a
        different-architecture predictor trained on different MOS data, so adding it catches
        artifacts (musical noise, over-suppression) that DNSMOS misses. A "do no harm" margin
        keeps BYPASS preferred on near-ties.

        Returns ``(expert_name, strength, enhanced_waveform, candidates, reason)``.
        """

        def _score_pair(wave: np.ndarray) -> tuple[Dict[str, float], float]:
            return dnsmos_full(wave, sr), float(utmos_score(wave, sr))

        def _ranking_score(d: Dict[str, float], utmos: float) -> float:
            """Weighted combination of DNSMOS P.835 components and UTMOS22.

            Signal-quality components (SIG, OVRL, UTMOS) get most of the weight; BAK alone
            isn't enough — an over-aggressive enhancer can crush background to near-zero
            without improving the actual speech.
            """
            ovrl = float(d.get("ovrl", 3.0))
            sig = float(d.get("sig", 3.0))
            bak = float(d.get("bak", 3.0))
            return 0.45 * ovrl + 0.20 * sig + 0.15 * bak + 0.20 * float(utmos)

        bypass_dns, bypass_utmos = _score_pair(waveform)
        candidates: list[dict[str, Any]] = [
            {
                "expert": "BYPASS",
                "strength": 0.0,
                "dnsmos": bypass_dns.get("ovrl"),
                "dnsmos_sig": bypass_dns.get("sig"),
                "dnsmos_bak": bypass_dns.get("bak"),
                "utmos": bypass_utmos,
                "rank_score": _ranking_score(bypass_dns, bypass_utmos),
            }
        ]
        cached: dict[tuple[str, float], np.ndarray] = {("BYPASS", 0.0): waveform}

        # Strength sweep — moderate to aggressive. UTMOS in the rank score automatically
        # penalises over-aggressive enhancers that introduce artifacts, so we can safely
        # try strong settings: 0.55 (gentle) / 0.80 (firm) / 0.95 (full).
        policy_clamped = float(np.clip(policy_strength, 0.4, 0.95))
        sweep = sorted({policy_clamped, 0.55, 0.80, 0.95})

        for name in _DYNAMIC_CANDIDATE_ORDER:
            if not self._expert_is_active(name):
                continue
            expert = self.experts.get(name)
            if expert is None:
                continue
            for s in sweep:
                try:
                    enh = expert.enhance(waveform, sr, float(s))
                    enh = np.asarray(enh, dtype=np.float32)
                    if len(enh) != len(waveform):
                        if len(enh) > len(waveform):
                            enh = enh[: len(waveform)].copy()
                        else:
                            enh = np.pad(enh, (0, len(waveform) - len(enh)))
                    dns, ut = _score_pair(enh)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("dynamic candidate %s@%.2f failed: %s", name, s, exc)
                    continue
                candidates.append({
                    "expert": name,
                    "strength": float(s),
                    "dnsmos": dns.get("ovrl"),
                    "dnsmos_sig": dns.get("sig"),
                    "dnsmos_bak": dns.get("bak"),
                    "utmos": ut,
                    "rank_score": _ranking_score(dns, ut),
                })
                cached[(name, float(s))] = enh

        # Pick the max rank_score, but penalize non-BYPASS by the margin so BYPASS wins on near-ties.
        def _key(c: dict[str, Any]) -> float:
            penalty = 0.0 if c["expert"] == "BYPASS" else _DYNAMIC_BYPASS_MARGIN
            return float(c["rank_score"]) - penalty

        best = max(candidates, key=_key)
        chosen_audio = cached[(best["expert"], float(best["strength"]))]
        bypass_rank = candidates[0]["rank_score"]
        if best["expert"] == "BYPASS":
            reason = (
                f"BYPASS rank={bypass_rank:.3f} (OVRL={bypass_dns.get('ovrl'):.3f}, "
                f"UTMOS={bypass_utmos:.3f}) - input is best of {len(candidates)} candidates"
            )
        else:
            reason = (
                f"{best['expert']}@{best['strength']:.2f} rank={best['rank_score']:.3f} "
                f"(OVRL={best['dnsmos']:.3f}, UTMOS={best['utmos']:.3f}) beat BYPASS rank={bypass_rank:.3f}"
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

        t_start = time.perf_counter()
        raw_waveform, sr = load_audio(input_path, self.config.system.sample_rate)
        t_load = time.perf_counter()
        # Preprocess: DC removal, 60Hz HPF, ITU-R BS.1770 loudness norm to -23 LUFS (or peak).
        waveform, preprocess_report = preprocess(raw_waveform, sr)
        logger.info(
            "Preprocess: dc=%.4f rms=%.1f->%.1fdB lufs=%s->%s",
            preprocess_report.dc_offset_removed,
            preprocess_report.rms_db_in,
            preprocess_report.rms_db_out,
            preprocess_report.loudness_lufs_in,
            preprocess_report.loudness_lufs_out,
        )
        distortion_summary = summarize_for_dashboard(waveform, sr)
        distortion = self.distortion_model.predict(waveform, sr).to(self.device).unsqueeze(0)
        wav_tensor = torch.tensor(waveform, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.amp.autocast(device_type="cuda", enabled=self.config.system.mixed_precision and self.device == "cuda"):
            policy_output = self.policy(wav_tensor, distortion)
        t_policy = time.perf_counter()

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

        # Post: brick-wall limiter at -1 dBFS so no expert ships an output that clips.
        enhanced = soft_limiter(enhanced.astype(np.float32), ceiling_db=-1.0)
        t_enhance = time.perf_counter()

        reference = None
        if reference_path:
            reference, _ = load_audio(reference_path, sr)
        metrics = compute_metrics(waveform, enhanced, sr, reference)
        t_metrics = time.perf_counter()

        timings = {
            "load_ms": round((t_load - t_start) * 1000, 1),
            "policy_ms": round((t_policy - t_load) * 1000, 1),
            "enhance_ms": round((t_enhance - t_policy) * 1000, 1),
            "metrics_ms": round((t_metrics - t_enhance) * 1000, 1),
            "total_ms": round((t_metrics - t_start) * 1000, 1),
            "audio_seconds": round(len(waveform) / sr, 2),
            "rtf": round((t_metrics - t_start) / max(len(waveform) / sr, 1e-3), 3),
        }
        logger.info("timings: %s", timings)

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
            "preprocess": preprocess_report.as_dict(),
            "timings": timings,
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
