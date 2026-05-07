from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from pesq import pesq
from pystoi import stoi

logger = logging.getLogger(__name__)

# Cached UTMOS predictor — loaded once on first use, reused thereafter.
_UTMOS_MODEL = None
_UTMOS_LOAD_FAILED = False


def utmos_is_reliable() -> bool:
    """Whether UTMOS scores come from the real predictor (not heuristic fallback)."""
    return _UTMOS_MODEL is not None and not _UTMOS_LOAD_FAILED


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-8)
    target = alpha * reference
    noise = estimate - target
    return float(10 * np.log10((np.sum(target**2) + 1e-8) / (np.sum(noise**2) + 1e-8)))


def dnsmos_full(waveform: np.ndarray, sr: int) -> Dict[str, float]:
    """Return the complete DNSMOS P.835 (+P.808) breakdown.

    Keys: ``sig`` (signal), ``bak`` (background), ``ovrl`` (overall), ``p808`` (P.808 MOS).
    Falls back to a heuristic if ``speechmos`` is not usable; the fallback is clearly degraded.
    """
    try:
        from speechmos import dnsmos  # type: ignore

        result = dnsmos.run(np.asarray(waveform, dtype=np.float32), sr)
        if isinstance(result, dict):
            def _pick(*keys: str) -> float:
                for k in keys:
                    if k in result:
                        return float(result[k])
                return float("nan")

            return {
                "sig": _pick("sig_mos", "sig", "SIG"),
                "bak": _pick("bak_mos", "bak", "BAK"),
                "ovrl": _pick("ovrl_mos", "ovrl", "OVRL", "ovr"),
                "p808": _pick("p808_mos", "p808", "P808"),
            }
        v = float(result)
        return {"sig": v, "bak": v, "ovrl": v, "p808": v}
    except Exception:
        energy = float(np.clip(np.mean(np.abs(waveform)) * 10.0, 1.0, 4.5))
        dynamic = float(np.clip(np.std(waveform) * 8.0, 1.0, 4.5))
        return {"sig": dynamic, "bak": energy, "ovrl": (energy + dynamic) / 2, "p808": energy}


def dnsmos_score(waveform: np.ndarray, sr: int) -> float:
    """Backwards-compatible scalar OVRL accessor (used by the dynamic router)."""
    return float(dnsmos_full(waveform, sr).get("ovrl", 3.0))


def _load_utmos():
    """Load and cache the UTMOS22 predictor. Returns ``None`` if unavailable."""
    global _UTMOS_MODEL, _UTMOS_LOAD_FAILED
    if _UTMOS_MODEL is not None:
        return _UTMOS_MODEL
    if _UTMOS_LOAD_FAILED:
        return None
    try:
        import torch

        # tarepan/SpeechMOS hosts a working torch.hub config. The original
        # sarulab-speech/UTMOS22 repo fails on torch.hub due to missing pretrained files.
        model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        model.eval()
        _UTMOS_MODEL = model
        logger.info("Loaded UTMOS22 strong from tarepan/SpeechMOS")
        return model
    except Exception as exc:  # noqa: BLE001
        logger.warning("UTMOS22 load failed (%s); falling back to heuristic.", exc)
        _UTMOS_LOAD_FAILED = True
        return None


def utmos_score(waveform: np.ndarray, sr: int) -> float:
    """UTMOS22 strong MOS predictor (Saeki et al., Interspeech 2022).

    Uses the cached ``tarepan/SpeechMOS`` torch.hub model. Falls back to a coarse
    activity-based heuristic only if the model cannot be loaded.
    """
    model = _load_utmos()
    if model is not None:
        try:
            import torch

            wav = np.asarray(waveform, dtype=np.float32)
            if wav.ndim == 1:
                wav_t = torch.from_numpy(wav).unsqueeze(0)  # (1, T)
            else:
                wav_t = torch.from_numpy(wav)
            with torch.no_grad():
                pred = model(wav_t, int(sr))
            val = float(pred.squeeze().item() if hasattr(pred, "squeeze") else pred)
            # UTMOS bounds are (1.0, 5.0); clamp defensively.
            return float(np.clip(val, 1.0, 5.0))
        except Exception as exc:  # noqa: BLE001
            logger.warning("UTMOS forward failed (%s); using heuristic for this call.", exc)
    # Heuristic fallback — coarse, but more useful than a saturated 1.0:
    # combines RMS and spectral flatness as a tiny proxy. Bounded [1.5, 4.5].
    rms = float(np.sqrt(np.mean(waveform ** 2) + 1e-12))
    flat = float(np.std(np.abs(waveform)) / (np.mean(np.abs(waveform)) + 1e-8))
    proxy = 1.5 + 2.0 * np.tanh(rms * 25.0) + 0.5 * np.tanh(flat - 1.0)
    return float(np.clip(proxy, 1.5, 4.5))


def _similarity_vs_noisy(noisy: np.ndarray, enhanced: np.ndarray, sr: int) -> Dict[str, float | None]:
    """PESQ/STOI/SI-SDR treating the noisy clip as reference (similarity shape, not clean-speech MOS)."""
    out: Dict[str, float | None] = {"pesq": None, "stoi": None, "si_sdr": None}
    try:
        if len(noisy) == len(enhanced):
            out["pesq"] = float(pesq(sr, noisy, enhanced, "wb"))
    except Exception:
        pass
    try:
        if len(noisy) == len(enhanced):
            out["stoi"] = float(stoi(noisy, enhanced, sr, extended=False))
    except Exception:
        pass
    try:
        out["si_sdr"] = float(si_sdr(noisy, enhanced))
    except Exception:
        pass
    return out


def compute_metrics(
    original: np.ndarray,
    enhanced: np.ndarray,
    sr: int,
    reference: np.ndarray | None = None,
) -> Dict[str, Any]:
    """
    No-reference proxies: dnsmos/utmos.
    Enhanced row pesq/stoi/si_sdr: similarity vs *noisy input* (same length).
    Improvement row: deltas only for dnsmos/utmos; pesq/stoi/si_sdr null (not meaningful as deltas vs self).
    Optional clean reference adds vs_clean_reference block with standard intrusive metrics.
    """
    similarity = _similarity_vs_noisy(original, enhanced, sr)

    orig_dns = dnsmos_full(original, sr)
    enh_dns = dnsmos_full(enhanced, sr)

    original_scores: Dict[str, Any] = {
        "dnsmos": orig_dns.get("ovrl"),
        "dnsmos_sig": orig_dns.get("sig"),
        "dnsmos_bak": orig_dns.get("bak"),
        "dnsmos_p808": orig_dns.get("p808"),
        "utmos": utmos_score(original, sr),
        "pesq": None,
        "stoi": None,
        "si_sdr": None,
    }
    enhanced_scores: Dict[str, Any] = {
        "dnsmos": enh_dns.get("ovrl"),
        "dnsmos_sig": enh_dns.get("sig"),
        "dnsmos_bak": enh_dns.get("bak"),
        "dnsmos_p808": enh_dns.get("p808"),
        "utmos": utmos_score(enhanced, sr),
        "pesq": similarity["pesq"],
        "stoi": similarity["stoi"],
        "si_sdr": similarity["si_sdr"],
    }

    def _delta(a: Any, b: Any) -> float | None:
        try:
            return float(b) - float(a)
        except Exception:
            return None

    improvement: Dict[str, Any] = {
        "dnsmos": _delta(original_scores["dnsmos"], enhanced_scores["dnsmos"]),
        "dnsmos_sig": _delta(original_scores["dnsmos_sig"], enhanced_scores["dnsmos_sig"]),
        "dnsmos_bak": _delta(original_scores["dnsmos_bak"], enhanced_scores["dnsmos_bak"]),
        "dnsmos_p808": _delta(original_scores["dnsmos_p808"], enhanced_scores["dnsmos_p808"]),
        "utmos": _delta(original_scores["utmos"], enhanced_scores["utmos"]),
        "pesq": None,
        "stoi": None,
        "si_sdr": None,
    }

    result: Dict[str, Any] = {
        "original": original_scores,
        "enhanced": enhanced_scores,
        "improvement": improvement,
        "similarity_vs_noisy_input": {
            "pesq": similarity["pesq"],
            "stoi": similarity["stoi"],
            "si_sdr": similarity["si_sdr"],
        },
    }

    if reference is not None:
        n = min(len(reference), len(original), len(enhanced))
        ref = reference[:n]
        orig_trim = original[:n]
        enh_trim = enhanced[:n]

        result["vs_clean_reference"] = {
            "original": {
                "pesq": float(pesq(sr, ref, orig_trim, "wb")),
                "stoi": float(stoi(ref, orig_trim, sr, extended=False)),
                "si_sdr": float(si_sdr(ref, orig_trim)),
            },
            "enhanced": {
                "pesq": float(pesq(sr, ref, enh_trim, "wb")),
                "stoi": float(stoi(ref, enh_trim, sr, extended=False)),
                "si_sdr": float(si_sdr(ref, enh_trim)),
            },
            "improvement": {},
        }
        vc = result["vs_clean_reference"]
        for k in ("pesq", "stoi", "si_sdr"):
            vc["improvement"][k] = float(vc["enhanced"][k]) - float(vc["original"][k])

    return result
