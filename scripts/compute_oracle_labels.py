#!/usr/bin/env python3
"""
Oracle routing labels via per-(expert, strength) composite quality search vs the clean reference.

Composite score = 0.6 * PESQ_wb + 0.3 * (STOI * 5) + 0.1 * tanh(SI_SDR / 15)
   (PESQ_wb in [1.0, 4.5]; STOI in [0, 1] scaled to [0, 5]; tanh keeps SI-SDR contribution bounded.)

For each non-BYPASS expert we sweep `--strengths` (default 0.6, 1.0) and keep the best (expert, strength)
combination. BYPASS gets a single score on the noisy waveform itself. The oracle action+strength are
derived from the global argmax — strength becomes a meaningful regression target instead of a constant.

Output columns added to the manifest:
  oracle_action          int       index into ACTIONS for the winning expert
  oracle_strength        float     the strength that produced the winning score (0.0 for BYPASS)
  oracle_refine          float     1.0 if a non-BYPASS expert won, else 0.0
  oracle_best_expert     str       human-readable expert name
  oracle_best_score      float     value of the composite score
  oracle_scores_json     str       JSON of all (expert, strength) -> score
Resilient: bad files are skipped (logged in <output>.skipped.txt) and rows are flushed every 25 files
so a crash never loses prior work; resume is automatic if the output CSV already exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from pesq import pesq
from pystoi import stoi as stoi_fn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.config import load_config
from datasets.loader import ACTIONS
from enhancement_experts.factory import build_experts
from evaluation.metrics import si_sdr


def _trim_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n].astype(np.float32), b[:n].astype(np.float32)


def composite_score(ref: np.ndarray, est: np.ndarray, sr: int) -> float:
    """Higher is better — bounded combination of PESQ-wb, STOI, SI-SDR."""
    ref_t, est_t = _trim_pair(ref, est)
    if len(ref_t) < int(sr * 0.25):
        return -1e6
    try:
        p = float(pesq(sr, ref_t, est_t, "wb"))
    except Exception:  # noqa: BLE001
        p = 1.0  # PESQ floor
    try:
        s_intel = float(stoi_fn(ref_t, est_t, sr, extended=False))
    except Exception:  # noqa: BLE001
        s_intel = 0.0
    try:
        sdr = si_sdr(ref_t, est_t)
        if not np.isfinite(sdr):
            sdr = -50.0
    except Exception:  # noqa: BLE001
        sdr = -50.0
    return float(0.6 * p + 0.3 * (s_intel * 5.0) + 0.1 * np.tanh(sdr / 15.0))


def _enhance_aligned(expert, noisy: np.ndarray, sr: int, strength: float) -> np.ndarray:
    out = expert.enhance(noisy.copy(), sr, strength)
    out = np.asarray(out, dtype=np.float32)
    if len(out) != len(noisy):
        if len(out) > len(noisy):
            out = out[: len(noisy)].copy()
        else:
            out = np.pad(out, (0, len(noisy) - len(out)))
    return out


def choose_oracle(
    noisy: np.ndarray,
    clean: np.ndarray,
    sr: int,
    experts: dict,
    strengths: list[float],
) -> tuple[int, float, float, float, dict[str, float]]:
    noisy_t, clean_t = _trim_pair(noisy.astype(np.float32), clean.astype(np.float32))
    scores: dict[str, float] = {}

    scores["BYPASS@0.0"] = composite_score(clean_t, noisy_t, sr)

    for name in ["DeepFilterNet3", "ResembleEnhance", "MossFormer2"]:
        expert = experts[name]
        for strength in strengths:
            try:
                est = _enhance_aligned(expert, noisy_t, sr, float(strength))
                scores[f"{name}@{strength:.2f}"] = composite_score(clean_t, est, sr)
            except Exception as exc:  # noqa: BLE001
                scores[f"{name}@{strength:.2f}"] = -1e6
                scores[f"{name}@{strength:.2f}_err"] = str(exc)[:200]

    # Pick best non-error key.
    valid = {k: v for k, v in scores.items() if isinstance(v, (int, float)) and v > -1e5}
    best_key = max(valid.keys(), key=lambda k: valid[k])
    expert_part, strength_part = best_key.split("@")
    best_strength = float(strength_part)
    idx = ACTIONS.index(expert_part)
    refine = 0.0 if expert_part == "BYPASS" else 1.0
    return idx, best_strength, refine, float(valid[best_key]), scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--strengths",
        type=str,
        default="0.6,1.0",
        help="Comma-separated strengths to sweep per non-BYPASS expert (default 0.6,1.0).",
    )
    args = parser.parse_args()
    strengths = [float(x.strip()) for x in args.strengths.split(",") if x.strip()]
    if not strengths:
        strengths = [1.0]

    cfg = load_config(args.config)
    device = "cuda" if cfg.system.device == "cuda" and torch.cuda.is_available() else "cpu"
    sr = args.sample_rate

    df = pd.read_csv(args.manifest)
    if "clean_path" not in df.columns:
        raise ValueError("manifest must include clean_path for oracle labeling")
    subset = df.iloc[args.start_index :]
    if args.max_files > 0:
        subset = subset.iloc[: args.max_files]

    experts = build_experts(device, mossformer_checkpoint=cfg.system.mossformer_checkpoint)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_log = out_path.with_suffix(".skipped.txt")

    # Resume support: if the output CSV already exists, append and skip rows already labeled
    # (matched by noisy_path string).
    existing_df: pd.DataFrame | None = None
    done_keys: set[str] = set()
    if out_path.exists():
        try:
            existing_df = pd.read_csv(out_path)
            done_keys = {str(p) for p in existing_df["noisy_path"].tolist()}
            print(f"Resuming: found {len(done_keys)} already-labeled rows in {out_path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: could not read existing {out_path} ({exc}); starting fresh", flush=True)
            existing_df = None
            done_keys = set()

    flush_every = 25
    pending: list[dict] = []
    written = 0
    skipped = 0

    def _flush() -> None:
        nonlocal pending, written, existing_df
        if not pending:
            return
        df_new = pd.DataFrame(pending)
        if existing_df is not None and not existing_df.empty:
            combined = pd.concat([existing_df, df_new], ignore_index=True)
        else:
            combined = df_new
        combined.to_csv(out_path, index=False)
        existing_df = combined
        written += len(pending)
        pending = []

    try:
        for _, row in tqdm(subset.iterrows(), total=len(subset)):
            np_path_str = str(row["noisy_path"])
            if np_path_str in done_keys:
                continue
            np_path = Path(np_path_str)
            cp_path = Path(str(row["clean_path"]))
            if not np_path.is_file() or not cp_path.is_file():
                skipped += 1
                continue
            try:
                noisy, _ = librosa.load(np_path.as_posix(), sr=sr, mono=True)
                clean, _ = librosa.load(cp_path.as_posix(), sr=sr, mono=True)
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                with skipped_log.open("a", encoding="utf-8") as fh:
                    fh.write(f"LOAD\t{np_path_str}\t{cp_path}\t{exc}\n")
                continue
            try:
                oid, stren, refin, best_score, scores = choose_oracle(noisy, clean, sr, experts, strengths)
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                with skipped_log.open("a", encoding="utf-8") as fh:
                    fh.write(f"SCORE\t{np_path_str}\t{cp_path}\t{exc}\n")
                continue

            rec = row.to_dict()
            rec["oracle_action"] = int(oid)
            rec["oracle_strength"] = float(stren)
            rec["oracle_refine"] = float(refin)
            rec["oracle_best_expert"] = ACTIONS[oid]
            rec["oracle_best_score"] = float(best_score)
            rec["oracle_scores_json"] = json.dumps(
                {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in scores.items()}
            )
            pending.append(rec)

            if len(pending) >= flush_every:
                _flush()
    finally:
        _flush()

    print(
        f"Wrote {out_path} with {written} new rows (total {len(existing_df) if existing_df is not None else 0}); "
        f"skipped {skipped}. See {skipped_log} for skipped file diagnostics.",
        flush=True,
    )


if __name__ == "__main__":
    main()
