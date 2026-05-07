#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.audio import load_audio
from backend.app.config import load_config
from backend.app.pipeline import EnhancementPipeline
from backend.app.preprocess import preprocess, soft_limiter
from evaluation.metrics import compute_metrics
from policy_agent.model import ACTIONS


def _to_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _dominance_count(candidates: list[dict[str, Any]], chosen: dict[str, Any] | None) -> int:
    if not chosen:
        return 0
    d = 0
    for c in candidates:
        ge_dns = _to_float(c.get("dnsmos")) >= _to_float(chosen.get("dnsmos"))
        ge_sig = _to_float(c.get("dnsmos_sig")) >= _to_float(chosen.get("dnsmos_sig"))
        ge_bak = _to_float(c.get("dnsmos_bak")) >= _to_float(chosen.get("dnsmos_bak"))
        ge_utm = _to_float(c.get("utmos")) >= _to_float(chosen.get("utmos"))
        strict = (
            _to_float(c.get("dnsmos")) > _to_float(chosen.get("dnsmos"))
            or _to_float(c.get("dnsmos_sig")) > _to_float(chosen.get("dnsmos_sig"))
            or _to_float(c.get("dnsmos_bak")) > _to_float(chosen.get("dnsmos_bak"))
            or _to_float(c.get("utmos")) > _to_float(chosen.get("utmos"))
        )
        if ge_dns and ge_sig and ge_bak and ge_utm and strict:
            d += 1
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Folder containing noisy blind-test files.")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--output-dir", default="outputs/reports/blind900_eval")
    ap.add_argument("--glob", default="*.flac")
    ap.add_argument("--limit", type=int, default=0, help="Optional file cap; 0 means all files.")
    ap.add_argument("--resume", action="store_true", help="Resume from existing summary.csv if present.")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.glob))
    if args.limit and args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No files found in {in_dir} with glob {args.glob}")

    cfg = load_config(args.config)
    pipe = EnhancementPipeline(cfg)

    summary_csv = out_dir / "summary.csv"
    rows: list[dict[str, Any]] = []
    processed: set[str] = set()
    if args.resume and summary_csv.exists() and summary_csv.stat().st_size > 0:
        prev = pd.read_csv(summary_csv)
        rows = prev.to_dict(orient="records")
        processed = {str(x) for x in prev["file"].tolist() if isinstance(x, str)}
        files = [p for p in files if p.name not in processed]
        print(f"Resuming: loaded {len(rows)} previous rows, remaining files: {len(files)}")
    flush_every = 10
    per_file_json = out_dir / "per_file.jsonl"
    if per_file_json.exists() and not args.resume:
        per_file_json.unlink()

    for i, p in enumerate(tqdm(files, desc="Blind batch", unit="file"), start=1):
        raw_wave, sr = load_audio(str(p), cfg.system.sample_rate)
        wave, _pre = preprocess(raw_wave, sr)

        distortion = pipe.distortion_model.predict(wave, sr).to(pipe.device).unsqueeze(0)
        wav_tensor = torch.tensor(wave, dtype=torch.float32, device=pipe.device).unsqueeze(0)
        with torch.amp.autocast(
            device_type="cuda",
            enabled=cfg.system.mixed_precision and pipe.device == "cuda",
        ):
            policy_output = pipe.policy(wav_tensor, distortion)

        if cfg.system.dynamic_routing:
            expert_name, chosen_strength, enhanced, candidates, reason = pipe._dynamic_select(
                wave, sr, float(policy_output.strength)
            )
        else:
            expert_name = ACTIONS[policy_output.expert_idx]
            chosen_strength = float(policy_output.strength)
            enhanced = pipe.experts[expert_name].enhance(wave, sr, chosen_strength)
            candidates = []
            reason = "dynamic_routing disabled"

        if policy_output.refine and expert_name != "BYPASS":
            enhanced = pipe.experts[expert_name].enhance(enhanced, sr, min(1.0, chosen_strength + 0.1))
        enhanced = soft_limiter(np.asarray(enhanced, dtype=np.float32), ceiling_db=-1.0)

        metrics = compute_metrics(wave, enhanced, sr, reference=None)

        chosen = None
        if candidates:
            for c in candidates:
                if c.get("expert") == expert_name and abs(_to_float(c.get("strength")) - chosen_strength) < 1e-6:
                    chosen = c
                    break
        if chosen is None:
            chosen = {
                "expert": expert_name,
                "strength": chosen_strength,
                "rank_score": float("nan"),
                "dnsmos": metrics["enhanced"].get("dnsmos"),
                "dnsmos_sig": metrics["enhanced"].get("dnsmos_sig"),
                "dnsmos_bak": metrics["enhanced"].get("dnsmos_bak"),
                "utmos": metrics["enhanced"].get("utmos"),
            }
            if not candidates:
                candidates = [chosen]

        top = max(candidates, key=lambda c: _to_float(c.get("rank_score"), -1e9))
        chosen_is_top = (
            top.get("expert") == expert_name
            and abs(_to_float(top.get("strength")) - chosen_strength) < 1e-6
        )

        row = {
            "file": p.name,
            "chosen_expert": expert_name,
            "chosen_strength": round(chosen_strength, 4),
            "top_rank_expert": top.get("expert"),
            "top_rank_strength": round(_to_float(top.get("strength")), 4),
            "chosen_is_top_rank": bool(chosen_is_top),
            "candidates": len(candidates),
            "rank_score": round(_to_float(chosen.get("rank_score")), 6),
            "dnsmos_delta": round(_to_float(metrics["improvement"].get("dnsmos")), 6),
            "sig_delta": round(_to_float(metrics["improvement"].get("dnsmos_sig")), 6),
            "bak_delta": round(_to_float(metrics["improvement"].get("dnsmos_bak")), 6),
            "utmos_delta": round(_to_float(metrics["improvement"].get("utmos")), 6),
            "pareto_dominated_count": _dominance_count(candidates, chosen),
            "policy_advice_expert": ACTIONS[policy_output.expert_idx],
            "policy_confidence": round(float(policy_output.confidence), 6),
            "decision_reason": reason,
        }
        rows.append(row)

        with per_file_json.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

        if i % flush_every == 0 or i == len(files):
            pd.DataFrame(rows).to_csv(summary_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(summary_csv, index=False)
    df.to_json(out_dir / "summary.json", orient="records", indent=2)

    leaderboard = (
        df.groupby("chosen_expert", dropna=False)
        .agg(
            files=("file", "count"),
            mean_dnsmos_delta=("dnsmos_delta", "mean"),
            mean_sig_delta=("sig_delta", "mean"),
            mean_bak_delta=("bak_delta", "mean"),
            mean_utmos_delta=("utmos_delta", "mean"),
            top_rank_match_rate=("chosen_is_top_rank", "mean"),
        )
        .sort_values(["files", "mean_dnsmos_delta"], ascending=[False, False])
        .reset_index()
    )
    leaderboard.to_csv(out_dir / "leaderboard_experts.csv", index=False)

    top_gain = df.sort_values("dnsmos_delta", ascending=False).head(20)
    top_gain.to_csv(out_dir / "top20_dnsmos_gain.csv", index=False)

    with (out_dir / "report.md").open("w", encoding="utf-8") as fh:
        fh.write("# Blind-Test Batch Evaluation (Full)\n\n")
        fh.write(f"- Files processed: **{len(df)}**\n")
        fh.write(f"- Chosen-is-top-rank rate: **{df['chosen_is_top_rank'].mean()*100:.2f}%**\n")
        fh.write(f"- Mean DNSMOS delta: **{df['dnsmos_delta'].mean():.4f}**\n")
        fh.write(f"- Mean UTMOS delta: **{df['utmos_delta'].mean():.4f}**\n")
        fh.write(f"- Pareto dominated selections: **{int((df['pareto_dominated_count'] > 0).sum())}**\n\n")
        fh.write("## Expert Leaderboard\n\n")
        fh.write(leaderboard.to_markdown(index=False))
        fh.write("\n\n## Top 20 DNSMOS Gains\n\n")
        fh.write(top_gain[["file", "chosen_expert", "chosen_strength", "dnsmos_delta", "sig_delta", "bak_delta", "utmos_delta"]].to_markdown(index=False))
        fh.write("\n")

    print(f"Wrote {out_dir / 'summary.csv'}")
    print(f"Wrote {out_dir / 'leaderboard_experts.csv'}")
    print(f"Wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()

