#!/usr/bin/env python3
"""
Re-derive ``oracle_action`` / ``oracle_strength`` / ``oracle_refine`` from an existing
``oracle_scores_json`` column with explicit tie-breaking rules — no expert re-runs.

Why: many experts return identity output for a given clip (missing checkpoint, no-op
fallback). When several "@<strength>" entries match BYPASS exactly, the original
``max()`` picked the first encountered key, polluting the label set with bogus expert
wins. We instead require a non-trivial improvement margin over BYPASS to pick a
non-BYPASS expert. Otherwise we choose BYPASS — the "do no harm" baseline.

Usage:
  python scripts/rederive_oracle_labels.py --in outputs/manifests/paired_train_oracle.csv \
                                            --out outputs/manifests/paired_train_oracle.csv \
                                            --margin 0.005
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ACTIONS = ["DeepFilterNet3", "ResembleEnhance", "MossFormer2", "BYPASS"]


def rederive_row(scores: dict, margin: float) -> tuple[int, float, float, str]:
    bypass = float(scores.get("BYPASS@0.0", 0.0))
    candidates: dict[str, tuple[float, float]] = {}  # expert -> (best_score, strength)
    for key, value in scores.items():
        if not isinstance(value, (int, float)) or key == "BYPASS@0.0":
            continue
        try:
            expert, strength_str = key.split("@", 1)
            strength = float(strength_str)
        except ValueError:
            continue
        prev = candidates.get(expert)
        if prev is None or value > prev[0]:
            candidates[expert] = (float(value), strength)

    best_expert = "BYPASS"
    best_strength = 0.0
    best_score = bypass
    for expert, (score, strength) in candidates.items():
        if score > best_score + margin:
            best_score = score
            best_expert = expert
            best_strength = strength
    idx = ACTIONS.index(best_expert)
    refine = 0.0 if best_expert == "BYPASS" else 1.0
    return idx, best_strength, refine, best_expert


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.005,
        help="Minimum composite-score improvement over BYPASS required to pick a non-BYPASS expert.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.inp)
    if "oracle_scores_json" not in df.columns:
        raise SystemExit("input CSV must contain oracle_scores_json")

    new_actions: list[int] = []
    new_strengths: list[float] = []
    new_refines: list[float] = []
    new_experts: list[str] = []

    for _, row in df.iterrows():
        try:
            scores = json.loads(row["oracle_scores_json"])
        except Exception:
            scores = {}
        idx, stren, refin, expert = rederive_row(scores, args.margin)
        new_actions.append(idx)
        new_strengths.append(stren)
        new_refines.append(refin)
        new_experts.append(expert)

    df["oracle_action"] = new_actions
    df["oracle_strength"] = new_strengths
    df["oracle_refine"] = new_refines
    df["oracle_best_expert"] = new_experts

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: rows={len(df)}")
    print("New action distribution:")
    print(df["oracle_best_expert"].value_counts())


if __name__ == "__main__":
    main()
