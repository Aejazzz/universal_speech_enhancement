#!/usr/bin/env python3
"""Build train/val manifests with noisy_path + clean_path (+ optional HF parquet ceilings)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.paired_discovery import discover_paired_flacs
from datasets.parquet_enrichment import aggregate_quality_by_sample, load_nonblind_parquet_frames, merge_manifest_with_hf_ceiling


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interspeech-root",
        default="",
        help="Folder containing 'Interspeech 2025 URGENT' (defaults to <dataset_root>/Interspeech 2025 URGENT)",
    )
    parser.add_argument(
        "--dataset-root",
        default="C:/Users/jsm10/OneDrive - Amrita vishwa vidyapeetham/agentic-speech-enhancement/datasets",
    )
    parser.add_argument("--hf-sqa-root", default="", help="HF urgent2025-sqa root (parquet under data/)")
    parser.add_argument("--val-ratio", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="outputs/manifests")
    args = parser.parse_args()

    ds = Path(args.dataset_root)
    urgent = Path(args.interspeech_root) if args.interspeech_root else ds / "Interspeech 2025 URGENT"

    if not urgent.is_dir():
        raise FileNotFoundError(f"Interspeech root not found: {urgent}")

    pairs = discover_paired_flacs(urgent)
    if not pairs:
        raise RuntimeError("No noisy/clean FLAC pairs found.")

    rows = [{"noisy_path": str(a).replace("\\", "/"), "clean_path": str(b).replace("\\", "/")} for a, b in pairs]
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    val_n = max(1, int(len(rows) * args.val_ratio))
    val_rows = rows[:val_n]
    train_rows = rows[val_n:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ceiling = pd.DataFrame()
    hf_root = Path(args.hf_sqa_root) if args.hf_sqa_root else ds / "HF urgent2025-sqa" / "data"
    if hf_root.is_dir():
        pq = load_nonblind_parquet_frames(hf_root)
        ceiling = aggregate_quality_by_sample(pq)

    df_train = merge_manifest_with_hf_ceiling(pd.DataFrame(train_rows), ceiling)
    df_val = merge_manifest_with_hf_ceiling(pd.DataFrame(val_rows), ceiling)

    tr_path = out_dir / "paired_train.csv"
    va_path = out_dir / "paired_val.csv"
    df_train.to_csv(tr_path, index=False)
    df_val.to_csv(va_path, index=False)

    summary = {
        "num_pairs_total": len(rows),
        "num_train": len(df_train),
        "num_val": len(df_val),
        "paired_train_csv": str(tr_path.as_posix()),
        "paired_val_csv": str(va_path.as_posix()),
        "parquet_ceiling_joined": not ceiling.empty,
    }
    (out_dir / "paired_manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
