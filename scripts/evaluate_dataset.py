#!/usr/bin/env python3
"""
Batch-run the enhancement pipeline on local dataset files, collect metrics, and aggregate summary stats.

Supports:
  - manifest CSV (train/val/test from prepare_dataset.py)
  - or recursive scan under dataset_root
  - optional filter to noisy clips only (recommended when manifest mixes clean + noisy paths)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.config import load_config
from backend.app.pipeline import EnhancementPipeline


def collect_audio_files(root: Path, limit: int | None = None) -> list[Path]:
    files: list[Path] = []
    for ext in ("*.wav", "*.mp3", "*.flac"):
        files.extend(root.rglob(ext))
    files.sort()
    if limit is not None:
        return files[:limit]
    return files


def paths_from_manifest(manifest_csv: Path, limit: int | None) -> list[Path]:
    df = pd.read_csv(manifest_csv)
    col = "noisy_path" if "noisy_path" in df.columns else df.columns[0]
    paths = [Path(str(p)) for p in df[col].tolist()]
    if limit is not None:
        paths = paths[:limit]
    return paths


def is_under_noisy_folder(path: Path) -> bool:
    """Prefer clips under a literal 'noisy' directory segment (avoids blind/clean mixes in manifests)."""
    parts = path.as_posix().lower().split("/")
    return any(p == "noisy" for p in parts)


def flat_metrics_row(file_path: str, result: dict[str, Any]) -> dict[str, Any]:
    r = result["routing"]
    m = result["metrics"]
    orig = m["original"]
    enh = m["enhanced"]
    imp = m["improvement"]
    sim = m["similarity_vs_noisy_input"]
    ds = r.get("distortion_summary") or {}
    row: dict[str, Any] = {
        "run_id": result["id"],
        "file": file_path,
        "expert": r["expert"],
        "strength": r["strength"],
        "refine": r["refine"],
        "confidence": r["confidence"],
        "reason": r["reason"],
        "snr_db": ds.get("snr_db"),
        "reverb": ds.get("reverb"),
        "clip": ds.get("clip"),
        "noise_level": ds.get("noise_level"),
        "prob_DeepFilterNet3": r["probabilities"].get("DeepFilterNet3"),
        "prob_ResembleEnhance": r["probabilities"].get("ResembleEnhance"),
        "prob_MossFormer2": r["probabilities"].get("MossFormer2"),
        "prob_BYPASS": r["probabilities"].get("BYPASS"),
        "dnsmos_original": orig.get("dnsmos"),
        "dnsmos_enhanced": enh.get("dnsmos"),
        "dnsmos_delta": imp.get("dnsmos"),
        "utmos_original": orig.get("utmos"),
        "utmos_enhanced": enh.get("utmos"),
        "utmos_delta": imp.get("utmos"),
        "pesq_vs_noisy": enh.get("pesq"),
        "stoi_vs_noisy": enh.get("stoi"),
        "si_sdr_vs_noisy": enh.get("si_sdr"),
        "enhanced_audio_rel": result["output_audio_path"],
    }
    return row


def summarize_numeric(df: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) == 0:
            summary[c] = {"count": 0.0}
            continue
        summary[c] = {
            "count": float(len(s)),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "median": float(s.median()),
            "min": float(s.min()),
            "max": float(s.max()),
            # Win rates for deltas
            **(
                {"pct_positive": float((s > 0).mean() * 100.0)}
                if "delta" in c or "_delta" in c
                else {}
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--manifest", default="", help="CSV with noisy_path column (e.g. outputs/manifests/test.csv)")
    parser.add_argument("--max-files", type=int, default=0, help="Max files (0 = all in manifest/list)")
    parser.add_argument(
        "--include-clean-paths",
        action="store_true",
        help="Also evaluate manifest paths not under a 'noisy' folder (default: noisy-only)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/eval_runs",
        help="Writes per-run CSV + summary JSON here",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        manifest = Path(args.manifest)
        paths = paths_from_manifest(manifest, None)
        source_note = str(manifest)
    else:
        root = Path(cfg.system.dataset_root)
        paths = collect_audio_files(root, None)
        source_note = str(root)

    if not args.include_clean_paths:
        before_len = len(paths)
        paths = [p for p in paths if is_under_noisy_folder(p)]
        print(f"Noisy-path filter: {before_len} -> {len(paths)} files", flush=True)

    if args.max_files > 0:
        paths = paths[: args.max_files]
        print(f"Limited to first {len(paths)} files after filter", flush=True)

    if not paths:
        raise SystemExit("No files to evaluate.")

    pipeline = EnhancementPipeline(cfg)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for idx, path in enumerate(paths):
        fp = path.as_posix()
        print(f"[{idx + 1}/{len(paths)}] {fp}", flush=True)
        try:
            if not path.exists():
                failures.append({"file": fp, "error": "missing_on_disk"})
                continue
            result = pipeline.run(str(path))
            rows.append(flat_metrics_row(fp, result))
        except Exception as exc:  # noqa: BLE001
            failures.append({"file": fp, "error": repr(exc)})
            continue

    tag = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"per_file_metrics_{tag}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    failures_path = out_dir / f"failures_{tag}.json"
    Path(failures_path).write_text(json.dumps(failures, indent=2), encoding="utf-8")

    df = pd.DataFrame(rows)
    metric_cols = [
        "dnsmos_delta",
        "utmos_delta",
        "confidence",
        "pesq_vs_noisy",
        "stoi_vs_noisy",
        "si_sdr_vs_noisy",
    ]
    expert_counts = df["expert"].value_counts(dropna=False).to_dict() if len(df) else {}
    numeric_summary = summarize_numeric(df, metric_cols)

    summary = {
        "source_manifest_or_root": source_note,
        "noisy_paths_only": not args.include_clean_paths,
        "num_requested": len(paths),
        "num_success": len(rows),
        "num_failed": len(failures),
        "output_csv": str(csv_path.as_posix()),
        "failures_json": str(failures_path.as_posix()),
        "expert_selection_counts": expert_counts,
        "numeric_summary": numeric_summary,
    }

    summary_path = out_dir / f"evaluation_summary_{tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nSaved: {csv_path}", flush=True)
    print(f"Saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
