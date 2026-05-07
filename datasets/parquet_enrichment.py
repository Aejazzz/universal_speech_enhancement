"""Aggregate HuggingFace URGENT parquet shards → per-utterance scalar features (optional training priors)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from datasets.paired_discovery import parquet_sample_id_to_flac


def load_nonblind_parquet_frames(hf_data_dir: Path) -> pd.DataFrame:
    paths = sorted((hf_data_dir / "nonblind_test").glob("nonblind_test-*.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def aggregate_quality_by_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Max objective scores across submissions for each utterance (ceiling diagnostic)."""
    if df.empty:
        return df
    df = df.copy()
    df["_flac"] = df["sample_id"].map(parquet_sample_id_to_flac)
    df = df.dropna(subset=["_flac"])
    grp = df.groupby("_flac", dropna=True)
    agg_cols = ["dnsmos_ovrl", "pesq", "utmos", "sdr"]
    present = [c for c in agg_cols if c in df.columns]
    if not present:
        return pd.DataFrame()
    out = grp[present].max().rename(columns=lambda c: f"hf_ceiling_{c}")
    return out.reset_index()


def merge_manifest_with_hf_ceiling(manifest_df: pd.DataFrame, ceiling: pd.DataFrame) -> pd.DataFrame:
    m = manifest_df.copy()
    if ceiling.empty:
        for c in ("hf_ceiling_dnsmos_ovrl", "hf_ceiling_pesq", "hf_ceiling_utmos", "hf_ceiling_sdr"):
            m[c] = pd.NA
        return m
    slim = ceiling.rename(columns={"_flac": "noisy_basename"})
    slim["noisy_basename"] = slim["noisy_basename"].str.lower()

    m["noisy_basename"] = m["noisy_path"].apply(lambda p: Path(str(p)).name.lower())
    cols = ["noisy_basename"] + [c for c in slim.columns if c.startswith("hf_ceiling")]
    merged = m.merge(slim[cols], on="noisy_basename", how="left").drop(columns=["noisy_basename"], errors="ignore")
    return merged
