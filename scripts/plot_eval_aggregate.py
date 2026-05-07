#!/usr/bin/env python3
"""
Build aggregate figures from batch eval CSVs (one or many merged).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_all_csvs(paths: list[Path]) -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in paths]
    if not dfs:
        raise FileNotFoundError("No CSV inputs")
    out = pd.concat(dfs, axis=0, ignore_index=True)
    if out.duplicated(subset=["file"], keep=False).any() if "file" in out.columns else False:
        out = out.drop_duplicates(subset=["file"], keep="last")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="outputs/eval_runs",
        help="Directory containing per_file_metrics_*.csv",
    )
    parser.add_argument("--csv", action="append", default=[], help="Explicit CSV (repeatable)")
    parser.add_argument("--out", default="outputs/eval_runs/eval_aggregate_dashboard.png")
    parser.add_argument("--html", action="store_true")
    args = parser.parse_args()

    if args.csv:
        csv_paths = [Path(c) for c in args.csv]
    else:
        d = Path(args.input_dir)
        csv_paths = sorted(d.glob("per_file_metrics_*.csv"))

    df = load_all_csvs(csv_paths)
    n_files = len(df)
    sns.set_theme(style="whitegrid", context="talk")

    fig = plt.figure(figsize=(14, 10), constrained_layout=False)
    fig.suptitle(
        f"Aggregated evaluation — {n_files} clips\n"
        f"Sources: {len(csv_paths)} CSV file(s)",
        fontsize=13,
        y=1.02,
    )

    # 3x2 grid
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # 1 Expert counts
    ax1 = fig.add_subplot(gs[0, 0])
    if "expert" in df.columns:
        order = df["expert"].value_counts().index.tolist()
        sns.countplot(
            data=df.assign(_e=df["expert"]),
            y="expert",
            hue="_e",
            order=order,
            ax=ax1,
            palette="viridis",
            legend=False,
        )
    ax1.set_title("Expert selection (count)")
    ax1.set_xlabel("Runs")

    # 2 Δ DNSMOS / Δ UTMOS
    ax2 = fig.add_subplot(gs[0, 1])
    deltas = pd.DataFrame(
        {
            "Δ dnsmos": df["dnsmos_delta"],
            "Δ utmos": df["utmos_delta"],
        }
    )
    deltas_m = deltas.melt(var_name="metric", value_name="value")
    sns.violinplot(
        data=deltas_m,
        x="metric",
        y="value",
        hue="metric",
        ax=ax2,
        palette=["#2ecc71", "#3498db"],
        inner="box",
        legend=False,
    )
    ax2.axhline(0, color="gray", linestyle="--", lw=1)
    ax2.set_title("Per-clip delta (enhanced − original)")
    ax2.set_ylabel("Score delta")

    # 3 Histograms overlaid
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(df["dnsmos_delta"].dropna(), bins=20, alpha=0.6, label="dnsmos_delta", color="#27ae60")
    ax3.hist(df["utmos_delta"].dropna(), bins=20, alpha=0.6, label="utmos_delta", color="#2980b9")
    ax3.axvline(0, color="black", linestyle="--", lw=1)
    ax3.legend()
    ax3.set_title("Distribution of no-reference deltas")
    ax3.set_xlabel("Delta")

    # 4 Similarity vs noisy (box)
    ax4 = fig.add_subplot(gs[1, 1])
    sim = pd.DataFrame(
        {
            "PESQ vs noisy": df["pesq_vs_noisy"],
            "STOI vs noisy": df["stoi_vs_noisy"],
            "SI-SDR vs noisy (dB)": df["si_sdr_vs_noisy"],
        }
    )
    sim_m = sim.melt(var_name="metric", value_name="value")
    sns.boxplot(
        data=sim_m,
        x="metric",
        y="value",
        hue="metric",
        ax=ax4,
        palette="muted",
        legend=False,
    )
    ax4.set_title("Similarity to input (noisy) — spread across clips")
    ax4.tick_params(axis="x", rotation=12)

    # 5 Confidence vs Δ dnsmos
    ax5 = fig.add_subplot(gs[2, 0])
    sns.scatterplot(
        data=df,
        x="confidence",
        y="dnsmos_delta",
        hue="expert",
        ax=ax5,
        alpha=0.85,
        s=60,
    )
    ax5.axhline(0, color="gray", linestyle="--", lw=1)
    ax5.set_title("Policy confidence vs Δ dnsmos")

    # 6 SNR summary vs delta
    ax6 = fig.add_subplot(gs[2, 1])
    if "snr_db" in df.columns:
        sns.scatterplot(data=df, x="snr_db", y="dnsmos_delta", hue="expert", ax=ax6, alpha=0.85, s=60)
        ax6.axhline(0, color="gray", linestyle="--", lw=1)
        ax6.set_title("Dashboard SNR (dB) vs Δ dnsmos")
    else:
        ax6.text(0.5, 0.5, "No snr_db column", ha="center", va="center")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.94)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path.resolve()}")

    if args.html:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig_p = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Expert counts",
                    "Δ dnsmos / Δ utmos (per clip)",
                    "PESQ vs noisy",
                    "SI-SDR vs noisy (dB)",
                ),
            )
            exp = df["expert"].value_counts()
            fig_p.add_trace(go.Bar(x=exp.index.astype(str), y=exp.values, name="count"), row=1, col=1)
            fig_p.add_trace(
                go.Box(y=df["dnsmos_delta"], name="Δ dnsmos"), row=1, col=2
            )
            fig_p.add_trace(go.Box(y=df["utmos_delta"], name="Δ utmos"), row=1, col=2)
            fig_p.add_trace(go.Histogram(x=df["pesq_vs_noisy"], name="PESQ", nbinsx=25), row=2, col=1)
            fig_p.add_trace(go.Histogram(x=df["si_sdr_vs_noisy"], name="SI-SDR", nbinsx=25), row=2, col=2)
            fig_p.update_layout(height=800, title_text=f"Eval aggregate (n={n_files})", showlegend=False)
            html_path = out_path.with_suffix(".html")
            fig_p.write_html(str(html_path), include_plotlyjs="cdn")
            print(f"Wrote {html_path.resolve()}")
        except Exception as exc:  # noqa: BLE001
            print(f"Plotly HTML skipped: {exc}")


if __name__ == "__main__":
    main()
