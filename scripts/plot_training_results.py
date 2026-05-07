#!/usr/bin/env python3
"""Plot policy training curves from CSV reports (matplotlib + optional Plotly HTML)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="outputs/reports/training_history.csv")
    parser.add_argument("--performance", default="outputs/reports/model_performance.csv")
    parser.add_argument("--out-dir", default="outputs/reports")
    parser.add_argument("--plotly-html", action="store_true", help="Also write interactive Plotly HTML")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path = Path(args.history)
    perf_path = Path(args.performance)

    if not hist_path.exists():
        raise FileNotFoundError(f"Missing {hist_path}. Run training first.")

    df = pd.read_csv(hist_path)
    sns.set_theme(style="whitegrid", context="talk", palette="muted")
    plt.rcParams["figure.dpi"] = 140

    # Loss curves
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["epoch"], df["train_loss"], marker="o", label="Train loss", linewidth=2)
    ax.plot(df["epoch"], df["val_loss"], marker="s", label="Val loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Policy training — loss vs epoch")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "training_loss_curves.png", bbox_inches="tight")
    plt.close(fig)

    # Accuracy
    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    ax2.bar(df["epoch"].astype(str), df["val_accuracy"], color=sns.color_palette("muted")[2], alpha=0.85)
    ax2.axhline(df["val_accuracy"].max(), color="crimson", linestyle="--", alpha=0.7, label="Max val accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation accuracy")
    ax2.set_title("Validation accuracy (routing vs weak labels)")
    ax2.set_ylim(0, 1.02)
    ax2.legend(loc="lower right")
    fig2.tight_layout()
    fig2.savefig(out_dir / "training_val_accuracy.png", bbox_inches="tight")
    plt.close(fig2)

    summary_lines = []
    if perf_path.exists():
        perf = pd.read_csv(perf_path)
        best_vl = perf["best_val_loss"].iloc[0]
        fv = perf["final_val_accuracy"].iloc[0]
        summary_lines = [
            f"Best val loss: {best_vl:.6f}",
            f"Reported final val accuracy: {fv:.4f}",
        ]

    # Combined dashboard
    fig3, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(df["epoch"], df["train_loss"], "o-", label="Train")
    axes[0].plot(df["epoch"], df["val_loss"], "s-", label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss")

    axes[1].fill_between(df["epoch"], df["val_accuracy"], alpha=0.3)
    axes[1].plot(df["epoch"], df["val_accuracy"], "o-", color="forestgreen")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val accuracy")
    axes[1].set_ylim(0.95, 1.0)
    axes[1].set_title("Validation accuracy")

    subtitle = "\n".join(summary_lines) if summary_lines else ""
    fig3.suptitle(f"Universal Speech Enhancement — policy training summary\n{subtitle}", fontsize=12, y=1.06)
    fig3.tight_layout()
    fig3.savefig(out_dir / "training_dashboard.png", bbox_inches="tight")
    plt.close(fig3)

    if args.plotly_html:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig_p = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Loss", "Val accuracy"),
            )
            fig_p.add_trace(
                go.Scatter(x=df["epoch"], y=df["train_loss"], mode="lines+markers", name="Train loss"),
                row=1,
                col=1,
            )
            fig_p.add_trace(
                go.Scatter(x=df["epoch"], y=df["val_loss"], mode="lines+markers", name="Val loss"),
                row=1,
                col=1,
            )
            fig_p.add_trace(
                go.Scatter(x=df["epoch"], y=df["val_accuracy"], mode="lines+markers", name="Val accuracy"),
                row=1,
                col=2,
            )
            fig_p.update_layout(
                title_text="Policy training (interactive)",
                height=440,
                showlegend=True,
            )
            html_path = out_dir / "training_dashboard.html"
            fig_p.write_html(str(html_path), include_plotlyjs="cdn")
            print(f"Wrote {html_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Plotly export skipped: {exc}")

    print(f"Wrote PNGs under {out_dir}:")
    for name in (
        "training_loss_curves.png",
        "training_val_accuracy.png",
        "training_dashboard.png",
    ):
        print(f"  - {out_dir / name}")


if __name__ == "__main__":
    main()
