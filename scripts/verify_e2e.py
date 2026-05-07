#!/usr/bin/env python3
"""
End-to-end smoke test for the inference pipeline.

* Loads ``configs/base.yaml``
* Instantiates ``EnhancementPipeline`` (which auto-loads ``checkpoints/policy_best.pt``)
* Pulls a few noisy clips from the val manifest
* Runs full pipeline (analyze -> route -> enhance -> metrics -> plots)
* Prints a compact routing/metrics summary per clip and the agg routing distribution
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.config import load_config  # noqa: E402
from backend.app.pipeline import EnhancementPipeline  # noqa: E402


def main() -> None:
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    cfg = load_config("configs/base.yaml")
    print(f"[verify] device={cfg.system.device} policy_ckpt={cfg.system.policy_checkpoint}")
    pipe = EnhancementPipeline(cfg)
    print(f"[verify] pipeline ready on {pipe.device}")

    manifest = pd.read_csv("outputs/manifests/paired_val_oracle.csv")
    sample_df = manifest.sample(n=min(n_samples, len(manifest)), random_state=0)

    routing_counts: Counter[str] = Counter()
    rows: list[dict] = []
    for i, (_, row) in enumerate(sample_df.iterrows()):
        noisy_path = str(row["noisy_path"])
        clean_path = str(row.get("clean_path") or "")
        oracle = row.get("oracle_best_expert", "?")
        result = pipe.run(noisy_path, reference_path=clean_path or None)
        routing = result["routing"]
        improvement = result["metrics"].get("improvement", {})
        chosen = routing["expert"]
        routing_counts[chosen] += 1
        def _f(v: object, default: float = 0.0) -> float:
            try:
                return float(v) if v is not None else default
            except Exception:
                return default

        print(
            f"[verify {i+1:02d}] oracle={oracle:<14s} chosen={chosen:<14s} "
            f"strength={routing['strength']:.2f} confidence={routing['confidence']:.2f} "
            f"d_pesq={_f(improvement.get('pesq')):+0.3f} d_stoi={_f(improvement.get('stoi')):+0.3f} "
            f"d_dnsmos={_f(improvement.get('dnsmos')):+0.3f} "
            f"d_utmos={_f(improvement.get('utmos')):+0.3f}"
        )
        rows.append(
            {
                "noisy": noisy_path,
                "oracle": oracle,
                "chosen": chosen,
                "agree_with_oracle": chosen == oracle,
                "strength": routing["strength"],
                "confidence": routing["confidence"],
                **{f"improv_{k}": v for k, v in improvement.items()},
            }
        )

    summary = pd.DataFrame(rows)
    out_path = Path("outputs/reports/e2e_verify.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print()
    print(f"[verify] routing distribution (over {len(summary)} clips): {dict(routing_counts)}")
    print(f"[verify] agreement with oracle: {summary['agree_with_oracle'].mean()*100:.1f}%")
    print(f"[verify] wrote {out_path}")


if __name__ == "__main__":
    main()
