#!/usr/bin/env python3
"""
Demo-day smoke test.

Runs the full ``EnhancementPipeline`` on a handful of real validation clips
(paired noisy/clean) and prints a one-line summary per clip plus an aggregate
routing distribution. No external manifests required.

Usage:
    python scripts/demo_smoke.py [n_samples] [--config configs/base.yaml]
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import load_config  # noqa: E402
from backend.app.pipeline import EnhancementPipeline  # noqa: E402


VAL_NOISY = Path(
    "C:/Users/jsm10/OneDrive - Amrita vishwa vidyapeetham/agentic-speech-enhancement/"
    "datasets/Interspeech 2025 URGENT/official validation set/validation.noisy/noisy"
)
VAL_CLEAN = Path(
    "C:/Users/jsm10/OneDrive - Amrita vishwa vidyapeetham/agentic-speech-enhancement/"
    "datasets/Interspeech 2025 URGENT/official validation set/validation.clean/clean"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=8, help="Number of clips to run")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not VAL_NOISY.exists():
        raise SystemExit(f"Validation noisy folder not found: {VAL_NOISY}")

    cfg = load_config(args.config)
    print(f"[demo] device={cfg.system.device} ckpt={cfg.system.policy_checkpoint}")
    pipe = EnhancementPipeline(cfg)
    print(f"[demo] pipeline ready on {pipe.device}")

    files = sorted(VAL_NOISY.glob("*.flac"))
    rng = random.Random(args.seed)
    rng.shuffle(files)
    files = files[: args.n]

    routing_counts: Counter[str] = Counter()
    for i, p in enumerate(files, 1):
        clean = VAL_CLEAN / p.name
        ref = str(clean) if clean.exists() else None
        result = pipe.run(str(p), reference_path=ref)
        routing = result["routing"]
        improv = result["metrics"].get("improvement", {})
        routing_counts[routing["expert"]] += 1

        def _f(v: object) -> float:
            try:
                return float(v) if v is not None else 0.0
            except Exception:
                return 0.0

        print(
            f"[demo {i:02d}] {p.name:<22s} -> {routing['expert']:<19s} "
            f"s={routing['strength']:.2f} d_dnsmos={_f(improv.get('dnsmos')):+0.3f} "
            f"d_sig={_f(improv.get('dnsmos_sig')):+0.3f} d_bak={_f(improv.get('dnsmos_bak')):+0.3f} "
            f"d_utmos={_f(improv.get('utmos')):+0.3f}"
        )

    print()
    print(f"[demo] routing distribution over {sum(routing_counts.values())} clips: {dict(routing_counts)}")


if __name__ == "__main__":
    main()
