from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import load_config
from backend.app.pipeline import EnhancementPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to noisy audio")
    parser.add_argument("--reference", default=None, help="Optional clean reference path")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = EnhancementPipeline(config)
    result = pipeline.run(args.input, args.reference)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
