from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def collect_audio_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for ext in ("*.wav", "*.mp3", "*.flac"):
        files.extend(root.rglob(ext))
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val/test manifests from local dataset.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-dir", default="outputs/manifests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = collect_audio_files(root)
    if not files:
        raise ValueError(f"No audio files found under: {root}")

    rnd = random.Random(args.seed)
    rnd.shuffle(files)

    n = len(files)
    train_end = int(n * args.train_ratio)
    val_end = train_end + int(n * args.val_ratio)
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    def write_manifest(paths: list[Path], split: str) -> str:
        df = pd.DataFrame({"noisy_path": [str(p) for p in paths], "split": [split] * len(paths)})
        out_path = out_dir / f"{split}.csv"
        df.to_csv(out_path, index=False)
        return str(out_path)

    paths = {
        "train_manifest": write_manifest(train_files, "train"),
        "val_manifest": write_manifest(val_files, "val"),
        "test_manifest": write_manifest(test_files, "test"),
        "num_train": len(train_files),
        "num_val": len(val_files),
        "num_test": len(test_files),
        "num_total": n,
    }
    with (out_dir / "manifest_summary.json").open("w", encoding="utf-8") as f:
        json.dump(paths, f, indent=2)
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
