"""Discover noisy ↔ clean pairs under URGENT-style layouts (*/*.noisy/noisy/*.flac)."""

from __future__ import annotations

import re
from pathlib import Path

PAIR_SUFFIX_MAP = (
    ("nonblind_test.noisy", "nonblind_test.clean"),
    ("validation.noisy", "validation.clean"),
)


def _clean_path_for_noisy(noisy_flac: Path) -> Path | None:
    parts = list(noisy_flac.parts)
    try:
        i_noise_subdir = parts.index("noisy")
    except ValueError:
        return None
    if i_noise_subdir < 1:
        return None
    pack = parts[i_noise_subdir - 1]
    for noisy_pack, clean_pack in PAIR_SUFFIX_MAP:
        if pack == noisy_pack:
            rebuilt = parts[:]
            rebuilt[i_noise_subdir - 1] = clean_pack
            rebuilt[i_noise_subdir] = "clean"
            cand = Path(*rebuilt)
            return cand if cand.is_file() else None
    return None


def discover_paired_flacs(urgent_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for noisy in urgent_root.rglob("*.flac"):
        ps = noisy.parts
        ps_l = tuple(p.lower() for p in ps)
        if len(ps_l) < 3 or ps_l[-2] != "noisy":
            continue
        cl = _clean_path_for_noisy(noisy)
        if cl is not None:
            pairs.append((noisy.resolve(), cl.resolve()))
    uniq: dict[tuple[str, str], tuple[Path, Path]] = {}
    for a, b in pairs:
        uniq[(str(a), str(b))] = (a, b)
    return sorted(uniq.values(), key=lambda t: str(t[0]))


_FILEID_RX = re.compile(r"fileid_(\d+)", re.I)


def parquet_sample_id_to_flac(sample_id: str) -> str:
    if "_fileid_" not in sample_id:
        return ""
    tail = sample_id.split("_fileid_", 1)[-1]
    try:
        n = int(tail)
    except ValueError:
        return ""
    return f"fileid_{n}.flac"


def parquet_filestem_key(path_str: str) -> str:
    stem = Path(path_str).stem.lower()
    m = _FILEID_RX.search(stem)
    return m.group(0).lower() if m else stem
