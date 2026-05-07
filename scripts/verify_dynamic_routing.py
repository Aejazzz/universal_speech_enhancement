"""
Verify the dynamic inference-time router via the live backend at :8001.

For each test clip we POST /enhance and pretty-print:
  - the trained policy's advisory expert vs the dynamically-chosen expert,
  - every candidate the router scored (expert, strength, DNSMOS),
  - the DNSMOS delta the chosen output achieved,
  - the decision_reason recorded in routing.json.

We test three classes of inputs:
  1. Clean URGENT validation clips (should pick BYPASS).
  2. Lightly-noisy URGENT validation clips (oracle is mostly BYPASS).
  3. Synthetically-noisy clips at 0/5/10 dB SNR (should pick DFN3).
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

API = "http://127.0.0.1:8001"
SAMPLE_RATE = 16000
ROOT = Path(__file__).resolve().parents[1]


def _post(file_path: Path) -> dict:
    boundary = "----CursorBoundary7f3a"
    body = io.BytesIO()
    write = body.write
    write(f"--{boundary}\r\n".encode())
    write(
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'.encode()
    )
    write(b"Content-Type: audio/wav\r\n\r\n")
    write(file_path.read_bytes())
    write(f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        f"{API}/enhance",
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def _save_temp_wav(audio: np.ndarray, sr: int, label: str) -> Path:
    tmp_dir = ROOT / "outputs" / "_dyn_test"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / f"{label}.wav"
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(out_path.as_posix(), audio.astype(np.float32), sr)
    return out_path


def _make_noisy(clean_path: Path, snr_db: float, label: str) -> Path:
    """Mix a clean clip with white-Gaussian noise at the requested SNR (no babble dataset assumed)."""
    clean, sr = sf.read(clean_path.as_posix(), always_2d=False)
    if clean.ndim > 1:
        clean = clean.mean(axis=1)
    if sr != SAMPLE_RATE:
        # Crude resample via librosa if rates differ.
        import librosa  # noqa: WPS433

        clean = librosa.resample(np.asarray(clean, dtype=np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    clean = clean.astype(np.float32)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(len(clean)).astype(np.float32)
    sig_p = float(np.mean(clean ** 2) + 1e-8)
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_p = sig_p / snr_lin
    noise = noise * np.sqrt(noise_p / (np.mean(noise ** 2) + 1e-8))
    noisy = clean + noise
    peak = float(np.max(np.abs(noisy)) + 1e-8)
    if peak > 0.99:
        noisy = noisy / peak * 0.99
    return _save_temp_wav(noisy, SAMPLE_RATE, label)


def _print_result(label: str, result: dict) -> None:
    routing = result.get("routing", {})
    candidates = routing.get("dynamic_candidates", []) or []
    advice = routing.get("policy_advice", {})
    improvement = (result.get("metrics") or {}).get("improvement", {}) or {}
    d_dnsmos = float(improvement.get("dnsmos") or 0.0)
    print(f"--- {label} ---")
    print(
        f"  policy_advice: expert={advice.get('expert','?')} "
        f"strength={advice.get('strength', 0):.2f} "
        f"confidence={advice.get('confidence', 0):.2f}"
    )
    print(
        f"  dynamic_chosen: expert={routing.get('expert','?')} "
        f"strength={routing.get('strength', 0):.2f}  d_dnsmos={d_dnsmos:+0.3f}"
    )
    if candidates:
        print(f"  candidates ({len(candidates)}):")
        for c in candidates:
            mark = " <-- chosen" if (
                c["expert"] == routing.get("expert") and abs(c["strength"] - routing.get("strength", -1)) < 1e-6
            ) else ""
            print(
                f"    - {c['expert']:<14s} strength={c['strength']:.2f}  "
                f"OVRL={float(c['dnsmos']):.3f}{mark}"
            )
    print(f"  decision_reason: {routing.get('decision_reason','?')}")


def main() -> None:
    val = pd.read_csv("outputs/manifests/paired_val_oracle.csv")

    # Class 1: clean clip (should be BYPASS)
    clean_row = val.iloc[0]
    clean_path = Path(str(clean_row["clean_path"]))
    print(f"[1/3] CLEAN: {clean_path.name}")
    t0 = time.time()
    res = _post(clean_path)
    print(f"  ({time.time()-t0:.2f}s)")
    _print_result(f"CLEAN {clean_path.name}", res)

    # Class 2: lightly-noisy URGENT validation clip
    noisy_path = Path(str(clean_row["noisy_path"]))
    print(f"\n[2/3] LIGHT NOISY (URGENT): {noisy_path.name}")
    t0 = time.time()
    res = _post(noisy_path)
    print(f"  ({time.time()-t0:.2f}s)")
    _print_result(f"LIGHT NOISY {noisy_path.name}", res)

    # Class 3: synthetic heavily-noisy at multiple SNRs
    for snr in (10, 5, 0):
        synth = _make_noisy(clean_path, snr_db=float(snr), label=f"synth_snr{snr}db")
        print(f"\n[3/3] SYNTH NOISY {snr} dB: {synth.name}")
        t0 = time.time()
        res = _post(synth)
        print(f"  ({time.time()-t0:.2f}s)")
        _print_result(f"SYNTH {snr}dB", res)


if __name__ == "__main__":
    main()
