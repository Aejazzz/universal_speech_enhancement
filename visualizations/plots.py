from __future__ import annotations

from pathlib import Path
from typing import Dict

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def save_waveform_plot(original: np.ndarray, enhanced: np.ndarray, sr: int, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    librosa.display.waveshow(original, sr=sr, ax=axes[0], color="gray")
    axes[0].set_title("Original Waveform")
    librosa.display.waveshow(enhanced, sr=sr, ax=axes[1], color="green")
    axes[1].set_title("Enhanced Waveform")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_spectrogram_plot(original: np.ndarray, enhanced: np.ndarray, sr: int, out_path: str) -> None:
    """Original / Enhanced / Removed-noise spectrograms on a shared dB scale.

    The third panel is the difference (|orig| - |enh|) clipped to non-negative — i.e. the
    energy the enhancer removed. This makes the impact of routing immediately obvious.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n = min(len(original), len(enhanced))
    o = original[:n]
    e = enhanced[:n]

    Mo = np.abs(librosa.stft(o))
    Me = np.abs(librosa.stft(e))
    Mdiff = np.maximum(Mo - Me, 0.0)

    ref = max(Mo.max(), Me.max(), 1e-8)
    spec_o = librosa.amplitude_to_db(Mo, ref=ref)
    spec_e = librosa.amplitude_to_db(Me, ref=ref)
    spec_d = librosa.amplitude_to_db(Mdiff + 1e-8, ref=ref)

    vmin, vmax = -80, 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    img0 = librosa.display.specshow(
        spec_o, sr=sr, x_axis="time", y_axis="log", ax=axes[0], cmap="magma", vmin=vmin, vmax=vmax
    )
    axes[0].set_title("Original")
    fig.colorbar(img0, ax=axes[0], format="%+2.0f dB")
    img1 = librosa.display.specshow(
        spec_e, sr=sr, x_axis="time", y_axis="log", ax=axes[1], cmap="magma", vmin=vmin, vmax=vmax
    )
    axes[1].set_title("Enhanced")
    fig.colorbar(img1, ax=axes[1], format="%+2.0f dB")
    img2 = librosa.display.specshow(
        spec_d, sr=sr, x_axis="time", y_axis="log", ax=axes[2], cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[2].set_title("Removed (Original − Enhanced)")
    fig.colorbar(img2, ax=axes[2], format="%+2.0f dB")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_policy_plot(probabilities: Dict[str, float], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    sns.barplot(x=labels, y=values, hue=labels, legend=False, ax=ax, palette="viridis")
    ax.set_ylim(0, 1)
    ax.set_title("Policy Expert Probabilities")
    ax.set_ylabel("Probability")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
