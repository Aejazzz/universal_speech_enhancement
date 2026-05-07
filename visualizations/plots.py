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
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (audio, title) in enumerate([(original, "Original"), (enhanced, "Enhanced")]):
        spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = librosa.display.specshow(spec, sr=sr, x_axis="time", y_axis="log", ax=axes[idx], cmap="magma")
        axes[idx].set_title(f"{title} Spectrogram")
        fig.colorbar(img, ax=axes[idx], format="%+2.0f dB")
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
