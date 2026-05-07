from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac"}


def load_audio(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    extension = Path(path).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {extension}")
    waveform, sr = librosa.load(path, sr=target_sr, mono=True)
    return waveform.astype(np.float32), target_sr


def save_audio(path: str, waveform: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, sr)
