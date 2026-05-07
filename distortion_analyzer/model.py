from __future__ import annotations

import librosa
import numpy as np
import torch
import torch.nn as nn


class DistortionAnalyzer(nn.Module):
    def __init__(self, n_mels: int = 80) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
            nn.Sigmoid(),
        )
        self.n_mels = n_mels

    def featurize(self, waveform: np.ndarray, sr: int) -> torch.Tensor:
        mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=self.n_mels)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor

    def predict(self, waveform: np.ndarray, sr: int) -> torch.Tensor:
        with torch.no_grad():
            features = self.featurize(waveform, sr)
            model_device = next(self.parameters()).device
            features = features.to(model_device)
            return self.net(features).squeeze(0).to("cpu")
