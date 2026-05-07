from __future__ import annotations



import json

from dataclasses import dataclass

from pathlib import Path

from typing import List



import librosa

import numpy as np

import pandas as pd

import torch

from torch.utils.data import DataLoader, Dataset





ACTIONS = ["DeepFilterNet3", "ResembleEnhance", "MossFormer2", "BYPASS"]





@dataclass

class SampleMeta:

    noisy_path: Path

    clean_path: Path | None





def _list_audio(root: Path) -> List[Path]:

    files: List[Path] = []

    for ext in ("*.wav", "*.mp3", "*.flac"):

        files.extend(root.rglob(ext))

    return sorted(files)





def _normalize(x: np.ndarray) -> np.ndarray:

    denom = np.max(np.abs(x)) + 1e-8

    return (x / denom).astype(np.float32)





def _distortion_features(waveform: np.ndarray, sr: int) -> np.ndarray:

    rms = float(np.sqrt(np.mean(waveform**2) + 1e-8))

    snr_proxy = float(np.clip(20.0 * np.log10(1.0 / (rms + 1e-8)), -5.0, 40.0))

    clip_prob = float(np.mean(np.abs(waveform) > 0.99))

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=waveform)))

    flatness = float(np.mean(librosa.feature.spectral_flatness(y=waveform)))

    reverb_proxy = float(np.clip(1.0 - zcr, 0.0, 1.0))

    intelligibility_proxy = float(np.clip(1.0 - flatness, 0.0, 1.0))

    codec_proxy = float(np.clip(flatness, 0.0, 1.0))

    noise_level = float(np.clip(1.0 - (snr_proxy / 40.0), 0.0, 1.0))

    return np.array(

        [

            (snr_proxy + 5.0) / 45.0,

            reverb_proxy,

            clip_prob,

            noise_level,

            codec_proxy,

            intelligibility_proxy,

        ],

        dtype=np.float32,

    )





def _target_from_features(feat: np.ndarray) -> tuple[int, float, float]:

    snr = feat[0]

    reverb = feat[1]

    clipping = feat[2]

    noise = feat[3]

    codec = feat[4]

    intelligibility = feat[5]

    severe = max(clipping, codec, 1.0 - intelligibility)

    difficult_mix = max(reverb, noise)

    if snr > 0.85 and severe < 0.15 and difficult_mix < 0.25:

        return ACTIONS.index("BYPASS"), 0.0, 0.0

    if severe >= 0.55:

        return ACTIONS.index("ResembleEnhance"), float(np.clip(0.6 + severe * 0.4, 0.0, 1.0)), 1.0

    if difficult_mix >= 0.5:

        return ACTIONS.index("MossFormer2"), float(np.clip(0.5 + difficult_mix * 0.5, 0.0, 1.0)), 1.0

    return ACTIONS.index("DeepFilterNet3"), float(np.clip(0.3 + noise * 0.5, 0.0, 1.0)), 0.0





class RoutingDataset(Dataset):

    def __init__(

        self,

        root: str | None = None,

        manifest_csv: str | None = None,

        sample_rate: int = 16000,

        max_seconds: float = 6.0,

    ) -> None:

        self.root = Path(root) if root else None

        self.sample_rate = sample_rate

        self.max_len = int(sample_rate * max_seconds)

        self.manifest_df = pd.DataFrame()

        self.samples: List[SampleMeta] = []



        if manifest_csv:

            self.manifest_df = pd.read_csv(manifest_csv).reset_index(drop=True)

            for _, row in self.manifest_df.iterrows():

                noisy_path = Path(str(row["noisy_path"]))

                clean_val = row.get("clean_path")

                clean_path = Path(str(clean_val)) if clean_val is not None and pd.notna(clean_val) else None

                self.samples.append(SampleMeta(noisy_path=noisy_path, clean_path=clean_path))

        elif self.root is not None:

            for p in _list_audio(self.root):

                self.samples.append(SampleMeta(noisy_path=p, clean_path=None))



        if not self.samples:

            source = manifest_csv or (str(self.root) if self.root else "<unknown>")

            raise ValueError(f"No audio files found for dataset source: {source}")



    def _labels_for_row(self, idx: int, features_6: np.ndarray) -> tuple[int, float, float]:

        heuristic = _target_from_features(features_6)

        if self.manifest_df.empty:

            return heuristic

        row = self.manifest_df.iloc[idx]

        if "oracle_action" in row.index and pd.notna(row["oracle_action"]):

            ai = int(float(row["oracle_action"]))

            st = (

                float(row["oracle_strength"])

                if "oracle_strength" in row.index and pd.notna(row.get("oracle_strength"))

                else heuristic[1]

            )

            rf = (

                float(row["oracle_refine"])

                if "oracle_refine" in row.index and pd.notna(row.get("oracle_refine"))

                else heuristic[2]

            )

            return ai, st, rf

        return heuristic



    def _soft_target(self, idx: int, temperature: float = 0.05) -> np.ndarray | None:

        """Convert oracle_scores_json -> per-expert soft probabilities.



        ``oracle_scores_json`` stores ``"<expert>@<strength>": score``. We take the

        max score per expert across strengths, scale by ``temperature``, and softmax to

        produce a probability distribution that mirrors how confident the oracle was

        between competing experts. Returns ``None`` if the column or JSON is missing.

        """

        if self.manifest_df.empty:

            return None

        if "oracle_scores_json" not in self.manifest_df.columns:

            return None

        raw = self.manifest_df.iloc[idx].get("oracle_scores_json")

        if not isinstance(raw, str) or not raw.strip():

            return None

        try:

            scores = json.loads(raw)

        except Exception:  # noqa: BLE001

            return None

        per_expert = {a: -1e9 for a in ACTIONS}

        for key, value in scores.items():

            if not isinstance(value, (int, float)) or not np.isfinite(value):

                continue

            expert = key.split("@", 1)[0]

            if expert in per_expert and value > per_expert[expert]:

                per_expert[expert] = float(value)

        arr = np.array([per_expert[a] for a in ACTIONS], dtype=np.float32)

        # Replace missing experts (e.g. MossFormer2 = -1e9) with the worst observed score

        # so they never dominate the softmax but still receive a tiny non-zero mass.

        valid = arr[arr > -1e8]

        if valid.size == 0:

            return None

        floor = float(valid.min()) - 0.5

        arr = np.where(arr <= -1e8, floor, arr)

        arr = arr - arr.max()  # numerical stability

        probs = np.exp(arr / max(temperature, 1e-3))

        probs = probs / probs.sum()

        return probs.astype(np.float32)



    def __len__(self) -> int:

        return len(self.samples)



    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:

        sample = self.samples[idx]

        wave, _ = librosa.load(sample.noisy_path.as_posix(), sr=self.sample_rate, mono=True)

        wave = _normalize(wave)

        if wave.shape[0] >= self.max_len:

            wave = wave[: self.max_len]

        else:

            wave = np.pad(wave, (0, self.max_len - wave.shape[0]))

        features = _distortion_features(wave, self.sample_rate)

        action_idx, strength, refine = self._labels_for_row(idx, features)

        soft = self._soft_target(idx)

        if soft is None:

            soft = np.zeros(len(ACTIONS), dtype=np.float32)

            soft[action_idx] = 1.0

        return {

            "wave": torch.tensor(wave, dtype=torch.float32),

            "distortion": torch.tensor(features, dtype=torch.float32),

            "action": torch.tensor(action_idx, dtype=torch.long),

            "soft_action": torch.tensor(soft, dtype=torch.float32),

            "strength": torch.tensor(strength, dtype=torch.float32),

            "refine": torch.tensor(refine, dtype=torch.float32),

        }





def build_routing_loader(root: str, batch_size: int, sample_rate: int, shuffle: bool = True) -> DataLoader:

    dataset = RoutingDataset(root=root, sample_rate=sample_rate)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)





def build_routing_loader_from_manifest(

    manifest_csv: str,

    batch_size: int,

    sample_rate: int,

    shuffle: bool = True,

) -> DataLoader:

    dataset = RoutingDataset(manifest_csv=manifest_csv, sample_rate=sample_rate)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


