from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class SystemConfig(BaseModel):
    device: str = "cuda"
    mixed_precision: bool = True
    sample_rate: int = 16000
    dataset_root: str
    output_root: str = "outputs"
    mossformer_checkpoint: str = ""
    policy_checkpoint: str = "checkpoints/policy_best.pt"
    dynamic_routing: bool = True


class PolicyConfig(BaseModel):
    wavlm_name: str
    hidden_dim: int
    num_heads: int
    num_layers: int
    dropout: float
    num_actions: int


class TrainingConfig(BaseModel):
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    early_stopping_patience: int
    gradient_clip_norm: float
    distributed: bool = False
    train_manifest: str = "outputs/manifests/train.csv"
    val_manifest: str = "outputs/manifests/val.csv"


class AppConfig(BaseModel):
    system: SystemConfig
    policy: PolicyConfig
    training: TrainingConfig


@lru_cache(maxsize=1)
def load_config(config_path: str = "configs/base.yaml") -> AppConfig:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return AppConfig(**raw)
