from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    image_size: int = 512


@dataclass(frozen=True)
class ModelConfig:
    detector_checkpoint: str = ""
    selector_checkpoint: str = ""
    longitudinal_checkpoint: str = ""
    language_model_path: str = ""
    hidden_dim: int = 1024
    dropout: float = 0.1
    selector_threshold: float = -1.0
    selector_top_k: int = 8
    max_length: int = 64
    num_beams: int = 3
    length_penalty: float = 1.1
    gradient_checkpointing: bool = True
    order_sensitive_scale: float = 0.3


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 1.0e-4


@dataclass(frozen=True)
class VEEConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: str | Path) -> VEEConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config must be a mapping")
    sections = {
        "data": DataConfig,
        "model": ModelConfig,
        "train": TrainConfig,
    }
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    values = {}
    for name, cls in sections.items():
        section = raw.get(name) or {}
        if not isinstance(section, dict):
            raise ValueError(f"config section {name!r} must be a mapping")
        values[name] = cls(**section)
    config = VEEConfig(**values)
    _validate(config)
    return config


def _validate(config: VEEConfig) -> None:
    if config.data.image_size != 512:
        raise ValueError("the anatomical detector requires image_size=512")
    model = config.model
    if model.hidden_dim < 1 or not 0 <= model.dropout < 1:
        raise ValueError("hidden_dim or dropout is invalid")
    if not math.isfinite(model.selector_threshold):
        raise ValueError("selector_threshold must be finite")
    if not 1 <= model.selector_top_k <= 29:
        raise ValueError("selector_top_k must be between 1 and 29")
    if min(model.max_length, model.num_beams) < 1:
        raise ValueError("decoder length and beam count must be positive")
    if model.length_penalty <= 0:
        raise ValueError("length_penalty must be positive")
    if not math.isfinite(model.order_sensitive_scale) or model.order_sensitive_scale < 0:
        raise ValueError("order_sensitive_scale must be finite and non-negative")
    if config.train.learning_rate <= 0 or not math.isfinite(config.train.learning_rate):
        raise ValueError("learning_rate must be finite and positive")


__all__ = ["DataConfig", "ModelConfig", "TrainConfig", "VEEConfig", "load_config"]
