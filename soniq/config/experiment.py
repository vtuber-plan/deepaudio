# coding=utf-8
"""Experiment configuration."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import os


@dataclass
class TrainConfig:
    max_epochs: int = 100
    max_steps: Optional[int] = None
    batch_size: int = 32
    learning_rate: float = 1e-4
    gradient_clip_val: float = 1.0
    gradient_accumulation_steps: int = 1


@dataclass
class DataConfig:
    train_manifest: str = ""
    val_manifest: str = ""
    sample_rate: int = 24000
    num_workers: int = 4
    segment_size: int = 8192
    persistent_workers: bool = True
    pin_memory: bool = True
    max_audio_duration: float = 10.0
    min_audio_duration: float = 0.1
    n_fft: int = 1024
    n_mel: int = 80
    hop_length: int = 256
    win_length: int = 1024
    f_min: float = 0.0
    f_max: float = 12000.0


@dataclass
class ModelBuildConfig:
    model_type: str = ""
    model_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    output_dir: str = "./outputs"
    model: ModelBuildConfig = field(default_factory=ModelBuildConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    accelerator: str = "auto"
    devices: Optional[List[int]] = None
    precision: str = "16-mixed"
    seed: int = 42

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle nested configs
        if 'train' in data and isinstance(data['train'], dict):
            data['train'] = TrainConfig(**data['train'])
        if 'data' in data and isinstance(data['data'], dict):
            data['data'] = DataConfig(**data['data'])
        if 'model' in data and isinstance(data['model'], dict):
            data['model'] = ModelBuildConfig(**data['model'])

        config = cls(**data)
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
