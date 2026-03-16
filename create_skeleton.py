#!/usr/bin/env python3
# coding=utf-8
"""
Script to create the new Soniq architecture skeleton files.

Run this script to generate all the base classes and directory structure
for the new architecture.
"""

import os


def create_file(path, content):
    """Create a file with the given content."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {path}")


def create_init(path):
    """Create an empty __init__.py file."""
    create_file(path, '# coding=utf-8\n"""Soniq module."""\n\n')


def main():
    base_dir = "/data/wangjun/github/deepaudio/soniq"

    # ========== MODELS ==========
    print("\n=== Creating models/ ===\n")

    # models/base/
    create_file(f"{base_dir}/models/base/configuration_base.py", '''# coding=utf-8
"""Base configuration class for Soniq models."""

from transformers import PretrainedConfig
from typing import Any, Dict


class SoniqModelConfig(PretrainedConfig):
    """Base configuration for all Soniq models."""

    model_type = "soniq"

    def __init__(self, initializer_range: float = 0.02, **kwargs):
        self.initializer_range = initializer_range
        super().__init__(**kwargs)

    @property
    def model_config(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_') and k not in ['return_dict']}
''')

    create_file(f"{base_dir}/models/base/modeling_base.py", '''# coding=utf-8
"""Base model class for Soniq models."""

from typing import Dict, Optional
import torch
from torch import nn
from transformers import PreTrainedModel
from .configuration_base import SoniqModelConfig


class SoniqModel(PreTrainedModel):
    """Abstract base class for all Soniq models."""

    config_class = SoniqModelConfig
    base_model_prefix = "soniq"
    supports_gradient_checkpointing = True

    def __init__(self, config: SoniqModelConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def infer(self, **inputs):
        self.eval()
        with torch.no_grad():
            return self.forward(**inputs)

    def init_weights(self) -> None:
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = getattr(self.config, 'initializer_range', 0.02)
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d,
                                nn.ConvTranspose1d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
''')

    create_file(f"{base_dir}/models/base/outputs.py", '''# coding=utf-8
"""Output dataclasses for Soniq models."""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class ModelOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None


@dataclass
class VocoderOutput(ModelOutput):
    waveform: Optional[torch.Tensor] = None


@dataclass
class TTSOutput(ModelOutput):
    waveform: Optional[torch.Tensor] = None
    mel_spectrogram: Optional[torch.Tensor] = None


@dataclass
class CodecOutput(ModelOutput):
    reconstructed: Optional[torch.Tensor] = None
    codes: Optional[torch.Tensor] = None


@dataclass
class SVCOutput(ModelOutput):
    waveform: Optional[torch.Tensor] = None
    content_features: Optional[torch.Tensor] = None
''')

    # models/vocoders/
    create_file(f"{base_dir}/models/vocoders/base.py", '''# coding=utf-8
"""Base class for vocoder models."""

from typing import Optional
import torch
from ..base.modeling_base import SoniqModel
from ..base.outputs import VocoderOutput


class BaseVocoderModel(SoniqModel):
    """Base class for all vocoder models."""

    def synthesize(self, acoustic_features: torch.Tensor, **kwargs) -> VocoderOutput:
        raise NotImplementedError("Subclasses must implement synthesize()")

    def forward(self, acoustic_features: torch.Tensor, **kwargs) -> VocoderOutput:
        return self.synthesize(acoustic_features, **kwargs)
''')

    # ========== TASKS ==========
    print("\n=== Creating tasks/ ===\n")

    # tasks/base/
    create_file(f"{base_dir}/tasks/base/system.py", '''# coding=utf-8
"""Base task system class."""

from typing import Any, Dict, List, Union
import torch
from torch import nn
from dataclasses import dataclass


@dataclass
class StepOutput:
    loss: torch.Tensor
    metrics: Dict[str, float]
    logs: Dict[str, Any]


class BaseTaskSystem(nn.Module):
    """Base class for task systems."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def training_step(self, batch, batch_idx: int) -> StepOutput:
        raise NotImplementedError

    def validation_step(self, batch, batch_idx: int) -> StepOutput:
        raise NotImplementedError

    def inference_step(self, batch) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict]:
        raise NotImplementedError

    def on_train_start(self) -> None: pass
    def on_train_end(self) -> None: pass
    def on_epoch_start(self, epoch: int) -> None: pass
    def on_epoch_end(self, epoch: int) -> None: pass
''')

    create_file(f"{base_dir}/tasks/base/objective.py", '''# coding=utf-8
"""Base objective class."""

from typing import Dict, Any
import torch
from torch import nn


class BaseObjective(nn.Module):
    """Base class for task objectives."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, predictions, targets, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
''')

    create_file(f"{base_dir}/tasks/base/evaluator.py", '''# coding=utf-8
"""Base evaluator class."""

from typing import Dict, List
from torch.utils.data import DataLoader


class BaseEvaluator:
    """Base class for task evaluators."""

    def __init__(self, config):
        self.config = config
        self.metrics_history: List[Dict[str, float]] = []

    def evaluate(self, system, dataloader: DataLoader, **kwargs) -> Dict[str, float]:
        raise NotImplementedError

    def reset(self) -> None:
        self.metrics_history = []

    def get_aggregate_metrics(self) -> Dict[str, float]:
        if not self.metrics_history:
            return {}
        aggregated = {}
        for key in self.metrics_history[0].keys():
            values = [m[key] for m in self.metrics_history if key in m]
            if values:
                aggregated[key] = sum(values) / len(values)
        return aggregated
''')

    create_file(f"{base_dir}/tasks/base/inferencer.py", '''# coding=utf-8
"""Base inferencer class."""

from typing import Any, Dict, List
import torch
from torch.utils.data import DataLoader


class BaseInferencer:
    """Base class for task inferencers."""

    def __init__(self, config, system):
        self.config = config
        self.system = system
        self.system.eval()

    @torch.no_grad()
    def infer(self, dataloader: DataLoader, **kwargs) -> List[Dict[str, Any]]:
        results = []
        for batch in dataloader:
            batch_results = self.infer_batch(batch, **kwargs)
            results.extend(batch_results)
        return results

    def infer_batch(self, batch, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError
''')

    create_file(f"{base_dir}/tasks/base/outputs.py", '''# coding=utf-8
"""Output dataclasses for tasks."""

from dataclasses import dataclass
from typing import Any, Dict
import torch


@dataclass
class StepOutput:
    loss: torch.Tensor
    metrics: Dict[str, float]
    logs: Dict[str, Any]


@dataclass
class EvalOutput:
    metrics: Dict[str, float]
    results: Dict[str, Any]


@dataclass
class InferOutput:
    results: Dict[str, Any]
''')

    # ========== DATA ==========
    print("\n=== Creating data/ ===\n")

    create_file(f"{base_dir}/data/dataset_base.py", '''# coding=utf-8
"""Base dataset class."""

from typing import Any, Callable, Dict, List, Optional
import json
import torch
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    """Dataset that reads from a manifest file."""

    def __init__(
        self,
        manifest_path: str,
        readers: Dict[str, Callable],
        processors: Optional[List[Callable]] = None,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
    ):
        self.manifest_path = manifest_path
        self.readers = readers
        self.processors = processors or []
        self.filter_fn = filter_fn
        self.metadata = self._load_manifest()

        if self.filter_fn:
            self.metadata = [m for m in self.metadata if self.filter_fn(m)]

    def _load_manifest(self) -> List[Dict[str, Any]]:
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            if self.manifest_path.endswith('.jsonl'):
                return [json.loads(line) for line in f]
            return json.load(f)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.metadata[index].copy()
        sample = self._apply_readers(entry)
        for processor in self.processors:
            sample = processor(sample)
        return sample

    def _apply_readers(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        sample = {"id": entry.get("id", str(index))}
        for key, reader in self.readers.items():
            if key in entry:
                sample[key] = reader(entry[key])
        for key in ["speaker", "duration", "text", "language"]:
            if key in entry:
                sample[key] = entry[key]
        return sample
''')

    create_file(f"{base_dir}/data/collator_base.py", '''# coding=utf-8
"""Base collator class."""

from typing import Any, Dict, List, Optional
import torch
from torch.nn.utils.rnn import pad_sequence


class BaseCollator:
    """Base class for collating samples."""

    def __init__(self, padding_value: float = 0.0, batch_first: bool = True,
                 pad_keys: Optional[List[str]] = None):
        self.padding_value = padding_value
        self.batch_first = batch_first
        self.pad_keys = pad_keys or []

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if len(batch) == 0:
            return {}
        result = {}
        all_keys = set()
        for sample in batch:
            all_keys.update(sample.keys())

        for key in all_keys:
            values = [s[key] for s in batch if key in s]
            if len(values) == 0:
                continue
            if isinstance(values[0], torch.Tensor):
                if key in self.pad_keys or self._is_sequence(values):
                    result[key] = pad_sequence(values, batch_first=self.batch_first,
                                               padding_value=self.padding_value)
                    result[f"{key}_lengths"] = torch.tensor(
                        [v.shape[0] if self.batch_first else v.shape[1] for v in values]
                    )
                else:
                    result[key] = torch.stack(values)
            else:
                result[key] = values
        return result

    def _is_sequence(self, values: List[torch.Tensor]) -> bool:
        if len(values) < 2:
            return False
        return len(set(v.shape for v in values)) > 1
''')

    create_file(f"{base_dir}/data/manifest.py", '''# coding=utf-8
"""Manifest utilities."""

import json
import os
from typing import Any, Dict, List


def load_manifest(path: str) -> List[Dict[str, Any]]:
    """Load manifest from JSON or JSONL file."""
    with open(path, 'r', encoding='utf-8') as f:
        if path.endswith('.jsonl'):
            return [json.loads(line) for line in f]
        return json.load(f)


def save_manifest(data: List[Dict[str, Any]], path: str) -> None:
    """Save manifest to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        if path.endswith('.jsonl'):
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\\n')
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)
''')

    # ========== PROCESSING ==========
    print("\n=== Creating processing/ ===\n")

    create_file(f"{base_dir}/processing/base.py", '''# coding=utf-8
"""Base processor class."""

from typing import Any, Dict
from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    """Base class for processors."""

    def __init__(self, config=None):
        self.config = config

    @abstractmethod
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
''')

    create_file(f"{base_dir}/processing/audio/io.py", '''# coding=utf-8
"""Audio I/O utilities."""

from typing import Optional
import torch
import torchaudio


def load_audio(path: str, sample_rate: Optional[int] = None,
               mono: bool = True) -> torch.Tensor:
    """Load audio from file."""
    waveform, sr = torchaudio.load(path)
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate and sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform


def save_audio(waveform: torch.Tensor, path: str, sample_rate: int,
               normalize: bool = True) -> None:
    """Save audio to file."""
    if normalize:
        waveform = waveform / (waveform.abs().max() + 1e-8)
    torchaudio.save(path, waveform, sample_rate)
''')

    create_file(f"{base_dir}/processing/features/mel.py", '''# coding=utf-8
"""Mel spectrogram extractor."""

from typing import Any, Dict
import torch
import torchaudio
from ..base import BaseProcessor


class MelSpectrogramExtractor(BaseProcessor):
    """Extract mel spectrograms from audio."""

    def __init__(self, config):
        super().__init__(config)
        self.sample_rate = getattr(config, 'sample_rate', 24000)
        self.n_fft = getattr(config, 'n_fft', 1024)
        self.n_mel = getattr(config, 'n_mel', 80)
        self.hop_length = getattr(config, 'hop_length', 256)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mel,
            hop_length=self.hop_length,
        )

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if 'audio' in sample:
            audio = sample['audio']
            mel = self.mel_transform(audio)
            sample['mel'] = mel.squeeze(0)
        return sample
''')

    # ========== RUNTIME ==========
    print("\n=== Creating runtime/ ===\n")

    create_file(f"{base_dir}/runtime/engine.py", '''# coding=utf-8
"""Fabric runtime engine."""

from typing import Any, Dict, List, Optional, Union
import torch
from torch.utils.data import DataLoader
from lightning import Fabric
from dataclasses import dataclass


@dataclass
class EngineState:
    epoch: int = 0
    global_step: int = 0


class FabricEngine:
    """Runtime engine based on Lightning Fabric."""

    def __init__(
        self,
        accelerator: str = "auto",
        strategy: str = "auto",
        devices: Optional[Union[int, List[int]]] = None,
        precision: str = "32-true",
        gradient_accumulation_steps: int = 1,
        gradient_clip_val: Optional[float] = None,
        loggers: Optional[List] = None,
        callbacks: Optional[List] = None,
    ):
        self.fabric = Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            loggers=loggers,
            callbacks=callbacks,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        self.gradient_clip_val = gradient_clip_val
        self.state = EngineState()

    def fit(self, system, train_dataloader, val_dataloader=None,
            max_epochs: int = 100, max_steps: Optional[int] = None,
            resume_from_checkpoint: Optional[str] = None):
        self.fabric.launch()
        optimizer = system.configure_optimizers()

        if isinstance(optimizer, dict):
            system, optimizers = self._prepare_multi_optimizer(system, optimizer)
        else:
            system, optimizer = self.fabric.setup(system, optimizer)
            optimizers = optimizer

        train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        if val_dataloader:
            val_dataloader = self.fabric.setup_dataloaders(val_dataloader)

        if resume_from_checkpoint:
            self._load_checkpoint(system, optimizers, resume_from_checkpoint)

        system.on_train_start()
        for epoch in range(self.state.epoch, max_epochs):
            self.state.epoch = epoch
            system.on_epoch_start(epoch)
            self._train_epoch(system, train_dataloader, optimizers)
            if val_dataloader:
                self._validate_epoch(system, val_dataloader)
            system.on_epoch_end(epoch)
            if max_steps and self.state.global_step >= max_steps:
                break
        system.on_train_end()

    def _train_epoch(self, system, dataloader, optimizer):
        system.train()
        for batch_idx, batch in enumerate(dataloader):
            self.state.global_step += 1
            step_output = system.training_step(batch, batch_idx)
            self.fabric.backward(step_output.loss)
            if self.gradient_clip_val:
                self.fabric.clip_gradients(system, optimizer,
                                           clip_val=self.gradient_clip_val)
            optimizer.step()
            optimizer.zero_grad()
            self._log_metrics(step_output)

    def _validate_epoch(self, system, dataloader):
        system.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                step_output = system.validation_step(batch, batch_idx)
                total_loss += step_output.loss.item()
                num_batches += 1
        avg_loss = total_loss / num_batches
        self.fabric.log("val/loss", avg_loss, on_epoch=True)
        return {"val/loss": avg_loss}

    def _log_metrics(self, step_output):
        self.fabric.log("train/loss", step_output.loss.item(), on_step=True)
        for name, value in step_output.metrics.items():
            self.fabric.log(f"train/{name}", value, on_step=True)

    def _prepare_multi_optimizer(self, system, optimizers):
        system = self.fabric.setup_module(system)
        for name, opt in optimizers.items():
            optimizers[name] = self.fabric.setup_optimizer(opt)
        return system, optimizers

    def _load_checkpoint(self, system, optimizer, path):
        checkpoint = self.fabric.load(path)
        system.load_state_dict(checkpoint["system"])
        if isinstance(optimizer, dict):
            for name, opt in optimizer.items():
                if name in checkpoint:
                    opt.load_state_dict(checkpoint[name])
        else:
            optimizer.load_state_dict(checkpoint["optimizer"])
        self.state.epoch = checkpoint.get("epoch", 0)
        self.state.global_step = checkpoint.get("global_step", 0)

    def save_checkpoint(self, system, optimizer, path, extra=None):
        checkpoint = {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "system": system.state_dict(),
            "optimizer": optimizer.state_dict() if not isinstance(optimizer, dict)
                         else {k: v.state_dict() for k, v in optimizer.items()},
        }
        if extra:
            checkpoint.update(extra)
        self.fabric.save(path, checkpoint)
''')

    # ========== PIPELINES ==========
    print("\n=== Creating pipelines/ ===\n")

    create_file(f"{base_dir}/pipelines/base.py", '''# coding=utf-8
"""Base pipeline class."""

from typing import Any, Dict
from abc import ABC, abstractmethod
import torch


class BasePipeline(ABC):
    """Base class for inference pipelines."""

    def __init__(self, model, processor=None, config=None):
        self.model = model
        self.processor = processor
        self.config = config
        self.model.eval()

    @abstractmethod
    def preprocess(self, inputs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, outputs: Dict[str, torch.Tensor]) -> Any:
        raise NotImplementedError

    def __call__(self, inputs, **kwargs) -> Any:
        processed = self.preprocess(inputs)
        outputs = self.forward(processed)
        return self.postprocess(outputs)
''')

    create_file(f"{base_dir}/pipelines/vocoder.py", '''# coding=utf-8
"""Vocoder pipeline."""

from typing import Any, Dict
import torch
from .base import BasePipeline


class VocoderPipeline(BasePipeline):
    """Pipeline for vocoder inference."""

    def preprocess(self, inputs) -> Dict[str, torch.Tensor]:
        if isinstance(inputs, torch.Tensor):
            return {"acoustic_features": inputs}
        return {"acoustic_features": torch.tensor(inputs)}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        waveform = self.model.synthesize(inputs["acoustic_features"])
        return {"waveform": waveform}

    def postprocess(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return outputs["waveform"].squeeze(0).cpu()
''')

    # ========== CONFIG ==========
    print("\n=== Creating config/ ===\n")

    create_file(f"{base_dir}/config/experiment.py", '''# coding=utf-8
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


@dataclass
class DataConfig:
    train_manifest: str = ""
    val_manifest: str = ""
    sample_rate: int = 24000
    num_workers: int = 4


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
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=2)
''')

    # Create __init__.py files
    print("\n=== Creating __init__.py files ===\n")
    modules = [
        "models", "models/base", "models/vocoders", "models/tts",
        "models/codec", "models/svc", "models/vc", "models/asr",
        "tasks", "tasks/base", "tasks/vocoder",
        "data", "processing", "processing/audio", "processing/features",
        "runtime", "pipelines", "config",
    ]
    for module in modules:
        create_init(f"{base_dir}/{module}/__init__.py")

    print("\n=== Skeleton creation complete! ===\n")


if __name__ == "__main__":
    main()