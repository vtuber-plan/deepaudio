# Soniq 新架构设计文档

## 1. 完整目录树（精确到文件级别）

```
soniq/
├── __init__.py                 # 包入口，导出公共 API
│
├── models/                     # Artifact 层：可发布模型
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── configuration_base.py   # SoniqModelConfig (继承 PretrainedConfig)
│   │   ├── modeling_base.py        # SoniqModel (继承 PreTrainedModel)
│   │   └── outputs.py              # ModelOutput dataclass
│   ├── vocoders/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseVocoderModel
│   │   └── hifigan/
│   │       ├── __init__.py
│   │       ├── configuration_hifigan.py
│   │       └── modeling_hifigan.py
│   ├── tts/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseTTSModel
│   │   ├── fastspeech2/
│   │   ├── vits/
│   │   └── valle/
│   ├── codec/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseCodecModel
│   │   └── ...
│   ├── svc/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseSVCModel
│   │   └── ...
│   ├── vc/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseVCModel
│   │   └── ...
│   ├── asr/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseASRModel
│   │   └── ...
│   ├── speaker_encoder/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseSpeakerEncoderModel
│   │   └── ...
│   └── f0/
│       ├── __init__.py
│       ├── base.py                 # BaseF0Model
│       └── ...
│
├── tasks/                      # Task 层：任务级闭环
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── system.py           # BaseTaskSystem
│   │   ├── objective.py        # BaseObjective
│   │   ├── evaluator.py        # BaseEvaluator
│   │   ├── inferencer.py       # BaseInferencer
│   │   └── outputs.py          # StepOutput, EvalOutput, InferOutput
│   ├── vocoder/
│   │   ├── __init__.py
│   │   ├── system.py           # VocoderTaskSystem
│   │   ├── datasets.py         # VocoderDataset
│   │   ├── collators.py        # VocoderCollator
│   │   ├── objectives/
│   │   │   ├── __init__.py
│   │   │   ├── generator.py    # GeneratorLoss
│   │   │   ├── discriminator.py# DiscriminatorLoss
│   │   │   └── feature_match.py# FeatureMatchingLoss
│   │   ├── evaluators/
│   │   │   ├── __init__.py
│   │   │   └── vocoder.py      # VocoderEvaluator
│   │   └── inferencers/
│   │       ├── __init__.py
│   │       └── vocoder.py      # VocoderInferencer
│   ├── tts/
│   │   ├── __init__.py
│   │   ├── system.py
│   │   ├── datasets.py
│   │   ├── collators.py
│   │   ├── objectives/
│   │   ├── evaluators/
│   │   └── inferencers/
│   ├── codec/
│   │   ├── __init__.py
│   │   ├── system.py
│   │   ├── datasets.py
│   │   ├── collators.py
│   │   ├── objectives/
│   │   ├── evaluators/
│   │   └── inferencers/
│   ├── svc/
│   ├── vc/
│   └── asr/
│
├── data/                       # 数据层：与任务无关
│   ├── __init__.py
│   ├── manifest.py             # Manifest 加载器
│   ├── dataset_base.py         # ManifestDataset
│   ├── collator_base.py        # BaseCollator
│   ├── samplers/
│   │   ├── __init__.py
│   │   ├── bucket.py           # BucketSampler
│   │   └── dynamic.py          # DynamicBatchSampler
│   └── readers/
│       ├── __init__.py
│       ├── audio.py            # AudioReader
│       ├── numpy.py            # NumpyReader
│       └── text.py             # TextReader
│
├── processing/                 # 处理层：音频/文本/特征
│   ├── __init__.py
│   ├── base.py                 # BaseProcessor
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── io.py               # load_audio, save_audio
│   │   ├── resample.py         # ResampleProcessor
│   │   ├── normalize.py        # NormalizeProcessor
│   │   ├── augment.py          # DataAugmentation
│   │   └── vad.py              # VoiceActivityDetection
│   ├── features/
│   │   ├── __init__.py
│   │   ├── mel.py              # MelSpectrogramExtractor
│   │   ├── f0.py               # F0Extractor (PM, CREAMPE, FCNF0)
│   │   ├── energy.py           # EnergyExtractor
│   │   └── spectral.py         # SpectralExtractor
│   └── text/
│       ├── __init__.py
│       ├── normalize.py        # TextNormalizer
│       ├── g2p.py              # G2PConverter
│       ├── symbols.py          # Phoneme/Character Sets
│       └── tokenizer.py        # TextTokenizer
│
├── runtime/                    # 运行层：执行引擎
│   ├── __init__.py
│   ├── engine.py               # FabricEngine (原 FabricTrainer)
│   ├── checkpoint.py           # CheckpointManager
│   ├── callbacks/
│   │   ├── __init__.py
│   │   ├── base.py             # Callback
│   │   ├── checkpoint.py       # CheckpointCallback
│   │   ├── early_stopping.py   # EarlyStoppingCallback
│   │   ├── progress.py         # ProgressCallback
│   │   └── logging.py          # LoggingCallback
│   ├── loggers/
│   │   ├── __init__.py
│   │   ├── base.py             # Logger
│   │   ├── tensorboard.py      # TensorBoardLogger
│   │   ├── csv.py              # CSVLogger
│   │   └── wandb.py            # WandBLogger
│   └── distributed.py          # 分布式工具函数
│
├── pipelines/                  # 用户推理 API
│   ├── __init__.py
│   ├── base.py                 # BasePipeline
│   ├── vocoder.py              # VocoderPipeline
│   ├── tts.py                  # TextToSpeechPipeline
│   ├── vc.py                   # VoiceConversionPipeline
│   ├── codec.py                # CodecPipeline
│   └── asr.py                  # ASRPipeline
│
├── modules/                    # 神经网络组件
│   ├── __init__.py
│   ├── commons/
│   │   ├── __init__.py
│   │   ├── norm.py             # LayerNorm, InstanceNorm
│   │   ├── res_block.py        # ResidualBlock
│   │   ├── conv.py             # Conv1d/Conv2d wrappers
│   │   └── embedding.py        # PositionalEmbedding
│   ├── transformer/
│   │   ├── __init__.py
│   │   ├── attention.py        # MultiHeadAttention
│   │   ├── encoder.py          # TransformerEncoder
│   │   ├── decoder.py          # TransformerDecoder
│   │   ├── embedding.py        # TokenEmbedding
│   │   └── ffm.py              # FeedForwardModule
│   ├── diffusion/
│   │   ├── __init__.py
│   │   ├── unet.py             # UNet
│   │   └── noise_scheduler.py  # NoiseScheduler
│   ├── flow/
│   │   ├── __init__.py
│   │   ├── flow.py             # FlowBase
│   │   └── coupling.py         # CouplingLayer
│   └── adversarial/
│       ├── __init__.py
│       ├── discriminator.py    # MultiScaleDiscriminator
│       └── losses.py           # AdversarialLoss
│
├── config/                     # 实验配置（非模型配置）
│   ├── __init__.py
│   ├── experiment.py           # ExperimentConfig
│   ├── train.py                # TrainConfig
│   ├── data.py                 # DataConfig
│   ├── model.py                # ModelBuildConfig
│   ├── infer.py                # InferConfig
│   └── base_config.py          # BaseConfig (保留向后兼容)
│
├── utils/                      # 工具函数
│   ├── __init__.py
│   ├── audio_utils.py
│   ├── model_utils.py
│   ├── data_utils.py
│   ├── cuda_utils.py
│   └── hub_utils.py            # HuggingFace Hub 工具
│
├── hub.py                      # HuggingFaceHub 高级类
├── hub_cli.py                  # CLI 工具
└── bins/                       # 训练/推理入口（移到包外或保留）
    ├── train.py
    ├── infer.py
    └── export.py
```

---

## 2. 基础类 Python Skeleton

### 2.1 `models/base/configuration_base.py`

```python
# coding=utf-8
"""Base configuration class for Soniq models."""

from transformers import PretrainedConfig
from typing import Any, Dict


class SoniqModelConfig(PretrainedConfig):
    """
    Base configuration for all Soniq models.

    This class handles model architecture parameters only.
    Training-related config (batch_size, optimizer, etc.) should
    be in ExperimentConfig.
    """

    model_type = "soniq"

    def __init__(
        self,
        initializer_range: float = 0.02,
        **kwargs
    ):
        self.initializer_range = initializer_range
        super().__init__(**kwargs)

    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model-specific config as dict."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') and k not in ['return_dict', 'output_hidden_states']
        }
```

---

### 2.2 `models/base/modeling_base.py`

```python
# coding=utf-8
"""Base model class for Soniq models."""

from typing import Dict, Optional, Union, List
import os
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import logging
from .configuration_base import SoniqModelConfig


logger = logging.get_logger(__name__)


class SoniqModel(PreTrainedModel):
    """
    Abstract base class for all Soniq models.

    Models should inherit from this class to get:
    - Hugging Face Hub integration (save_pretrained, from_pretrained, push_to_hub)
    - Common utilities (device, num_parameters)
    - Standard forward/infer interface
    """

    config_class = SoniqModelConfig
    base_model_prefix = "soniq"
    supports_gradient_checkpointing = True

    def __init__(self, config: SoniqModelConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device

    @property
    def num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def infer(self, **inputs) -> "ModelOutput":
        """
        Inference mode forward pass.

        Default implementation wraps forward() with no_grad and eval mode.
        Subclasses can override for custom inference logic.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(**inputs)

    def init_weights(self) -> None:
        """Initialize model weights."""
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with normal distribution."""
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
```

---

### 2.3 `models/base/outputs.py`

```python
# coding=utf-8
"""Output dataclasses for Soniq models."""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import torch


@dataclass
class ModelOutput:
    """Base class for model outputs."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None


@dataclass
class VocoderOutput(ModelOutput):
    """Output for vocoder models."""
    waveform: Optional[torch.Tensor] = None
    mel_reconstruction: Optional[torch.Tensor] = None


@dataclass
class TTSOutput(ModelOutput):
    """Output for TTS models."""
    waveform: Optional[torch.Tensor] = None
    mel_spectrogram: Optional[torch.Tensor] = None
    durations: Optional[torch.Tensor] = None
    alignments: Optional[torch.Tensor] = None


@dataclass
class CodecOutput(ModelOutput):
    """Output for codec models."""
    reconstructed: Optional[torch.Tensor] = None
    codes: Optional[torch.Tensor] = None
    quantized: Optional[torch.Tensor] = None


@dataclass
class SVCOutput(ModelOutput):
    """Output for SVC models."""
    waveform: Optional[torch.Tensor] = None
    content_features: Optional[torch.Tensor] = None
    speaker_embedding: Optional[torch.Tensor] = None


@dataclass
class VCOutput(ModelOutput):
    """Output for voice conversion models."""
    waveform: Optional[torch.Tensor] = None
    converted_features: Optional[torch.Tensor] = None
    source_content: Optional[torch.Tensor] = None
    target_speaker: Optional[torch.Tensor] = None


@dataclass
class ASROutput(ModelOutput):
    """Output for ASR models."""
    transcription: Optional[str] = None
    logits: Optional[torch.Tensor] = None
    token_ids: Optional[torch.Tensor] = None
    alignments: Optional[torch.Tensor] = None
```

---

### 2.4 `models/vocoders/base.py`

```python
# coding=utf-8
"""Base class for vocoder models."""

from typing import Optional, Tuple
import torch
from torch import nn
from ..base.modeling_base import SoniqModel
from ..base.outputs import VocoderOutput


class BaseVocoderModel(SoniqModel):
    """
    Base class for all vocoder models.

    Vocoders convert acoustic features (e.g., mel spectrograms) to waveforms.
    """

    config_class = SoniqModel  # Override in subclass

    def synthesize(
        self,
        acoustic_features: torch.Tensor,
        **kwargs
    ) -> VocoderOutput:
        """
        Synthesize waveform from acoustic features.

        Args:
            acoustic_features: Acoustic features (batch, channels, time)
            **kwargs: Additional arguments

        Returns:
            VocoderOutput with waveform
        """
        raise NotImplementedError(
            "Subclasses must implement synthesize() method"
        )

    def forward(
        self,
        acoustic_features: torch.Tensor,
        **kwargs
    ) -> VocoderOutput:
        """Forward pass for training."""
        return self.synthesize(acoustic_features, **kwargs)
```

---

### 2.5 `tasks/base/system.py`

```python
# coding=utf-8
"""Base task system class."""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataclasses import dataclass


@dataclass
class StepOutput:
    """Output of a training/validation step."""
    loss: torch.Tensor
    metrics: Dict[str, float]
    logs: Dict[str, Any]


@dataclass
class EvalOutput:
    """Output of evaluation."""
    metrics: Dict[str, float]
    results: Dict[str, Any]


class BaseTaskSystem(nn.Module):
    """
    Base class for task systems.

    A task system encapsulates:
    - One or more models
    - Training/validation logic
    - Optimizer configuration
    - Loss computation

    The runtime engine only interacts with the system interface,
    not individual models.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> StepOutput:
        """
        Execute one training step.

        Args:
            batch: Mini-batch of data
            batch_idx: Batch index within epoch

        Returns:
            StepOutput with loss and metrics
        """
        raise NotImplementedError("Subclasses must implement training_step()")

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> StepOutput:
        """Execute one validation step."""
        raise NotImplementedError("Subclasses must implement validation_step()")

    def inference_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Execute one inference step."""
        raise NotImplementedError("Subclasses must implement inference_step()")

    def configure_optimizers(
        self
    ) -> Union[
        torch.optim.Optimizer,
        Dict[str, torch.optim.Optimizer]
    ]:
        """
        Configure optimizers.

        Returns:
            Single optimizer or dict of optimizers for multi-model systems.
        """
        raise NotImplementedError("Subclasses must implement configure_optimizers()")

    def configure_schedulers(
        self
    ) -> Union[
        Any,
        Dict[str, Any]
    ]:
        """Configure learning rate schedulers."""
        return None

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        return list(self.parameters())

    def on_train_start(self) -> None:
        """Called when training starts."""
        pass

    def on_train_end(self) -> None:
        """Called when training ends."""
        pass

    def on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, epoch: int) -> None:
        """Called at the end of each epoch."""
        pass
```

---

### 2.6 `tasks/base/objective.py`

```python
# coding=utf-8
"""Base objective (loss) class."""

from typing import Dict, Any, Optional
import torch
from torch import nn


class BaseObjective(nn.Module):
    """
    Base class for task objectives (loss functions).

    Separates loss computation from the model, allowing:
    - Multiple objectives per model
    - Loss composition
    - Easier testing
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute loss and metrics.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
            - 'loss': Total loss (scalar tensor)
            - 'loss_items': Individual loss components
            - 'metrics': Computed metrics
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute total loss. To be implemented by subclasses."""
        raise NotImplementedError

    def compute_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute metrics. Override in subclasses."""
        return {}
```

---

### 2.7 `tasks/base/evaluator.py`

```python
# coding=utf-8
"""Base evaluator class."""

from typing import Dict, Any, List
import torch
from torch.utils.data import DataLoader


class BaseEvaluator:
    """
    Base class for task evaluators.

    Handles evaluation logic including:
    - Metric aggregation
    - Result logging
    - Checkpoint-based evaluation
    """

    def __init__(self, config):
        self.config = config
        self.metrics_history: List[Dict[str, float]] = []

    def evaluate(
        self,
        system,
        dataloader: DataLoader,
        **kwargs
    ) -> Dict[str, float]:
        """
        Run evaluation on a dataloader.

        Args:
            system: Task system to evaluate
            dataloader: Validation dataloader

        Returns:
            Dictionary of metric name -> value
        """
        raise NotImplementedError("Subclasses must implement evaluate()")

    def reset(self) -> None:
        """Reset metrics."""
        self.metrics_history = []

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics across all batches."""
        if not self.metrics_history:
            return {}

        aggregated = {}
        for key in self.metrics_history[0].keys():
            values = [m[key] for m in self.metrics_history if key in m]
            if values:
                aggregated[key] = sum(values) / len(values)
        return aggregated
```

---

### 2.8 `tasks/base/inferencer.py`

```python
# coding=utf-8
"""Base inferencer class."""

from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import DataLoader


class BaseInferencer:
    """
    Base class for task inferencers.

    Handles inference logic including:
    - Batch processing
    - Result aggregation
    - Output formatting
    """

    def __init__(self, config, system):
        self.config = config
        self.system = system
        self.system.eval()

    @torch.no_grad()
    def infer(
        self,
        dataloader: DataLoader,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run inference on dataloader.

        Args:
            dataloader: Inference dataloader

        Returns:
            List of result dictionaries
        """
        results = []
        for batch in dataloader:
            batch_results = self.infer_batch(batch, **kwargs)
            results.extend(batch_results)
        return results

    def infer_batch(
        self,
        batch: Dict[str, torch.Tensor],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a single batch.

        Args:
            batch: Mini-batch of data

        Returns:
            List of results for each item in batch
        """
        raise NotImplementedError("Subclasses must implement infer_batch()")
```

---

### 2.9 `runtime/engine.py`

```python
# coding=utf-8
"""Fabric runtime engine."""

from typing import Any, Dict, List, Optional, Callable, Union
import torch
from torch import nn
from torch.utils.data import DataLoader
from lightning import Fabric
from lightning.fabric.loggers import Logger as FabricLogger
from dataclasses import dataclass
from ..tasks.base.system import BaseTaskSystem


@dataclass
class EngineState:
    """Training state container."""
    epoch: int = 0
    global_step: int = 0
    total_steps: int = 0
    best_metric: Optional[float] = None


class FabricEngine:
    """
    Runtime engine based on Lightning Fabric.

    This engine handles:
    - Accelerator setup (GPU/CPU/MPS)
    - Distributed training (DDP/FSDP)
    - Mixed precision
    - Checkpointing
    - Logging

    It does NOT define task logic - that's the responsibility
    of BaseTaskSystem.
    """

    def __init__(
        self,
        accelerator: str = "auto",
        strategy: str = "auto",
        devices: Optional[Union[int, List[int]]] = None,
        num_nodes: int = 1,
        precision: str = "32-true",
        gradient_accumulation_steps: int = 1,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: str = "norm",
        loggers: Optional[List[FabricLogger]] = None,
        callbacks: Optional[List] = None,
        default_root_dir: str = "./logs",
    ):
        self.accelerator = accelerator
        self.strategy = strategy
        self.devices = devices
        self.num_nodes = num_nodes
        self.precision = precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.default_root_dir = default_root_dir
        self.callbacks = callbacks or []

        # Initialize Fabric
        self.fabric = Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            loggers=loggers,
            gradient_accumulation_steps=gradient_accumulation_steps,
            callbacks=self.callbacks,
        )

        # State
        self.state = EngineState()

    def fit(
        self,
        system: BaseTaskSystem,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        max_epochs: int = 100,
        max_steps: Optional[int] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> None:
        """
        Train the system.

        Args:
            system: Task system to train
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            max_epochs: Maximum number of epochs
            max_steps: Optional maximum number of steps (overrides epochs)
            resume_from_checkpoint: Optional checkpoint path to resume from
        """
        self.fabric.launch()

        # Prepare system and optimizer
        optimizer = system.configure_optimizers()

        if isinstance(optimizer, dict):
            # Multi-optimizer case (e.g., GAN)
            system, optimizers = self._prepare_multi_optimizer(
                system, optimizer
            )
        else:
            system, optimizer = self.fabric.setup(system, optimizer)

        train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        if val_dataloader is not None:
            val_dataloader = self.fabric.setup_dataloaders(val_dataloader)

        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self._load_checkpoint(system, optimizer, resume_from_checkpoint)

        # Training loop
        system.on_train_start()

        for epoch in range(self.state.epoch, max_epochs):
            self.state.epoch = epoch
            system.on_epoch_start(epoch)

            # Training epoch
            self._train_epoch(system, train_dataloader, optimizer)

            # Validation
            if val_dataloader is not None:
                self._validate_epoch(system, val_dataloader)

            system.on_epoch_end(epoch)

            # Check max_steps
            if max_steps and self.state.global_step >= max_steps:
                break

        system.on_train_end()
        self.fabric.barrier()

    def _train_epoch(
        self,
        system: BaseTaskSystem,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Train for one epoch."""
        system.train()

        for batch_idx, batch in enumerate(dataloader):
            self.state.global_step += 1

            # Training step
            step_output = system.training_step(batch, batch_idx)

            # Backward pass
            self.fabric.backward(step_output.loss)

            # Gradient clipping
            if self.gradient_clip_val is not None:
                self.fabric.clip_gradients(
                    system, optimizer,
                    clip_val=self.gradient_clip_val,
                    clip_algorithm=self.gradient_clip_algorithm
                )

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            # Log metrics
            self._log_step_metrics(step_output)

    def _validate_epoch(
        self,
        system: BaseTaskSystem,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Validate for one epoch."""
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

    def _prepare_multi_optimizer(
        self,
        system: BaseTaskSystem,
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Prepare system with multiple optimizers."""
        # For GAN-style training
        system = self.fabric.setup_module(system)
        for name, opt in optimizers.items():
            optimizers[name] = self.fabric.setup_optimizer(opt)
        return system, optimizers

    def _log_step_metrics(self, step_output) -> None:
        """Log metrics from step output."""
        self.fabric.log("train/loss", step_output.loss.item(), on_step=True)
        for name, value in step_output.metrics.items():
            self.fabric.log(f"train/{name}", value, on_step=True)

    def _load_checkpoint(
        self,
        system: BaseTaskSystem,
        optimizer: torch.optim.Optimizer,
        path: str,
    ) -> None:
        """Load checkpoint."""
        checkpoint = self.fabric.load(path)
        system.load_state_dict(checkpoint["system"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        self.state.epoch = checkpoint.get("epoch", 0)
        self.state.global_step = checkpoint.get("global_step", 0)

    def save_checkpoint(
        self,
        system: BaseTaskSystem,
        optimizer: torch.optim.Optimizer,
        path: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save checkpoint."""
        checkpoint = {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "system": system.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if extra:
            checkpoint.update(extra)
        self.fabric.save(path, checkpoint)
```

---

### 2.10 `data/dataset_base.py`

```python
# coding=utf-8
"""Base dataset class."""

from typing import Any, Dict, List, Callable, Optional
import json
import os
import torch
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    """
    Dataset that reads from a manifest file.

    The manifest is a JSON/JSONL file where each line/entry contains
    metadata about one sample. This class handles loading the manifest
    and delegating feature loading to readers/processors.

    Example manifest entry:
    {
        "id": "utt_001",
        "audio_path": "/path/to/audio.wav",
        "text": "Hello world",
        "speaker": "spk_001",
        "duration": 3.5
    }
    """

    def __init__(
        self,
        manifest_path: str,
        readers: Dict[str, Callable],
        processors: Optional[List[Callable]] = None,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
    ):
        """
        Initialize dataset.

        Args:
            manifest_path: Path to manifest JSON/JSONL file
            readers: Dictionary of reader functions for different data types
            processors: List of processor functions to apply to each sample
            filter_fn: Optional filter function for samples
        """
        self.manifest_path = manifest_path
        self.readers = readers
        self.processors = processors or []
        self.filter_fn = filter_fn

        self.metadata = self._load_manifest()

        if self.filter_fn:
            self.metadata = [m for m in self.metadata if self.filter_fn(m)]

    def _load_manifest(self) -> List[Dict[str, Any]]:
        """Load manifest from file."""
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            if self.manifest_path.endswith('.jsonl'):
                return [json.loads(line) for line in f]
            else:
                return json.load(f)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get item by index."""
        entry = self.metadata[index].copy()

        # Apply readers to load data from paths
        sample = self._apply_readers(entry)

        # Apply processors
        for processor in self.processors:
            sample = processor(sample)

        return sample

    def _apply_readers(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Apply readers to entry."""
        sample = {"id": entry.get("id", str(index))}

        for key, reader in self.readers.items():
            if key in entry:
                sample[key] = reader(entry[key])

        # Copy over metadata
        for key in ["speaker", "duration", "text", "language"]:
            if key in entry:
                sample[key] = entry[key]

        return sample
```

---

### 2.11 `processing/base.py`

```python
# coding=utf-8
"""Base processor class."""

from typing import Any, Dict, Optional
from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    """
    Base class for processors.

    A processor transforms a single sample dictionary.
    Processors are composable and can be chained.
    """

    def __init__(self, config=None):
        self.config = config

    @abstractmethod
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a sample.

        Args:
            sample: Input sample dictionary

        Returns:
            Processed sample dictionary
        """
        raise NotImplementedError

    def update_config(self, **kwargs) -> None:
        """Update processor config."""
        if self.config:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
```

---

### 2.12 `processing/audio/io.py`

```python
# coding=utf-8
"""Audio I/O utilities."""

from typing import Union, Optional
import torch
import torchaudio
import numpy as np


def load_audio(
    path: str,
    sample_rate: Optional[int] = None,
    mono: bool = True,
) -> torch.Tensor:
    """
    Load audio from file.

    Args:
        path: Path to audio file
        sample_rate: Target sample rate (resamples if different)
        mono: Convert to mono

    Returns:
        Audio tensor of shape (channels, time) or (time,) if mono
    """
    waveform, sr = torchaudio.load(path)

    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate and sr != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, sr, sample_rate
        )

    return waveform


def save_audio(
    waveform: torch.Tensor,
    path: str,
    sample_rate: int,
    normalize: bool = True,
) -> None:
    """
    Save audio to file.

    Args:
        waveform: Audio tensor
        path: Output path
        sample_rate: Sample rate
        normalize: Normalize to [-1, 1]
    """
    if normalize:
        waveform = waveform / (waveform.abs().max() + 1e-8)

    torchaudio.save(path, waveform, sample_rate)
```

---

### 2.13 `data/collator_base.py`

```python
# coding=utf-8
"""Base collator class."""

from typing import Any, Dict, List
import torch
from torch.nn.utils.rnn import pad_sequence


class BaseCollator:
    """
    Base class for collating samples into batches.

    Handles:
    - Padding variable-length sequences
    - Tensor conversion
    - Batch organization
    """

    def __init__(
        self,
        padding_value: float = 0.0,
        batch_first: bool = True,
        pad_keys: Optional[List[str]] = None,
    ):
        self.padding_value = padding_value
        self.batch_first = batch_first
        self.pad_keys = pad_keys or []

    def __call__(
        self,
        batch: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.

        Args:
            batch: List of sample dictionaries

        Returns:
            Collated batch dictionary
        """
        if len(batch) == 0:
            return {}

        result = {}

        # Collect all keys
        all_keys = set()
        for sample in batch:
            all_keys.update(sample.keys())

        for key in all_keys:
            values = [s[key] for s in batch if key in s]

            if len(values) == 0:
                continue

            if isinstance(values[0], torch.Tensor):
                if key in self.pad_keys or self._is_sequence(values):
                    # Pad sequences
                    result[key] = pad_sequence(
                        values,
                        batch_first=self.batch_first,
                        padding_value=self.padding_value
                    )
                    result[f"{key}_lengths"] = torch.tensor(
                        [v.shape[0] if self.batch_first else v.shape[1]
                         for v in values]
                    )
                else:
                    # Stack tensors
                    result[key] = torch.stack(values)
            else:
                # Keep as list
                result[key] = values

        return result

    def _is_sequence(self, values: List[torch.Tensor]) -> bool:
        """Check if values are variable-length sequences."""
        if len(values) < 2:
            return False
        shapes = [v.shape for v in values]
        return len(set(shapes)) > 1
```

---

### 2.14 `pipelines/base.py`

```python
# coding=utf-8
"""Base pipeline class."""

from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
import torch


class BasePipeline(ABC):
    """
    Base class for inference pipelines.

    Pipelines provide a high-level interface for end-to-end inference,
    handling all preprocessing and postprocessing internally.
    """

    def __init__(self, model, processor=None, config=None):
        self.model = model
        self.processor = processor
        self.config = config
        self.model.eval()

    @abstractmethod
    def __call__(self, inputs, **kwargs):
        """
        Run pipeline on inputs.

        Args:
            inputs: Input data (type depends on pipeline)
            **kwargs: Additional arguments

        Returns:
            Output data
        """
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, inputs) -> Dict[str, torch.Tensor]:
        """Preprocess inputs."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Model forward pass."""
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, outputs: Dict[str, torch.Tensor]) -> Any:
        """Postprocess outputs."""
        raise NotImplementedError

    def __call__(self, inputs, **kwargs) -> Any:
        """Full pipeline: preprocess -> forward -> postprocess."""
        processed = self.preprocess(inputs)
        outputs = self.forward(processed)
        return self.postprocess(outputs)
```

---

### 2.15 `config/experiment.py`

```python
# coding=utf-8
"""Experiment configuration."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import os


@dataclass
class TrainConfig:
    """Training configuration."""
    max_epochs: int = 100
    max_steps: Optional[int] = None
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    gradient_clip_val: float = 1.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    save_interval: int = 1000
    eval_interval: int = 1000
    log_interval: int = 100


@dataclass
class DataConfig:
    """Data configuration."""
    train_manifest: str = ""
    val_manifest: str = ""
    sample_rate: int = 24000
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class ModelBuildConfig:
    """Model build configuration."""
    model_type: str = ""
    model_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """
    Full experiment configuration.

    This combines all configuration needed to run an experiment.
    Model architecture config should be in ModelBuildConfig.model_args
    or a separate SoniqModelConfig.
    """
    name: str = "experiment"
    output_dir: str = "./outputs"

    model: ModelBuildConfig = field(default_factory=ModelBuildConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Runtime
    accelerator: str = "auto"
    strategy: str = "auto"
    devices: Optional[List[int]] = None
    precision: str = "16-mixed"
    seed: int = 42

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        config = cls()

        if 'name' in data:
            config.name = data['name']
        if 'output_dir' in data:
            config.output_dir = data['output_dir']

        if 'model' in data:
            config.model = ModelBuildConfig(**data['model'])
        if 'train' in data:
            config.train = TrainConfig(**data['train'])
        if 'data' in data:
            config.data = DataConfig(**data['data'])

        for key in ['accelerator', 'strategy', 'devices', 'precision', 'seed']:
            if key in data:
                setattr(config, key, data[key])

        return config

    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=2)
```
