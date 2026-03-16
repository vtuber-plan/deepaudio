# HiFiGAN 迁移到新架构指南

## 当前状态

HiFiGAN 目前在 `soniq/models/vocoders/hifigan/` 目录下，包含：
- `configuration_hifigan.py` - 模型配置
- `modeling_hifigan.py` - 模型实现

训练相关代码分散在：
- `bins/train_vocoder.py` - 训练入口
- `training/losses/` - 损失函数
- `config/base/hifigan.json` - 训练配置

## 迁移目标

将 HiFiGAN 迁移到新架构后：

```
soniq/
├── models/vocoders/hifigan/
│   ├── configuration_hifigan.py   # 保留，继承 SoniqModelConfig
│   └── modeling_hifigan.py        # 保留，继承 BaseVocoderModel
│
└── tasks/vocoder/
    ├── system.py                  # 新增：VocoderTaskSystem
    ├── datasets.py                # 新增：VocoderDataset
    ├── collators.py               # 新增：VocoderCollator
    ├── objectives/
    │   ├── generator.py           # 从 training/losses/generator_loss.py 迁移
    │   ├── discriminator.py       # 从 training/losses/discriminator_loss.py 迁移
    │   └── feature_match.py       # 从 training/losses/feature_loss.py 迁移
    └── inferencers/
        └── vocoder.py             # 新增：推理逻辑
```

## 迁移步骤

### 步骤 1：更新模型基类

**原代码** (`modeling_hifigan.py:92`):
```python
from soniq.models.base.modeling_base import SoniqPreTrainedModel

class HifiGAN(SoniqPreTrainedModel):
    config_class = HifiGANConfig
    base_model_prefix = "hifigan"
```

**新代码**:
```python
from soniq.models.base.modeling_base import SoniqModel
from soniq.models.vocoders.base import BaseVocoderModel

class HifiGAN(BaseVocoderModel):
    config_class = HifiGANConfig
    base_model_prefix = "hifigan"

    def synthesize(self, acoustic_features, **kwargs) -> VocoderOutput:
        waveform = self.forward(acoustic_features)
        return VocoderOutput(waveform=waveform)
```

### 步骤 2：创建 VocoderTaskSystem

新建 `tasks/vocoder/system.py`:

```python
from typing import Dict, Any
import torch
from soniq.tasks.base.system import BaseTaskSystem
from soniq.tasks.base.outputs import StepOutput


class VocoderTaskSystem(BaseTaskSystem):
    def __init__(self, config, generator, discriminator=None):
        super().__init__(config)
        self.generator = generator
        self.discriminator = discriminator

    def training_step(self, batch, batch_idx) -> StepOutput:
        mel = batch["mel"]
        audio = batch["audio"]

        # Generator step
        fake_audio = self.generator(mel)

        # Compute losses (use objectives)
        loss_g, loss_g_items = self.generator_objective(fake_audio, audio)
        loss_d, loss_d_items = self.discriminator_objective(fake_audio, audio)

        # Total loss
        total_loss = loss_g + loss_d

        metrics = {
            "loss_g": loss_g.item(),
            "loss_d": loss_d.item(),
            **loss_g_items,
            **loss_d_items,
        }

        return StepOutput(loss=total_loss, metrics=metrics)

    def configure_optimizers(self):
        gen_opt = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.config.train.learning_rate
        )
        disc_opt = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.config.train.learning_rate
        )
        return {"generator": gen_opt, "discriminator": disc_opt}
```

### 步骤 3：迁移损失函数

**原代码** (`training/losses/generator_loss.py`):
```python
class GeneratorLoss:
    def __init__(self, discriminator):
        self.discriminator = discriminator
```

**新代码** (`tasks/vocoder/objectives/generator.py`):
```python
from soniq.tasks.base.objective import BaseObjective


class GeneratorLoss(BaseObjective):
    def __init__(self, config, discriminator):
        super().__init__(config)
        self.discriminator = discriminator

    def forward(self, predictions, targets, **kwargs):
        fake_audio = predictions["fake_audio"]
        real_audio = targets["audio"]

        # Discriminator output for fake audio
        disc_fake = self.discriminator(fake_audio)

        # Generator wants discriminator to think fake is real
        loss = sum((d - 1) ** 2 for d in disc_fake) / len(disc_fake)

        return {
            "loss": loss,
            "loss_items": {"loss_gen": loss.item()},
            "metrics": {},
        }
```

### 步骤 4：创建 Dataset 和 Collator

新建 `tasks/vocoder/datasets.py`:

```python
from soniq.data.dataset_base import ManifestDataset
from soniq.processing.audio.io import load_audio
from soniq.processing.features.mel import MelSpectrogramExtractor


class VocoderDataset(ManifestDataset):
    def __init__(self, manifest_path, config):
        readers = {
            "audio": lambda path: load_audio(path, config.sample_rate),
        }

        processors = [
            MelSpectrogramExtractor(config),
        ]

        super().__init__(
            manifest_path=manifest_path,
            readers=readers,
            processors=processors,
            filter_fn=lambda x: x.get("duration", 0) > 0.1,
        )
```

新建 `tasks/vocoder/collators.py`:

```python
from soniq.data.collator_base import BaseCollator


class VocoderCollator(BaseCollator):
    def __init__(self, config):
        super().__init__(
            padding_value=0.0,
            batch_first=True,
            pad_keys=["audio", "mel"],
        )
        self.config = config
```

### 步骤 5：更新训练入口

**原代码** (`bins/train_vocoder.py`):
```python
# 直接使用 FabricTrainer
trainer = FabricTrainer(...)
trainer.fit(model, dataloader)
```

**新代码**:
```python
from soniq.runtime.engine import FabricEngine
from soniq.tasks.vocoder.system import VocoderTaskSystem
from soniq.tasks.vocoder.datasets import VocoderDataset
from soniq.tasks.vocoder.collators import VocoderCollator

# Create system
system = VocoderTaskSystem(config, generator, discriminator)

# Create dataset
dataset = VocoderDataset(config.data.train_manifest, config)
collator = VocoderCollator(config)
dataloader = DataLoader(dataset, collate_fn=collator)

# Create engine and train
engine = FabricEngine(...)
engine.fit(system, dataloader, max_epochs=config.train.max_epochs)
```

## 迁移检查清单

- [ ] 更新 `HifiGAN` 继承 `BaseVocoderModel`
- [ ] 创建 `VocoderTaskSystem`
- [ ] 迁移损失函数到 `tasks/vocoder/objectives/`
- [ ] 创建 `VocoderDataset`
- [ ] 创建 `VocoderCollator`
- [ ] 更新训练入口使用 `FabricEngine`
- [ ] 测试保存/加载
- [ ] 测试 `push_to_hub`
- [ ] 测试 `from_pretrained`

## 文件对照表

| 原文件 | 新位置 | 操作 |
|--------|--------|------|
| `models/vocoders/hifigan/configuration_hifigan.py` | 保持不变 | 继承 `SoniqModelConfig` |
| `models/vocoders/hifigan/modeling_hifigan.py` | 保持不变 | 继承 `BaseVocoderModel` |
| `training/losses/generator_loss.py` | `tasks/vocoder/objectives/generator.py` | 迁移 + 继承 `BaseObjective` |
| `training/losses/discriminator_loss.py` | `tasks/vocoder/objectives/discriminator.py` | 迁移 + 继承 `BaseObjective` |
| `training/losses/feature_loss.py` | `tasks/vocoder/objectives/feature_match.py` | 迁移 + 继承 `BaseObjective` |
| `bins/train_vocoder.py` | `bins/train.py` | 更新为使用 `FabricEngine` |
| N/A | `tasks/vocoder/system.py` | 新建 |
| N/A | `tasks/vocoder/datasets.py` | 新建 |
| N/A | `tasks/vocoder/collators.py` | 新建 |

## 优势

迁移后的优势：
1. **清晰的职责分离**：模型只负责 forward，system 负责训练逻辑
2. **可复用的组件**：`VocoderTaskSystem` 可用于任何 vocoder
3. **更易测试**：objectives/datasets 可独立测试
4. **更好的扩展性**：添加新 vocoder 只需继承 `BaseVocoderModel`
5. **与 runtime 解耦**：可以更换训练引擎而不影响模型
