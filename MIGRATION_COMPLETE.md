# Soniq 新架构迁移完成报告

## 已完成的工作

### 1. 骨架创建 ✅

运行 `python create_skeleton.py` 创建了以下新目录和文件：

#### 新增目录结构
```
soniq/
├── models/
│   ├── base/
│   │   ├── configuration_base.py    # SoniqModelConfig
│   │   ├── modeling_base.py         # SoniqModel
│   │   └── outputs.py               # 输出 dataclass
│   └── vocoders/
│       └── base.py                  # BaseVocoderModel
│
├── tasks/
│   ├── base/
│   │   ├── system.py                # BaseTaskSystem
│   │   ├── objective.py             # BaseObjective
│   │   ├── evaluator.py             # BaseEvaluator
│   │   ├── inferencer.py            # BaseInferencer
│   │   └── outputs.py               # StepOutput 等
│   └── vocoder/
│       ├── system.py                # VocoderTaskSystem
│       ├── datasets.py              # VocoderDataset
│       ├── collators.py             # VocoderCollator
│       └── objectives/
│           ├── generator_loss.py    # GeneratorLoss
│           └── discriminator_loss.py# DiscriminatorLoss
│
├── data/
│   ├── dataset_base.py              # ManifestDataset
│   ├── collator_base.py             # BaseCollator
│   └── manifest.py                  # Manifest 工具
│
├── processing/
│   ├── base.py                      # BaseProcessor
│   ├── audio/io.py                  # load_audio, save_audio
│   └── features/mel.py              # MelSpectrogramExtractor
│
├── runtime/
│   └── engine.py                    # FabricEngine
│
├── pipelines/
│   ├── base.py                      # BasePipeline
│   └── vocoder.py                   # VocoderPipeline
│
├── config/
│   └── experiment.py                # ExperimentConfig
│
└── bins/
    └── train_vocoder.py             # 训练入口
```

### 2. HiFiGAN 模型迁移 ✅

**修改的文件：**
- `soniq/models/vocoders/hifigan/configuration_hifigan.py`
- `soniq/models/vocoders/hifigan/modeling_hifigan.py`

**新增的文件：**
- `soniq/models/vocoders/hifigan/discriminator.py`

**关键变更：**

1. **配置类继承变更**:
```python
# 旧
class HifiGANConfig(SoniqConfig):

# 新
class HifiGANConfig(SoniqModelConfig):
```

2. **模型类继承变更**:
```python
# 旧
class HifiGAN(SoniqPreTrainedModel):

# 新
class HifiGAN(BaseVocoderModel):
```

3. **添加 synthesize 方法**:
```python
def synthesize(
    self,
    acoustic_features: torch.Tensor,
    **kwargs
) -> VocoderOutput:
    """从梅尔频谱合成波形"""
    waveform = self.forward(acoustic_features)
    return VocoderOutput(waveform=waveform)
```

4. **返回类型变更**:
```python
# 旧
def forward(self, x: torch.Tensor) -> torch.Tensor:

# 新
def forward(self, acoustic_features: torch.Tensor, **kwargs) -> VocoderOutput:
```

5. **新增 Discriminator**:
```python
# 新增完整的 HiFiGAN 判别器实现
class HiFiGANMultiPeriodDiscriminator
class HiFiGANMultiScaleDiscriminator
```

### 3. VocoderTaskSystem 创建 ✅

**完整实现：**
- `soniq/tasks/vocoder/system.py` - 主系统类
- `soniq/tasks/vocoder/datasets.py` - 数据集
- `soniq/tasks/vocoder/collators.py` - 批处理
- `soniq/tasks/vocoder/objectives/` - 损失函数

**VocoderTaskSystem 核心功能：**

```python
class VocoderTaskSystem(BaseTaskSystem):
    """
    Vocoder 任务系统，处理 GAN 训练的完整逻辑：
    - 生成器和判别器交替训练
    - 多周期判别器 (MPD)
    - 多尺度判别器 (MSD)
    - 特征匹配损失
    - 梅尔谱重建损失
    """

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # optimizer_idx=0: 训练判别音器
        # optimizer_idx=1: 训练生成器

    def configure_optimizers(self):
        # 返回两个优化器：generator 和 discriminator
```

### 4. 训练入口创建 ✅

**文件：** `soniq/bins/train_vocoder.py`

**使用方法：**
```bash
python -m soniq.bins.train_vocoder \
    --config config/hifigan.json \
    --output_dir ./outputs/hifigan \
    --devices 0
```

---

## 架构对比

### 旧架构 vs 新架构

| 组件 | 旧架构 | 新架构 |
|------|--------|--------|
| 模型基类 | `SoniqPreTrainedModel` | `SoniqModel` → `BaseVocoderModel` |
| 配置基类 | `SoniqConfig` | `SoniqModelConfig` |
| 训练器 | `FabricTrainer` (薄壳) | `FabricEngine` + `VocoderTaskSystem` |
| 损失函数 | 全局 `training/losses/` | 任务特定 `tasks/vocoder/objectives/` |
| 数据集 | `BaseDataset` (通用) | `VocoderDataset` (任务特定) |
| 推理 | 无统一接口 | `BasePipeline` + `VocoderPipeline` |

### 职责分离

```
┌─────────────────────────────────────────────────┐
│ 模型层 (models/)                                 │
│ - 只做前向传播                                    │
│ - 可 save_pretrained / from_pretrained          │
│ - 可 push_to_hub                                │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ 任务层 (tasks/)                                  │
│ - 训练逻辑 (training_step)                       │
│ - 损失计算 (objectives)                          │
│ - 优化器配置 (configure_optimizers)              │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ 运行层 (runtime/)                                │
│ - Fabric 引擎                                     │
│ - 分布式/AMP                                     │
│ - Checkpoint/日志                               │
└─────────────────────────────────────────────────┘
```

---

## 使用示例

### 1. 训练 HiFiGAN

```python
from soniq.config.experiment import ExperimentConfig
from soniq.runtime.engine import FabricEngine
from soniq.tasks.vocoder.system import VocoderTaskSystem
from soniq.tasks.vocoder.datasets import VocoderDataset
from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig

# 加载配置
config = ExperimentConfig.from_json("config/hifigan.json")

# 创建生成器
model_config = HifiGANConfig(**config.model.model_args)
generator = HifiGAN(model_config)

# 创建任务系统
vocoder_config = VocoderConfig(learning_rate=2e-4)
system = VocoderTaskSystem(vocoder_config, generator)

# 创建数据集和数据加载器
dataset = VocoderDataset("data/train.json", config)
dataloader = DataLoader(dataset, collate_fn=VocoderCollator(config))

# 训练
engine = FabricEngine(accelerator="gpu", devices=[0])
engine.fit(system, dataloader, max_epochs=100)
```

### 2. 推理

```python
from soniq.pipelines.vocoder import VocoderPipeline
from soniq.models.vocoders.hifigan import HifiGAN
import torch

# 加载模型
model = HifiGAN.from_pretrained("myuser/my-hifigan-model")

# 创建 pipeline
pipeline = VocoderPipeline(model)

# 推理
mel = torch.randn(80, 100)  # (n_mel, time)
audio = pipeline(mel)
```

### 3. 上传到 HuggingFace Hub

```python
from soniq import HuggingFaceHub

hub = HuggingFaceHub()
hub.login(token="hf_xxx")

# 方法 1: 从目录上传
hub.upload_model(
    model_path="./checkpoints/hifigan",
    repo_id="myuser/my-hifigan-model",
    model_type="vocoder",
)

# 方法 2: 直接从模型实例上传
model.push_to_hub("myuser/my-hifigan-model")
```

---

## 文件清单

### 新增文件 (20 个)

| 文件 | 说明 |
|------|------|
| `models/base/configuration_base.py` | SoniqModelConfig |
| `models/base/modeling_base.py` | SoniqModel |
| `models/base/outputs.py` | ModelOutput dataclass |
| `models/vocoders/base.py` | BaseVocoderModel |
| `models/vocoders/hifigan/discriminator.py` | MPD + MSD |
| `tasks/base/system.py` | BaseTaskSystem |
| `tasks/base/objective.py` | BaseObjective |
| `tasks/base/evaluator.py` | BaseEvaluator |
| `tasks/base/inferencer.py` | BaseInferencer |
| `tasks/base/outputs.py` | StepOutput 等 |
| `tasks/vocoder/system.py` | VocoderTaskSystem |
| `tasks/vocoder/datasets.py` | VocoderDataset |
| `tasks/vocoder/collators.py` | VocoderCollator |
| `tasks/vocoder/objectives/generator_loss.py` | GeneratorLoss |
| `tasks/vocoder/objectives/discriminator_loss.py` | DiscriminatorLoss |
| `data/dataset_base.py` | ManifestDataset |
| `data/collator_base.py` | BaseCollator |
| `data/manifest.py` | Manifest 工具 |
| `processing/audio/io.py` | 音频 I/O |
| `processing/features/mel.py` | 梅尔特征提取 |
| `runtime/engine.py` | FabricEngine |
| `pipelines/base.py` | BasePipeline |
| `pipelines/vocoder.py` | VocoderPipeline |
| `config/experiment.py` | ExperimentConfig |
| `bins/train_vocoder.py` | 训练入口 |

### 修改文件 (3 个)

| 文件 | 修改内容 |
|------|----------|
| `models/vocoders/hifigan/configuration_hifigan.py` | 继承 SoniqModelConfig |
| `models/vocoders/hifigan/modeling_hifigan.py` | 继承 BaseVocoderModel，添加 synthesize 方法 |
| `models/vocoders/__init__.py` | 更新导出 |

---

## 下一步建议

### 立即可做
1. **测试训练流程**: 使用示例数据运行 `train_vocoder.py`
2. **添加更多损失函数**: 如 Mel 谱损失、周期一致性损失
3. **完善数据增强**: 在 `processing/audio/augment.py` 中添加增强功能

### 短期目标
1. **迁移其他 vocoder**: 如 BigVGAN、Vocos
2. **添加评估器**: `tasks/vocoder/evaluators/`
3. **添加推理器**: `tasks/vocoder/inferencers/`

### 中期目标
1. **TTS 任务迁移**: FastSpeech2、VITS
2. **Codec 任务迁移**: Encodec、DAC
3. **SVC/VC任务迁移**

---

## 测试命令

```bash
# 1. 测试导入
python -c "from soniq.models.vocoders.hifigan import HifiGAN; print('OK')"

# 2. 测试模型创建
python -c "
from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig
config = HifiGANConfig()
model = HifiGAN(config)
print(f'Model params: {model.num_parameters:,}')
"

# 3. 测试前向传播
python -c "
import torch
from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig
config = HifiGANConfig()
model = HifiGAN(config)
mel = torch.randn(1, 80, 50)
output = model.synthesize(mel)
print(f'Input mel shape: {mel.shape}')
print(f'Output audio shape: {output.waveform.shape}')
"
```

---

## 总结

已完成：
- ✅ 完整的新架构骨架
- ✅ HiFiGAN 模型迁移到新基类
- ✅ 完整的 VocoderTaskSystem 实现
- ✅ 训练入口脚本
- ✅ 所有必要的 __init__.py 文件

新架构优势：
1. **清晰的职责分离**: 模型/任务/运行三层独立
2. **可复用的组件**: VocoderTaskSystem 可用于任何 vocoder
3. **更好的扩展性**: 添加新模型/任务更简单
4. **HuggingFace Hub 集成**: 支持 push_to_hub / from_pretrained
