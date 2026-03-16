# Soniq 新架构完整设计总结

## 文档导航

1. [架构设计文档](ARCHITECTURE_DESIGN.md) - 完整架构说明
2. [HuggingFace Hub 集成](docs/zh/hub_integration.md) - Hub 使用指南
3. [HiFiGAN 迁移指南](docs/zh/hifigan_migration.md) - 迁移示例

## 一、新目录结构

```
soniq/
├── models/              # Artifact 层：可发布模型
│   ├── base/
│   │   ├── configuration_base.py   # SoniqModelConfig
│   │   ├── modeling_base.py        # SoniqModel
│   │   └── outputs.py              # 输出 dataclass
│   ├── vocoders/
│   │   ├── base.py                 # BaseVocoderModel
│   │   └── hifigan/
│   ├── tts/
│   ├── codec/
│   ├── svc/
│   ├── vc/
│   └── asr/
│
├── tasks/               # Task 层：任务级闭环
│   ├── base/
│   │   ├── system.py               # BaseTaskSystem
│   │   ├── objective.py            # BaseObjective
│   │   ├── evaluator.py            # BaseEvaluator
│   │   ├── inferencer.py           # BaseInferencer
│   │   └── outputs.py              # StepOutput 等
│   ├── vocoder/
│   │   ├── system.py               # VocoderTaskSystem
│   │   ├── datasets.py
│   │   ├── collators.py
│   │   ├── objectives/
│   │   └── inferencers/
│   └── ...
│
├── data/                # 数据层
│   ├── manifest.py
│   ├── dataset_base.py             # ManifestDataset
│   ├── collator_base.py            # BaseCollator
│   ├── samplers/
│   └── readers/
│
├── processing/          # 处理层
│   ├── base.py                     # BaseProcessor
│   ├── audio/
│   │   ├── io.py                   # load_audio, save_audio
│   │   ├── resample.py
│   │   └── normalize.py
│   ├── features/
│   │   ├── mel.py                  # MelSpectrogramExtractor
│   │   ├── f0.py
│   │   └── energy.py
│   └── text/
│       ├── normalize.py
│       ├── g2p.py
│       └── tokenizer.py
│
├── runtime/             # 运行层
│   ├── engine.py                   # FabricEngine
│   ├── checkpoint.py
│   ├── callbacks/
│   └── loggers/
│
├── pipelines/           # 用户推理 API
│   ├── base.py                     # BasePipeline
│   ├── vocoder.py                  # VocoderPipeline
│   ├── tts.py
│   └── vc.py
│
├── modules/             # 神经网络组件
│   ├── commons/
│   ├── transformer/
│   ├── diffusion/
│   └── flow/
│
├── config/              # 实验配置
│   ├── experiment.py               # ExperimentConfig
│   ├── train.py                    # TrainConfig
│   ├── data.py                     # DataConfig
│   └── model.py                    # ModelBuildConfig
│
└── utils/               # 工具函数
```

## 二、核心基础类

### 2.1 模型层 (Artifact Layer)

| 类 | 位置 | 职责 |
|----|------|------|
| `SoniqModelConfig` | `models/base/configuration_base.py` | 模型结构配置，继承 PretrainedConfig |
| `SoniqModel` | `models/base/modeling_base.py` | 模型基类，继承 PreTrainedModel |
| `BaseVocoderModel` | `models/vocoders/base.py` | Vocoder 模型接口 |
| `BaseTTSModel` | `models/tts/base.py` | TTS 模型接口 |
| `BaseCodecModel` | `models/codec/base.py` | Codec 模型接口 |

### 2.2 任务层 (Task Layer)

| 类 | 位置 | 职责 |
|----|------|------|
| `BaseTaskSystem` | `tasks/base/system.py` | 任务系统，封装训练逻辑 |
| `BaseObjective` | `tasks/base/objective.py` | 损失函数基类 |
| `BaseEvaluator` | `tasks/base/evaluator.py` | 评估器基类 |
| `BaseInferencer` | `tasks/base/inferencer.py` | 推理器基类 |
| `StepOutput` | `tasks/base/outputs.py` | 训练步骤输出 |

### 2.3 数据层 (Data Layer)

| 类 | 位置 | 职责 |
|----|------|------|
| `ManifestDataset` | `data/dataset_base.py` | 基于 manifest 的数据集 |
| `BaseCollator` | `data/collator_base.py` | 批处理_collate_fn_ |

### 2.4 处理层 (Processing Layer)

| 类 | 位置 | 职责 |
|----|------|------|
| `BaseProcessor` | `processing/base.py` | 处理器基类 |
| `MelSpectrogramExtractor` | `processing/features/mel.py` | 梅尔特征提取 |
| `F0Extractor` | `processing/features/f0.py` | 基频提取 |

### 2.5 运行层 (Runtime Layer)

| 类 | 位置 | 职责 |
|----|------|------|
| `FabricEngine` | `runtime/engine.py` | Fabric 训练引擎 |
| `CheckpointManager` | `runtime/checkpoint.py` | 检查点管理 |

### 2.6 管道层 (Pipeline Layer)

| 类 | 位置 | 职责 |
|----|------|------|
| `BasePipeline` | `pipelines/base.py` | 推理管道基类 |
| `VocoderPipeline` | `pipelines/vocoder.py` | Vocoder 推理 |
| `TTSPipeline` | `pipelines/tts.py` | TTS 推理 |

### 2.7 配置层 (Config Layer)

| 类 | 位置 | 职责 |
|----|------|------|
| `ExperimentConfig` | `config/experiment.py` | 实验配置 |
| `TrainConfig` | `config/train.py` | 训练配置 |
| `DataConfig` | `config/data.py` | 数据配置 |
| `ModelBuildConfig` | `config/model.py` | 模型构建配置 |

## 三、设计原则

### 3.1 分层原则

1. **Artifact 层**：只包含可保存、可发布、可复用的模型
   - 无训练逻辑
   - 支持 `save_pretrained` / `from_pretrained`
   - 支持 HuggingFace Hub

2. **Task 层**：任务特定的训练/推理逻辑
   - 可组合多个模型
   - 处理损失计算
   - 处理优化器配置

3. **Runtime 层**：纯执行引擎
   - 不感知任务语义
   - 只消费标准接口
   - 处理分布式/AMP/Checkpoint

### 3.2 数据流原则

```
manifest → dataset → readers → processors → collator → system → engine
```

### 3.3 推理流原则

```
pipeline → preprocess → model.infer() → postprocess → output
```

## 四、与 Amphion 对比

| 特性 | Amphion | Soniq 新架构 |
|------|---------|-------------|
| 组织方式 | 按任务目录 | 分层 + 任务 |
| Trainer | 厚重，负责一切 | 轻量引擎 + 系统 |
| 模型 | 与训练耦合 | 独立可发布 |
| 配置 | 单一 config | 模型配置 + 实验配置分离 |
| 推理 | BaseInference 耦合 | Pipeline 解耦 |
| Hub 集成 | 无 | 完整支持 |

## 五、HiFiGAN 迁移对照

### 文件迁移

| 原文件 | 新位置/操作 |
|--------|-------------|
| `models/vocoders/hifigan/configuration_hifigan.py` | 保持不变，改继承 `SoniqModelConfig` |
| `models/vocoders/hifigan/modeling_hifigan.py` | 保持不变，改继承 `BaseVocoderModel` |
| `training/losses/generator_loss.py` | → `tasks/vocoder/objectives/generator.py` |
| `training/losses/discriminator_loss.py` | → `tasks/vocoder/objectives/discriminator.py` |
| `training/losses/feature_loss.py` | → `tasks/vocoder/objectives/feature_match.py` |
| `bins/train_vocoder.py` | → `bins/train.py` (使用 FabricEngine) |
| N/A | ← 新建 `tasks/vocoder/system.py` |
| N/A | ← 新建 `tasks/vocoder/datasets.py` |
| N/A | ← 新建 `tasks/vocoder/collators.py` |

### 代码变更示例

**模型类变更**:
```python
# 旧
class HifiGAN(SoniqPreTrainedModel):

# 新
class HifiGAN(BaseVocoderModel):
    def synthesize(self, acoustic_features, **kwargs) -> VocoderOutput:
        waveform = self.forward(acoustic_features)
        return VocoderOutput(waveform=waveform)
```

**训练入口变更**:
```python
# 旧
trainer = FabricTrainer(...)
trainer.fit(model, dataloader)

# 新
system = VocoderTaskSystem(config, generator, discriminator)
engine = FabricEngine(...)
engine.fit(system, train_dataloader, val_dataloader)
```

## 六、实施路线图

### 阶段 1：骨架搭建 (Week 1-2)
- [ ] 运行 `create_skeleton.py` 创建目录和基础类
- [ ] 更新 `models/base/` 继承关系
- [ ] 创建 `tasks/vocoder/` 基础结构

### 阶段 2：Vocoder 迁移 (Week 3-4)
- [ ] 迁移 HiFiGAN 到新架构
- [ ] 创建 `VocoderTaskSystem`
- [ ] 迁移损失函数
- [ ] 测试训练流程

### 阶段 3：数据/处理层重构 (Week 5-6)
- [ ] 实现 `ManifestDataset`
- [ ] 实现 `processors` 链
- [ ] 迁移特征提取代码
- [ ] 统一音频 I/O

### 阶段 4：Runtime 引擎 (Week 7-8)
- [ ] 实现 `FabricEngine`
- [ ] 实现 `CheckpointManager`
- [ ] 实现 callbacks 系统
- [ ] 与 TaskSystem 集成测试

### 阶段 5：Pipeline 和文档 (Week 9-10)
- [ ] 实现 `VocoderPipeline`
- [ ] 实现 `TTSPipeline`
- [ ] 完善文档
- [ ] 示例代码

### 阶段 6：扩展其他任务 (Week 11+)
- [ ] TTS 任务迁移
- [ ] Codec 任务迁移
- [ ] SVC/VC 任务迁移
- [ ] ASR 任务迁移

## 七、快速开始

### 创建骨架
```bash
cd /data/wangjun/github/deepaudio
python create_skeleton.py
```

### 测试模型加载
```python
from soniq.models.vocoders.hifigan import HifiGAN
model = HifiGAN.from_pretrained("./checkpoints/hifigan")
```

### 测试 Hub 上传
```python
from soniq import HuggingFaceHub
hub = HuggingFaceHub()
hub.login(token="hf_xxx")
model.push_to_hub("myuser/my-hifigan-model")
```

## 八、关键设计决策

### 为什么要分层？

1. **可维护性**：每层职责清晰
2. **可测试性**：各层独立测试
3. **可扩展性**：新任务不影响其他层
4. **可发布性**：模型可独立发布

### 为什么 System 在 Task 层？

1. 音频任务常涉及多个模型（GAN、Codec 等）
2. 训练逻辑是任务特定的
3. Runtime 应保持通用

### 为什么分离模型配置和实验配置？

1. 模型配置是模型的一部分
2. 实验配置是运行时的
3. HuggingFace Hub 只需要模型配置

## 九、API 示例

### 训练一个新模型
```python
from soniq.config.experiment import ExperimentConfig
from soniq.runtime.engine import FabricEngine
from soniq.tasks.vocoder.system import VocoderTaskSystem
from soniq.tasks.vocoder.datasets import VocoderDataset
from soniq.tasks.vocoder.collators import VocoderCollator
from torch.utils.data import DataLoader

# 加载配置
config = ExperimentConfig.from_json("config/hifigan.json")

# 创建系统
system = VocoderTaskSystem(config, generator, discriminator)

# 创建数据集
dataset = VocoderDataset(config.data.train_manifest, config)
collator = VocoderCollator(config)
dataloader = DataLoader(dataset, collate_fn=collator)

# 训练
engine = FabricEngine(accelerator="gpu", devices=[0])
engine.fit(system, dataloader, max_epochs=config.train.max_epochs)
```

### 推理
```python
from soniq.pipelines.vocoder import VocoderPipeline
from soniq.models.vocoders.hifigan import HifiGAN

# 加载模型
model = HifiGAN.from_pretrained("myuser/my-hifigan-model")

# 创建 pipeline
pipeline = VocoderPipeline(model)

# 推理
mel = torch.randn(80, 100)  # (n_mel, time)
audio = pipeline(mel)
```

## 十、总结

新架构的核心思想：

> **HF 友好的模型基座 + 任务级 System 抽象 + 薄 Runtime 引擎 + 清晰的数据/处理边界**

借鉴 Amphion 的优点（任务导向、完整闭环），但避免其缺点（耦合过深、边界不清），最终形成一个更规范、更易扩展的音频 AI 框架。
