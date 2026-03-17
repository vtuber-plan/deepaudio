# Soniq Training Framework

## 概述

Soniq 训练框架是一个基于 **Lightning Fabric** 和 **HuggingFace Accelerate** 双引擎驱动的分布式训练框架，采用类似 PyTorch Lightning 的抽象设计，但更加轻量和灵活。

---

## 架构设计

```
soniq/training/
├── trainer.py                 # 统一训练器入口
├── base/                      # 基础抽象类
│   ├── __init__.py
│   ├── system.py             # BaseTaskSystem (类似 LightningModule)
│   ├── callback.py           # Callback 基类和 CallbackList
│   ├── context.py            # EngineContext (训练状态管理)
│   ├── outputs.py            # StepOutput, EvalOutput, InferOutput
│   ├── evaluator.py          # 评估器基类
│   ├── objective.py          # 目标函数基类
│   └── inferencer.py         # 推理器基类
│
├── engines/                   # 训练引擎适配层
│   ├── __init__.py
│   ├── base.py               # BaseEngine (引擎抽象基类)
│   ├── fabric_engine.py      # FabricEngineAdapter
│   ├── accelerate_engine.py  # AccelerateEngineAdapter
│   └── factory.py            # 引擎工厂 (create_engine)
│
├── vocoder/                   # Vocoder 任务系统
│   ├── __init__.py
│   ├── system.py             # VocoderTaskSystem (HiFiGAN)
│   ├── vocos_system.py       # VocosTaskSystem
│   ├── datasets.py           # VocoderDataset
│   ├── collators.py          # VocoderCollator
│   └── objectives/           # 损失函数
│       ├── generator_loss.py
│       └── discriminator_loss.py
│
├── callbacks/                 # 内置回调
│   ├── __init__.py
│   ├── checkpoint.py         # CheckpointCallback
│   └── early_stopping.py     # EarlyStoppingCallback
│
└── losses/                    # 通用损失函数
    ├── __init__.py
    ├── generator_loss.py
    ├── discriminator_loss.py
    ├── feature_loss.py
    └── gan_loss.py
```

---

## 核心组件

### 1. Trainer (统一训练器)

**位置**: `soniq/training/trainer.py`

`Trainer` 是训练框架的统一入口，支持在不同引擎间切换：

```python
from soniq.training import Trainer

# 使用 Fabric 引擎
trainer = Trainer(
    engine="fabric",
    run_path="./runs/exp1",
    max_steps=100000,
    gradient_accumulation_steps=4,
    gradient_clip_val=1.0,
    log_interval_steps=200,
    val_interval_steps=2000,
    save_interval_steps=10000,
)

# 使用 Accelerate 引擎
trainer = Trainer(
    engine="accelerate",
    run_path="./runs/exp1",
    max_epochs=100,
    mixed_precision="bf16",
)
```

**主要功能**:
- 引擎选择和管理（Fabric/Accelerate）
- 训练循环控制（按步数或按 epoch）
- 检查点管理（保存/恢复）
- 指标日志记录
- 回调系统集成

---

### 2. BaseEngine (引擎抽象基类)

**位置**: `soniq/training/engines/base.py`

定义统一的训练引擎接口，屏蔽底层框架差异：

```python
from soniq.training.engines import BaseEngine

class BaseEngine(ABC):
    # 核心方法
    - setup(model, optimizer, train_dataloader, val_dataloader)
    - backward(loss)
    - step(optimizer)
    - clip_gradients(model, clip_val, clip_algorithm)
    - save_checkpoint(path, state)
    - load_checkpoint(path, state)
    - save_model(model, path)

    # 日志方法
    - log(metrics, step)
    - log_audio(name, audio, sample_rate, step)
    - log_image(name, image, step)

    # 分布式方法
    - is_main_process()
    - is_local_main_process()
    - gather(tensor)
    - all_reduce(tensor, op)
    - barrier()
    - wait_for_everyone()

    # 属性
    - device
    - precision
    - gradient_accumulation_steps
```

**已实现引擎**:
- `FabricEngineAdapter`: 基于 Lightning Fabric
- `AccelerateEngineAdapter`: 基于 HuggingFace Accelerate

---

### 3. EngineContext (训练上下文)

**位置**: `soniq/training/base/context.py`

集中管理训练的所有状态信息，支持序列化用于断点续训：

```python
from dataclasses import dataclass
from soniq.training import EngineContext

ctx = EngineContext(
    seed=3407,
    num_iterations=100000,
    gradient_accumulation_steps=4,
    log_interval_steps=200,
    val_interval_steps=2000,
    save_interval_steps=10000,
)

# 便利属性
ctx.need_to_log      # 是否需要记录日志
ctx.need_to_validate # 是否需要验证
ctx.need_to_save     # 是否需要保存检查点
ctx.seed_on_rank     # 当前进程的随机种子
```

**状态管理**:
- 训练进度：`iteration`, `epoch`
- 分布式信息：`rank`, `local_rank`, `world_size`
- 路径管理：`run_path`, `ckpt_save_path`, `metrics_path`
- 超参数记录：`hp` 字典

---

### 4. BaseTaskSystem (任务系统基类)

**位置**: `soniq/training/base/system.py`

类似 PyTorch Lightning 的 `LightningModule`，定义任务的训练逻辑：

```python
from soniq.training import BaseTaskSystem, StepOutput

class MyTaskSystem(BaseTaskSystem):
    def __init__(self, config):
        super().__init__(config)
        self.model = MyModel(config)

    def training_step(self, batch, batch_idx) -> StepOutput:
        outputs = self.model(batch)
        loss = outputs["loss"]
        return StepOutput(loss=loss, metrics=outputs, logs={})

    def validation_step(self, batch, batch_idx) -> StepOutput:
        # 类似 training_step
        pass

    def inference_step(self, batch) -> Dict[str, torch.Tensor]:
        # 推理步骤
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-4)
```

**已实现任务系统**:
- `VocoderTaskSystem`: HiFiGAN 声码器训练
- `VocosTaskSystem`: Vocos 声码器训练

---

### 5. Callback System (回调系统)

**位置**: `soniq/training/base/callback.py`

提供训练全生命周期的钩子：

```python
from soniq.training import Callback

class MyCallback(Callback):
    def on_fit_start(self, trainer, ctx):
        print("Training started")

    def on_epoch_end(self, trainer, ctx, epoch):
        print(f"Epoch {epoch} completed")

    def on_train_batch_end(self, trainer, ctx, batch, batch_idx, outputs):
        if ctx.need_to_log:
            print(f"Step {ctx.iteration}: loss = {outputs.loss}")

    def on_validation_end(self, trainer, ctx, outputs, metrics):
        print(f"Validation loss: {metrics['loss']}")
```

**钩子方法**:
- 生命周期：`on_fit_start`, `on_fit_end`
- Epoch 钩子：`on_epoch_start`, `on_epoch_end`
- Batch 钩子：`on_train_batch_start/end`, `on_validation_batch_start/end`
- 反向传播：`on_before_backward`, `on_after_backward`
- 优化器：`on_before_optimizer_step`, `on_after_optimizer_step`
- 检查点：`on_save_checkpoint`, `on_load_checkpoint`

**内置回调**:
- `CheckpointCallback`: 检查点管理
- `EarlyStoppingCallback`: 早停

---

### 6. 引擎工厂

**位置**: `soniq/training/engines/factory.py`

支持动态创建和注册引擎：

```python
from soniq.training.engines import create_engine, register_engine

# 创建引擎
engine = create_engine(
    engine_type="fabric",  # 或 "accelerate"
    ctx=ctx,
    callbacks=callbacks,
    precision="16-mixed",
)

# 注册自定义引擎
@register_engine("my_engine")
class MyCustomEngine(BaseEngine):
    ...
```

---

## 使用示例

### HiFiGAN 训练

```python
from soniq.training import Trainer, VocoderTaskSystem, VocoderConfig
from soniq.training.vocoder import VocoderDataset, VocoderCollator
from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig
from torch.utils.data import DataLoader

# 配置
config = VocoderConfig(
    learning_rate=2e-4,
    betas=(0.8, 0.99),
    segment_size=8192,
    lambda_mel=45.0,
    lambda_adv=1.0,
    lambda_feat_match=2.0,
)

# 模型
generator = HifiGAN(HifiGANConfig())

# 任务系统
system = VocoderTaskSystem(config, generator)

# 数据
train_dataset = VocoderDataset("manifests/train.json", config)
val_dataset = VocoderDataset("manifests/val.json", config)
collator = VocoderCollator(config)

train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collator)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collator)

# 训练器
trainer = Trainer(
    engine="fabric",
    run_path="./runs/hifigan_24k",
    max_steps=100000,
    gradient_accumulation_steps=4,
    log_interval_steps=200,
    val_interval_steps=2000,
    save_interval_steps=10000,
    precision="16-mixed",
)

# 开始训练
trainer.fit(system, train_loader, val_loader)
```

### Vocos 训练

```python
from soniq.training import Trainer, VocosTaskSystem, VocosConfig
from soniq.models.vocoders.vocos import Vocos, VocosConfig as ModelConfig

# 模型配置
model_config = ModelConfig(
    input_channels=128,
    dim=512,
    num_layers=8,
    n_fft=800,
    hop_length=200,
)

# 训练配置
train_config = VocosConfig(
    learning_rate=2e-4,
    lambda_mel=10.0,
    lambda_adv=2.0,
    lambda_fm=2.0,
)

# 模型和系统
vocos = Vocos(model_config)
system = VocosTaskSystem(train_config, vocos)

# 训练器
trainer = Trainer(
    engine="fabric",
    run_path="./runs/vocos_24k",
    max_steps=100000,
)

trainer.fit(system, train_loader, val_loader)
```

---

## 分布式训练

框架自动支持分布式训练，无需修改代码：

```bash
# Fabric (使用 Lightning)
fabric run --devices 8 train.py

# Accelerate (使用 HuggingFace)
accelerate launch --num_processes 8 train.py
```

**分布式特性**:
- DDP (Distributed Data Parallel)
- FSDP (Fully Sharded Data Parallel)
- DeepSpeed
- 混合精度训练 (FP16/BF16)
- 梯度累积
- 梯度同步控制

---

## 检查点管理

### 保存检查点

```python
# 自动保存 (通过 save_interval_steps)
trainer = Trainer(
    save_interval_steps=10000,
    save_last_n=5,  # 保留最近 5 个检查点
)

# 手动保存
trainer.save_model("./checkpoints/my_model")
```

### 恢复训练

```python
trainer.fit(
    system,
    train_loader,
    val_loader,
    ckpt_path="./runs/checkpoints/checkpoint_50000",
)
```

### 检查点结构

```
runs/exp1/
├── checkpoints/
│   ├── checkpoint_10000/
│   │   └── state.ckpt
│   ├── checkpoint_20000/
│   │   └── state.ckpt
│   └── ...
├── weights/
│   ├── step_10000/
│   │   └── model.ckpt
│   └── ...
├── metrics/          # TensorBoard 日志
└── logs/             # 文本日志
```

---

## 损失函数

### GAN 相关损失

```python
from soniq.training.losses import (
    GANDiscriminatorLoss,
    GANGeneratorLoss,
    FeatureMatchingLoss,
    MelSpectrogramLoss,
)

# 判别器损失
disc_loss_fn = GANDiscriminatorLoss()
real_loss, fake_loss = disc_loss_fn(real_outputs, fake_outputs)

# 生成器损失
gen_loss_fn = GANGeneratorLoss()
gen_loss, gen_losses = gen_loss_fn(fake_outputs)

# 特征匹配损失
fm_loss_fn = FeatureMatchingLoss()
fm_loss = fm_loss_fn(fmap_r, fmap_g)

#  mel 谱损失
mel_loss_fn = MelSpectrogramLoss()
mel_loss = mel_loss_fn(audio_hat, audio)
```

---

## 回调示例

### 自定义回调

```python
from soniq.training import Callback

class AudioLoggingCallback(Callback):
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate

    def on_validation_end(self, trainer, ctx, outputs, metrics):
        if not ctx.is_main_process:
            return

        # 记录生成的音频
        for i, output in enumerate(outputs[:4]):
            audio = output.get("audio_hat")
            if audio is not None:
                trainer.engine.log_audio(
                    f"val/audio_{i}",
                    audio,
                    sample_rate=self.sample_rate,
                    step=ctx.iteration,
                )

# 使用回调
trainer = Trainer(
    callbacks=[AudioLoggingCallback(sample_rate=24000)],
)
```

### 早停回调

```python
from soniq.training.callbacks import EarlyStoppingCallback

early_stop = EarlyStoppingCallback(
    monitor="val/loss",
    patience=10,
    min_delta=0.0001,
    mode="min",
)

trainer = Trainer(callbacks=[early_stop])
```

---

## 扩展指南

### 添加新任务系统

1. 继承 `BaseTaskSystem`
2. 实现必需方法

```python
from soniq.training import BaseTaskSystem, StepOutput

class MyTaskSystem(BaseTaskSystem):
    def __init__(self, config):
        super().__init__(config)
        self.model = MyModel(config)
        self.loss_fn = MyLoss()

    def training_step(self, batch, batch_idx) -> StepOutput:
        outputs = self.model(batch)
        loss = self.loss_fn(outputs, batch)
        return StepOutput(loss=loss, metrics={"loss": loss.item()})

    def validation_step(self, batch, batch_idx) -> StepOutput:
        with torch.no_grad():
            outputs = self.model(batch)
            loss = self.loss_fn(outputs, batch)
        return StepOutput(loss=loss, metrics={"val_loss": loss.item()})

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-4)
```

### 添加新引擎

1. 继承 `BaseEngine`
2. 实现所有抽象方法
3. 使用 `@register_engine` 装饰器注册

```python
from soniq.training.engines import BaseEngine, register_engine

@register_engine("my_engine")
class MyCustomEngine(BaseEngine):
    def setup(self, model, optimizer, train_dataloader, val_dataloader):
        # 自定义 setup 逻辑
        return model, optimizer, train_dataloader, val_dataloader

    def backward(self, loss):
        loss.backward()

    # ... 实现其他方法
```

---

## 设计原则

1. **引擎无关性**: 任务系统不依赖特定引擎，通过 `BaseEngine` 接口交互
2. **关注点分离**: 训练循环 (Trainer) vs 任务逻辑 (TaskSystem) vs 引擎实现 (Engine)
3. **回调优先**: 所有副作用操作通过回调实现，保持核心逻辑纯净
4. **类型安全**: 使用 `StepOutput` 等数据类确保输出格式一致
5. **分布式透明**: 任务系统代码无需修改即可支持分布式

---

## 与其他框架对比

| 特性 | Soniq | PyTorch Lightning | HuggingFace Trainer |
|------|-------|-------------------|---------------------|
| 引擎后端 | Fabric + Accelerate | 自研 | Accelerate |
| 分布式支持 | ✓ | ✓ | ✓ |
| 自定义训练循环 | ✓ | 有限 | 困难 |
| GAN 多优化器 | ✓ | ✓ | 困难 |
| 检查点管理 | ✓ | ✓ | ✓ |
| 回调系统 | ✓ | ✓ | ✓ |
| 音频专用 | ✓ | ✗ | ✗ |

---

## 注意事项

1. **不要直接引用** `trainer/core` 下的代码（将被删除）
2. 任务系统应继承 `BaseTaskSystem`，而不是直接使用引擎
3. 分布式训练时，确保使用 `ctx.is_main_process` 控制日志和保存
4. 梯度累积通过引擎自动处理，任务系统无需关心
5. 检查点恢复时，`iteration` 会从检查点继续

---

## API 参考

### Trainer

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| engine | str | "accelerate" | 引擎类型 |
| run_path | str | "./runs/default" | 运行目录 |
| max_steps | int | None | 最大训练步数 |
| max_epochs | int | None | 最大 epoch 数 |
| gradient_accumulation_steps | int | 1 | 梯度累积步数 |
| gradient_clip_val | float | 1.0 | 梯度裁剪值 |
| log_interval_steps | int | 200 | 日志间隔 |
| val_interval_steps | int | 2000 | 验证间隔 |
| save_interval_steps | int | None | 检查点保存间隔 |
| save_last_n | int | 2 | 保留最近 N 个检查点 |

### EngineContext

| 属性 | 说明 |
|------|------|
| iteration | 当前迭代次数 |
| epoch | 当前 epoch |
| rank | 全局进程索引 |
| local_rank | 本地进程索引 |
| world_size | 进程总数 |
| need_to_log | 是否需要记录日志 |
| need_to_validate | 是否需要验证 |
| need_to_save | 是否需要保存检查点 |

### StepOutput

| 字段 | 类型 | 说明 |
|------|------|------|
| loss | torch.Tensor | 损失值 |
| metrics | Dict | 其他指标 |
| logs | Dict | 日志信息 |
