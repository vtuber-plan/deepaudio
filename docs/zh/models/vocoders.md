# 声码器模型 (Vocoders)

声码器将梅尔频谱图转换为音频波形。

## 支持的模型

| 模型 | 描述 | 采样率 |
|------|------|--------|
| HiFi-GAN | 高保真 GAN 声码器 | 16k, 24k, 44k, 48k |
| MelGAN | 快速 GAN 声码器 | 16k, 24k |
| BigVGAN | 通用神经声码器 | 24k, 44k |
| WaveNet | 自回归声码器 | 16k, 24k |

## 使用预训练模型

### 加载 HiFi-GAN

```python
from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig
from soniq.pipelines import MelPipeline
import torchaudio

# 加载模型
config = HifiGANConfig.from_pretrained("soniq/hifigan-24k")
model = HifiGAN.from_pretrained("soniq/hifigan-24k")
model.eval()

# 准备输入
audio, sr = torchaudio.load("reference.wav")
mel_pipeline = MelPipeline(sample_rate=24000, n_mel=80)
mel = mel_pipeline(audio)

# 生成波形
with torch.no_grad():
    output = model(mel.unsqueeze(0))

# 保存输出
torchaudio.save("output.wav", output.cpu(), sample_rate=24000)
```

## 训练声码器

### 准备数据集

创建元数据 JSON 文件：

```json
[
    {"id": "utterance_001", "duration": 2.5, "speaker": "spk1"},
    {"id": "utterance_002", "duration": 3.1, "speaker": "spk2"}
]
```

### 训练脚本

```python
import torch
from soniq.training import FabricTrainer
from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig
from soniq.datasets import BaseDataset, BaseCollator, build_dataloader

# 配置
config = HifiGANConfig(
    inter_channels=128,
    upsample_rates=[8, 8, 4, 2],
    upsample_initial_channel=512,
)

# 模型
model = HifiGAN(config)

# 数据集
dataset = BaseDataset(
    metadata_path="data/train.json",
    feature_dirs={"mel": "data/mels", "wav": "data/wavs"},
    sample_rate=24000,
)

# 数据加载器
dataloader = build_dataloader(
    dataset,
    collator=BaseCollator(),
    batch_size=16,
    num_workers=4,
)

# 训练器
trainer = FabricTrainer(
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    max_epochs=100,
)

# 训练步骤函数
def train_step(model, batch):
    mel = batch["mel"]
    audio = batch["wav"]
    generated = model(mel)
    loss = torch.nn.functional.l1_loss(generated, audio)
    return loss

# 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
trainer.fit(
    model,
    dataloader,
    optimizer=optimizer,
    train_step_fn=train_step,
)
```

## 训练技巧

1. **使用高质量训练数据** - 干净的音频，最小噪声
2. **训练足够长的时间** - HiFi-GAN 至少需要 100k steps
3. **使用混合精度** - `precision="16-mixed"` 可以加快训练
4. **监控验证损失** - 当损失趋于平稳时停止训练
