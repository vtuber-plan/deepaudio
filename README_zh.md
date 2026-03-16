# Soniq

**Soniq** 是一个基于 PyTorch 的音频机器学习库，提供采用 Transformers 风格 API 的最先进语音和音频模型。

[![PyPI](https://img.shields.io/pypi/v/soniq.svg)](https://pypi.org/project/soniq/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 特性

- **Transformers 风格 API** - Hugging Face 用户熟悉的接口
- **Lightning Fabric** - 灵活高效的训练，支持分布式
- **丰富模型库** - 声码器、TTS、ASR、SVC 等
- **多语言文档** - 支持中文和英文文档

## 安装

### 从 PyPI 安装

```bash
pip install soniq
```

### 从源码安装

```bash
git clone https://github.com/vtuber-plan/soniq.git
cd soniq
pip install -e .
```

## 快速开始

### 加载预训练模型

```python
from soniq import AutoModel, MelPipeline

# 加载模型
model = AutoModel.from_pretrained("soniq/hifigan-24k")

# 创建流水线
pipeline = MelPipeline(sample_rate=24000, n_mel=80)

# 运行推理
import torchaudio
audio, sr = torchaudio.load("audio.wav")
mel = pipeline(audio)
output = model(mel.unsqueeze(0))
```

### 训练模型

```python
from soniq.training import FabricTrainer
from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig
from soniq.datasets import BaseDataset, BaseCollator, build_dataloader

# 模型
config = HifiGANConfig()
model = HifiGAN(config)

# 数据集
dataset = BaseDataset(
    metadata_path="data/train.json",
    feature_dirs={"mel": "data/mels", "wav": "data/wavs"},
)
dataloader = build_dataloader(dataset, BaseCollator(), batch_size=32)

# 训练器
trainer = FabricTrainer(
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    max_epochs=100,
)

# 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
trainer.fit(model, dataloader, optimizer=optimizer)
```

## 支持的模型

### 声码器

| 模型 | 描述 | 采样率 |
|------|------|--------|
| HiFi-GAN | 高保真 GAN 声码器 | 16k, 24k, 44k, 48k |
| MelGAN | 快速 GAN 声码器 | 16k, 24k |
| BigVGAN | 通用神经声码器 | 24k, 44k |
| WaveNet | 自回归声码器 | 16k, 24k |

### 文本转语音

| 模型 | 描述 |
|------|------|
| VITS | 条件变分自编码器 |
| FastSpeech2 | 快速高质量 TTS |
| VALLE | VALL-E 神经编解码语言模型 |

### 其他

| 任务 | 模型 |
|------|------|
| ASR | Whisper |
| 基频检测 | CREPE, RMVPE, Harvest |
| 内容编码器 | HubERT, ContentVec |

## 文档

- [英文文档](docs/en/README.md)
- [中文文档](docs/zh/README.md)

### 快速链接

- [安装指南](docs/zh/installation.md)
- [快速开始](docs/zh/quickstart.md)
- [教程](docs/zh/tutorials/)
- [API 参考](docs/zh/api/)

## 项目结构

```
soniq/
├── soniq/                  # 主包
│   ├── models/             # 模型定义
│   ├── pipelines/          # 音频处理流水线
│   ├── features/           # 特征提取
│   ├── datasets/           # 数据集类
│   ├── training/           # 训练工具
│   ├── processors/         # 数据预处理
│   ├── modules/            # 可复用神经模块
│   ├── text/               # 文本处理
│   └── utils/              # 工具函数
├── docs/                   # 文档
│   ├── en/                 # 英文文档
│   └── zh/                 # 中文文档
├── config/                 # 配置文件
├── bins/                   # 训练脚本
└── examples/               # 示例代码
```

## 许可证

Soniq 使用 MIT 许可证发布。

## 致谢

本项目的灵感来源于以下优秀项目：

- [Amphion](https://github.com/open-mmlab/Amphion) - 音频、音乐和语音生成工具包
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - 最先进的机器学习库
- [Lightning AI](https://lightning.ai/) - PyTorch Lightning 和 Lightning Fabric

## 贡献

我们欢迎贡献！请随时提交问题和拉取请求。

## 联系方式

- GitHub Issues: [报告错误或请求功能](https://github.com/vtuber-plan/soniq/issues)
