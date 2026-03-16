# Soniq 文档

欢迎使用 Soniq 文档。

Soniq 是一个基于 PyTorch 的音频机器学习库，提供采用 Transformers 风格 API 的最先进语音和音频模型。

## 目录

### 快速开始

- [安装指南](installation.md) - 如何安装 Soniq
- [快速开始](quickstart.md) - 快速上手 Soniq

### 教程

- [声码器教程](tutorials/vocoder.md) - 训练和使用神经声码器
- [TTS 教程](tutorials/tts.md) - 文本转语音合成
- [ASR 教程](tutorials/asr.md) - 自动语音识别

### 模型

- [声码器](models/vocoders.md) - 神经声码器模型 (HiFi-GAN, MelGAN, BigVGAN)
- [TTS 模型](models/tts.md) - 文本转语音模型 (VITS, FastSpeech2)
- [ASR 模型](models/asr.md) - 自动语音识别模型

### API 参考

- [模型 API](api/models.md) - 模型类和配置
- [流水线 API](api/pipelines.md) - 音频处理流水线
- [训练 API](api/training.md) - 使用 Lightning Fabric 的训练工具
- [数据集 API](api/datasets.md) - 数据集类和批处理

### 指南

- [配置指南](guides/configuration.md) - 如何配置模型和训练
- [自定义模型](guides/custom_models.md) - 创建你自己的模型
- [分布式训练](guides/distributed_training.md) - 多 GPU 训练指南

---

## 快速示例

```python
from soniq import AutoModel, MelPipeline

# 加载预训练模型
model = AutoModel.from_pretrained("soniq/hifigan-24k")

# 创建流水线
pipeline = MelPipeline(sample_rate=24000, n_mel=80)

# 运行推理
import torchaudio
audio, sr = torchaudio.load("audio.wav")
mel = pipeline(audio)
output = model(mel.unsqueeze(0))
```

## 支持的模型

| 任务 | 模型 |
|------|------|
| 声码器 | HiFi-GAN, MelGAN, BigVGAN, WaveNet, DiffWave |
| TTS | VITS, FastSpeech2, VALLE, NaturalSpeech2 |
| ASR | Whisper |
| 基频检测 | CREPE, RMVPE, Harvest |

## 许可证

Soniq 使用 MIT 许可证发布。

## 致谢

- [Amphion](https://github.com/open-mmlab/Amphion)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Lightning AI](https://lightning.ai/)
