# 快速开始

本指南将帮助你在几分钟内开始使用 Soniq。

## 加载预训练模型

### 加载 HiFi-GAN 声码器

```python
from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig
from soniq.pipelines import MelPipeline
import torchaudio

# 加载配置和模型
config = HifiGANConfig.from_pretrained("soniq/hifigan-24k")
model = HifiGAN.from_pretrained("soniq/hifigan-24k")

# 创建梅尔流水线
mel_pipeline = MelPipeline(
    sample_rate=24000,
    n_fft=1024,
    n_mel=80,
    hop_length=256,
)

# 加载音频并提取梅尔频谱
audio, sr = torchaudio.load("audio.wav")
mel = mel_pipeline(audio)

# 生成波形
output = model(mel.unsqueeze(0))
```

### 使用 AutoModel

```python
from soniq.models import AutoModel

# 自动加载正确的模型类
model = AutoModel.from_pretrained("soniq/hifigan-24k")
```

## 训练你的第一个模型

### 基本训练设置

```python
from soniq.training import FabricTrainer
from soniq.models.vocoders.hifigan import HifiGAN, HifiGANConfig
from soniq.datasets import build_dataloader, BaseDataset, BaseCollator

# 创建模型
config = HifiGANConfig()
model = HifiGAN(config)

# 创建训练器
trainer = FabricTrainer(
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    max_epochs=100,
)

# 准备数据集和数据加载器
dataset = BaseDataset(
    metadata_path="data/train.json",
    feature_dirs={"mel": "data/mels", "wav": "data/wavs"},
)
dataloader = build_dataloader(
    dataset,
    collator=BaseCollator(),
    batch_size=32,
)

# 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
trainer.fit(model, dataloader, optimizer=optimizer)
```

## 音频处理流水线

### 音频流水线

```python
from soniq.pipelines import AudioPipeline

# 创建流水线
pipeline = AudioPipeline(sample_rate=24000, mono=True)

# 加载和预处理音频
audio = pipeline("path/to/audio.wav")
print(audio.shape)  # (1, num_samples)
```

### 梅尔流水线

```python
from soniq.pipelines import MelPipeline

# 创建流水线
pipeline = MelPipeline(
    sample_rate=24000,
    n_fft=1024,
    n_mel=80,
    hop_length=256,
)

# 提取梅尔频谱
mel = pipeline(audio)
print(mel.shape)  # (80, num_frames)
```

## 特征提取

### 声学特征

```python
from soniq.processors import AcousticExtractor

# 创建提取器
extractor = AcousticExtractor(sample_rate=24000)

# 提取特征
mel = extractor.extract_mel(audio)
spectrogram = extractor.extract_spectrogram(audio)
mfcc = extractor.extract_mfcc(audio)
f0 = extractor.extract_f0(audio)
energy = extractor.extract_energy(audio)
```

### 音素提取

```python
from soniq.processors import PhoneExtractor

# 创建提取器
extractor = PhoneExtractor()

# 提取音素
result = extractor("hello world", return_ids=True)
print(result["phonemes"])  # "h ə l oʊ w ɝ l d"
print(result["ids"])  # [20, 35, 15, ...]
```

## 下一步

- 查看 [教程](tutorials/) 获取更详细的指南
- 阅读 [API 参考](api/) 获取完整文档
- 探索 [模型库](models/) 了解所有支持的模型
