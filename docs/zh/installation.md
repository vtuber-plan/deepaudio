# 安装指南

本指南说明如何安装 Soniq。

## 系统要求

- Python >= 3.8
- PyTorch >= 1.13
- torchaudio >= 0.14

## 安装方法

### 方法一：从 PyPI 安装

```bash
pip install soniq
```

### 方法二：从源码安装

```bash
# 克隆仓库
git clone https://github.com/vtuber-plan/soniq.git
cd soniq

# 安装包
pip install -e .
```

### 方法三：安装开发依赖

```bash
pip install -e ".[dev]"
```

## 可选依赖

### 高质量音频处理

```bash
pip install librosa soundfile
```

### G2P (图文转换)

```bash
pip install g2p_en  # 英语
pip install pypinyin  # 中文
```

### 训练

```bash
pip install lightning tensorboard wandb
```

## 验证安装

```python
import soniq
print(soniq.__version__)
# 应输出：0.1.0
```

## 平台支持

| 平台 | 支持状态 |
|------|---------|
| Linux | ✅ |
| macOS | ✅ |
| Windows | ✅ |
| GPU (CUDA) | ✅ |
| CPU | ✅ |

## 故障排除

### CUDA 内存不足

减小批量大小或使用梯度累积：

```python
trainer = FabricTrainer(
    gradient_accumulation_steps=4,  # 在 4 个步骤上累积梯度
)
```

### 导入错误

确保已安装所有必需的依赖：

```bash
pip install torch torchaudio transformers lightning
```

### 音频加载问题

确保音频文件为支持的格式（WAV、FLAC、MP3）。
