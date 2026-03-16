# 使用 Hugging Face CLI 管理模型

Soniq 使用 Hugging Face Hub 进行模型分享。推荐使用 `huggingface-cli` 命令行工具来上传和下载模型。

## 安装

```bash
pip install huggingface_hub
```

## 登录

首次使用前需要登录：

```bash
huggingface-cli login
```

或者使用 token 登录：

```bash
huggingface-cli login --token YOUR_TOKEN
```

## 下载模型

### 方法 1: 使用 Python

```python
from soniq import load_from_hub

# 下载到缓存目录
model_path = load_from_hub("username/model-name")

# 下载到指定目录
model_path = load_from_hub(
    "username/model-name",
    local_dir="./models/my-model",
)
```

### 方法 2: 使用命令行

```bash
# 下载整个模型仓库
huggingface-cli download username/model-name --local-dir ./models/my-model

# 下载特定文件
huggingface-cli download username/model-name model.safetensors --local-dir ./models
```

## 上传模型

### 准备模型文件

确保模型目录包含以下文件：

```
my-model/
├── model.safetensors    # 或 pytorch_model.bin
├── config.json          # 模型配置
├── README.md            # 模型说明（可选但推荐）
└── ...
```

### 使用命令行上传

```bash
# 上传整个目录
huggingface-cli upload username/model-name ./my-model .

# 上传单个文件
huggingface-cli upload username/model-name ./model.safetensors model.safetensors

# 上传到私有仓库
huggingface-cli upload username/model-name ./my-model . --private
```

### 创建新仓库

可以通过以下方式创建新仓库：

1. 在 [huggingface.co](https://huggingface.co) 网站上手动创建
2. 使用命令行创建：

```bash
huggingface-cli repo create model-name
```

## 常用命令

| 命令 | 说明 |
|------|------|
| `huggingface-cli login` | 登录到 Hugging Face Hub |
| `huggingface-cli whoami` | 查看当前用户信息 |
| `huggingface-cli download <repo_id>` | 下载模型 |
| `huggingface-cli upload <repo_id> <local_path> <remote_path>` | 上传模型 |
| `huggingface-cli repo create <name>` | 创建新仓库 |
| `huggingface-cli logout` | 退出登录 |

## 从 Soniq 加载模型

```python
from soniq.models.vocoders import HifiGAN

# 从本地目录加载
model = HifiGAN.from_pretrained("./models/my-model")

# 从 Hugging Face Hub 加载
model = HifiGAN.from_pretrained("username/model-name")
```

## 模型卡建议

上传模型时，建议包含 `README.md` 文件，包含：

- 模型描述
- 训练数据
- 使用方法
- 许可证信息

示例：

```markdown
---
language:
- en
- zh
license: apache-2.0
tags:
- soniq
- audio
- vocoder
---

# HiFiGAN 声码器

这是一个使用 Soniq 训练的 HiFiGAN 声码器模型。

## 使用方法

```python
from soniq.models.vocoders import HifiGAN

model = HifiGAN.from_pretrained("username/hifigan-model")
```

## 训练详情

该模型使用 24kHz 采样率的音频数据训练。
```
