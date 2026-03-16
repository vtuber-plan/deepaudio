# Soniq Hugging Face Hub 集成实现总结

## 新增文件

### 1. 核心模块

| 文件 | 说明 |
|------|------|
| `soniq/utils/hub_utils.py` | Hugging Face Hub 工具函数 |
| `soniq/hub.py` | HuggingFaceHub 高级管理类 |
| `soniq/hub_cli.py` | 命令行工具 |
| `examples/hub/huggingface_hub_example.py` | 使用示例 |
| `docs/zh/hub.md` | 中文文档 |

### 2. 修改的文件

| 文件 | 修改内容 |
|------|----------|
| `soniq/models/base/modeling_base.py` | 添加 `push_to_hub` 方法，增强 `from_pretrained` |
| `soniq/__init__.py` | 导出 `HuggingFaceHub`, `HubModelInfo` |
| `soniq/utils/__init__.py` | 导出 hub_utils 函数 |
| `pyproject.toml` | 添加 CLI 入口 |

## 功能概览

### 1. Python API

```python
from soniq import HuggingFaceHub

hub = HuggingFaceHub()
hub.login(token="hf_xxx")

# 上传模型
url = hub.upload_model(
    model_path="./checkpoints/hifigan",
    repo_id="myuser/my-model",
    model_type="vocoder",
)

# 下载模型
model_path = hub.download_model(
    repo_id="myuser/my-model",
    local_dir="./models",
)

# 直接从 Hub 加载模型
from soniq.models.vocoders import HifiGAN
model = HifiGAN.from_pretrained("myuser/my-model")
```

### 2. 模型内置方法

```python
# 使用模型类的 push_to_hub 方法
model.push_to_hub(
    repo_id="myuser/my-model",
    model_type="vocoder",
)
```

### 3. 命令行工具

```bash
# 安装后使用
pip install -e .

# 登录
soniq-hub login --token hf_xxx

# 上传
soniq-hub upload ./checkpoints/hifigan --repo-id myuser/my-model --model-type vocoder

# 下载
soniq-hub download myuser/my-model --local-dir ./models

# 列出模型
soniq-hub list --author myuser

# 获取模型信息
soniq-hub info myuser/my-model
```

## 支持的模型类型

- `vocoder` - 声码器
- `tts` - 文本转语音
- `svc` - 歌声转换
- `svs` - 歌声合成
- `vc` - 语音转换
- `asr` - 自动语音识别
- `codec` - 音频编解码器
- `f0` - 基频检测
- `speaker_encoder` - 说话人嵌入

## 使用步骤

### 1. 获取 Token
访问 https://huggingface.co/settings/tokens 创建 API Token

### 2. 安装/更新 Soniq
```bash
cd /data/wangjun/github/deepaudio
pip install -e .
```

### 3. 登录
```bash
soniq-hub login --token hf_xxx
```

### 4. 上传/下载模型
```bash
# 上传
soniq-hub upload ./path/to/model --repo-id username/model-name --model-type vocoder

# 下载
soniq-hub download username/model-name --local-dir ./models
```

## HubUtils 函数列表

- `upload_model_to_hub()` - 上传模型
- `download_model_from_hub()` - 下载模型
- `download_file_from_hub()` - 下载单个文件
- `list_hub_models()` - 列出模型
- `repo_exists()` - 检查仓库是否存在
- `get_repo_files()` - 获取文件列表
- `check_login_status()` - 检查登录状态
- `create_model_card()` - 创建 Model Card
- `save_pretrained_for_hub()` - 保存模型为 Hub 格式
- `upload_file_to_hub()` - 上传单个文件

## HuggingFaceHub 类方法

- `login()` / `logout()` - 认证
- `check_auth()` - 检查认证状态
- `upload_model()` - 上传模型
- `download_model()` - 下载模型
- `download_file()` - 下载文件
- `list_models()` - 列出模型
- `get_model_info()` - 获取模型信息
- `repo_exists()` - 检查仓库
- `create_repo()` - 创建仓库
- `delete_repo()` - 删除仓库
- `upload_file()` - 上传文件
- `save_and_upload()` - 保存并上传

## 下一步建议

1. 在 Hugging Face 上创建组织账号（可选）
2. 测试上传一个小模型
3. 添加模型卡片自定义模板
4. 考虑添加自动上传回调（训练完成后自动上传）
