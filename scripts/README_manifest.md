# 音频数据集 Manifest 生成工具

通用的音频数据集 manifest 生成脚本，支持多种数据集结构和音频格式。

## 功能特性

- **多种数据集结构支持**:
  - 子目录模式：自动扫描子目录（适用于多说话人/多语言数据集）
  - 扁平模式：忽略子目录分类，处理所有音频文件

- **灵活的元数据**:
  - 支持将子目录名作为 speaker 或 language 字段
  - 自定义 ID 前缀

- **高效处理**:
  - 多进程并行处理
  - 可选择跳过时长计算（快速模式）

- **多种音频格式**:
  - 支持 .wav, .flac, .mp3, .ogg, .m4a, .aiff, .wma

## 使用示例

### 1. 基本用法（自动扫描子目录）

```bash
python scripts/prepare_manifest.py --dataset-dir /path/to/dataset
```

### 2. 指定子目录处理

```bash
# 指定要处理的子目录
python scripts/prepare_manifest.py --dataset-dir /path/to/dataset \
    --subdirs chinese english japanese

# 子目录作为 language 字段
python scripts/prepare_manifest.py --dataset-dir /path/to/dataset \
    --subdir-as-language

# 子目录作为 speaker 字段
python scripts/prepare_manifest.py --dataset-dir /path/to/dataset \
    --subdir-as-speaker
```

### 3. 扁平模式（忽略子目录分类）

```bash
# 处理目录中所有音频文件
python scripts/prepare_manifest.py --dataset-dir /path/to/dataset --flat

# 递归搜索所有子目录
python scripts/prepare_manifest.py --dataset-dir /path/to/dataset \
    --flat --recursive
```

### 4. 限制样本数量（测试用）

```bash
python scripts/prepare_manifest.py --dataset-dir /path/to/dataset \
    --max-samples 100
```

### 5. 调整验证集比例

```bash
# 10% 作为验证集
python scripts/prepare_manifest.py --dataset-dir /path/to/dataset \
    --val-split 0.1
```

### 6. 多进程加速

```bash
python scripts/prepare_manifest.py --dataset-dir /path/to/dataset \
    --num-workers 16
```

### 7. 快速模式（跳过时长计算）

```bash
python scripts/prepare_manifest.py --dataset-dir /path/to/dataset \
    --no-compute-duration
```

### 8. 指定音频格式

```bash
python scripts/prepare_manifest.py --dataset-dir /path/to/dataset \
    --extensions .wav .flac
```

## 输出格式

生成的 manifest 为 JSON 格式，每个条目包含：

```json
[
  {
    "id": "chinese_utt_001",
    "audio_path": "/path/to/audio.wav",
    "duration": 3.5,
    "language": "chinese"
  },
  {
    "id": "spk1_utt_002",
    "audio_path": "/path/to/audio2.wav",
    "duration": 2.1,
    "speaker": "spk1"
  }
]
```

## 完整参数

```
--dataset-dir DATASET_DIR
    数据集根目录（必需）

--output-dir OUTPUT_DIR
    输出目录（默认为 dataset-dir/manifests）

--subdirs SUBDIRS [SUBDIRS ...]
    要处理的子目录列表（默认自动扫描所有子目录）

--flat
    扁平模式：忽略子目录分类

--recursive
    递归搜索子目录（仅在 flat 模式下有效）

--subdir-as-speaker
    将子目录名作为 speaker 字段

--subdir-as-language
    将子目录名作为 language 字段

--id-prefix ID_PREFIX
    manifest ID 的前缀

--val-split VAL_SPLIT
    验证集比例（默认 0.05 = 5%）

--max-samples MAX_SAMPLES
    每个子目录的最大样本数（默认无限制）

--num-workers NUM_WORKERS
    并行处理的工作进程数（默认 4）

--no-compute-duration
    跳过时长计算（快速生成，duration 字段为 0）

--extensions EXTENSIONS [EXTENSIONS ...]
    音频文件扩展名（默认：.wav .flac .mp3 .ogg .m4a）
```

## 使用案例

### Genshin 数据集

```bash
python scripts/prepare_manifest.py \
    --dataset-dir /data/wangjun/datasets/genshin_dataset_2022_11_12 \
    --output-dir ./data/genshin \
    --subdir-as-language \
    --num-workers 8
```

### LibriTTS 数据集

```bash
python scripts/prepare_manifest.py \
    --dataset-dir /data/wangjun/datasets/LibriTTS/train-clean-360 \
    --output-dir ./data/libritts_train \
    --subdir-as-speaker \
    --recursive \
    --num-workers 16
```

### 单说话人数据集

```bash
python scripts/prepare_manifest.py \
    --dataset-dir /data/wangjun/datasets/single_speaker \
    --output-dir ./data/single_speaker \
    --flat \
    --recursive
```
