#!/usr/bin/env python
# coding=utf-8
"""
通用音频数据集 manifest 生成工具

支持多种数据集结构和音频格式，可自定义 manifest 字段。

Usage:
    # 基本用法 - 自动扫描子目录
    python scripts/prepare_manifest.py --dataset-dir /path/to/dataset

    # 指定子目录（多说话人/多语言）
    python scripts/prepare_manifest.py --dataset-dir /path/to/dataset --subdirs spk1 spk2 spk3

    # 指定验证集比例
    python scripts/prepare_manifest.py --dataset-dir /path/to/dataset --val-split 0.1

    # 限制样本数量（测试用）
    python scripts/prepare_manifest.py --dataset-dir /path/to/dataset --max-samples 1000

    # 多进程加速处理
    python scripts/prepare_manifest.py --dataset-dir /path/to/dataset --num-workers 8
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import functools

import soundfile as sf


# 支持的音频格式
AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".wma"]


@dataclass
class ManifestEntry:
    """Manifest 条目."""
    id: str
    audio_path: str
    duration: float
    extra_fields: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典."""
        result = {
            "id": self.id,
            "audio_path": self.audio_path,
            "duration": round(self.duration, 4),
        }
        if self.extra_fields:
            result.update(self.extra_fields)
        return result


def get_audio_duration(path: str) -> float:
    """
    获取音频文件时长（秒）。

    Args:
        path: 音频文件路径。

    Returns:
        时长（秒），失败返回 0.0。
    """
    try:
        info = sf.info(path)
        return info.duration
    except Exception as e:
        print(f"Warning: Error reading {path}: {e}")
        return 0.0


def find_audio_files(
    directory: Path,
    extensions: List[str] = None,
    recursive: bool = False,
) -> List[Path]:
    """
    在目录中查找音频文件。

    Args:
        directory: 搜索目录。
        extensions: 文件扩展名列表。
        recursive: 是否递归搜索子目录。

    Returns:
        音频文件路径列表。
    """
    if extensions is None:
        extensions = AUDIO_EXTENSIONS

    audio_files = []
    for ext in extensions:
        if recursive:
            audio_files.extend(directory.rglob(f"*{ext}"))
        else:
            audio_files.extend(directory.glob(f"*{ext}"))

    # 也查找大写扩展名
    for ext in extensions:
        upper_ext = ext.upper()
        if upper_ext != ext:
            if recursive:
                audio_files.extend(directory.rglob(f"*{upper_ext}"))
            else:
                audio_files.extend(directory.glob(f"*{upper_ext}"))

    return sorted(audio_files)


def create_entry(
    audio_path: Path,
    prefix: str = "",
    compute_duration: bool = True,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Optional[ManifestEntry]:
    """
    创建 manifest 条目。

    Args:
        audio_path: 音频文件路径。
        prefix: ID 前缀。
        compute_duration: 是否计算时长。
        extra_fields: 额外字段。

    Returns:
        ManifestEntry 或 None（如果处理失败）。
    """
    # 生成 ID
    file_id = audio_path.stem
    if prefix:
        file_id = f"{prefix}_{file_id}"

    # 计算时长
    duration = 0.0
    if compute_duration:
        duration = get_audio_duration(str(audio_path))
        if duration <= 0:
            print(f"Warning: Skipping {audio_path} (invalid duration)")
            return None

    # 构建额外字段
    fields = extra_fields.copy() if extra_fields else {}

    return ManifestEntry(
        id=file_id,
        audio_path=str(audio_path),
        duration=duration,
        extra_fields=fields,
    )


def process_file_with_metadata(
    audio_path: Path,
    prefix: str,
    speaker_name: Optional[str] = None,
    language_name: Optional[str] = None,
    compute_duration: bool = True,
) -> Optional[ManifestEntry]:
    """处理单个文件并添加元数据。"""
    extra_fields = {}
    if speaker_name:
        extra_fields["speaker"] = speaker_name
    if language_name:
        extra_fields["language"] = language_name

    return create_entry(
        audio_path=audio_path,
        prefix=prefix,
        compute_duration=compute_duration,
        extra_fields=extra_fields,
    )


class ManifestBuilder:
    """Manifest 构建器."""

    def __init__(
        self,
        dataset_dir: str,
        output_dir: str,
        val_split: float = 0.05,
        max_samples: Optional[int] = None,
        num_workers: int = 4,
        compute_duration: bool = True,
        file_extensions: List[str] = None,
        id_prefix: str = "",
    ):
        """
        初始化构建器。

        Args:
            dataset_dir: 数据集根目录。
            output_dir: 输出目录。
            val_split: 验证集比例（0-1）。
            max_samples: 最大样本数（None 表示全部）。
            num_workers: 并行处理的工作进程数。
            compute_duration: 是否计算音频时长。
            file_extensions: 支持的音频格式。
            id_prefix: manifest ID 的前缀。
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.val_split = val_split
        self.max_samples = max_samples
        self.num_workers = num_workers
        self.compute_duration = compute_duration
        self.file_extensions = file_extensions or AUDIO_EXTENSIONS
        self.id_prefix = id_prefix

    def build_from_subdirectories(
        self,
        subdirs: List[str] = None,
        subdir_as_speaker: bool = True,
        subdir_as_language: bool = False,
    ) -> tuple:
        """
        从子目录构建 manifest。

        Args:
            subdirs: 要处理的子目录列表（None 表示自动扫描）。
            subdir_as_speaker: 是否将子目录名作为 speaker 字段。
            subdir_as_language: 是否将子目录名作为 language 字段。

        Returns:
            (train_manifest_path, val_manifest_path)
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 如果没有指定子目录，自动扫描
        if subdirs is None:
            subdirs = [
                d.name for d in self.dataset_dir.iterdir()
                if d.is_dir() and not d.name.startswith('.')
            ]
            subdirs.sort()
            print(f"Auto-detected subdirectories: {subdirs}")

        train_entries = []
        val_entries = []

        for subdir in subdirs:
            subdir_path = self.dataset_dir / subdir
            if not subdir_path.exists():
                print(f"Warning: Directory {subdir} does not exist, skipping...")
                continue

            print(f"\nProcessing {subdir}...")

            # 构建元数据
            speaker_name = subdir if subdir_as_speaker else None
            language_name = subdir if subdir_as_language else None
            entry_prefix = f"{self.id_prefix}_{subdir}" if self.id_prefix else subdir

            # 查找音频文件
            audio_files = find_audio_files(
                subdir_path,
                extensions=self.file_extensions,
                recursive=False,
            )
            print(f"  Found {len(audio_files)} audio files")

            if self.max_samples:
                audio_files = audio_files[:self.max_samples]
                print(f"  Limited to {len(audio_files)} samples")

            # 处理文件
            entries = self._process_files_parallel(
                audio_files=audio_files,
                prefix=entry_prefix,
                speaker_name=speaker_name,
                language_name=language_name,
            )

            # 划分训练集和验证集
            for i, entry in enumerate(entries):
                if entry is None:
                    continue
                # 按固定间隔选择验证集样本
                if self.val_split > 0 and i % int(1 / self.val_split) == 0:
                    val_entries.append(entry)
                else:
                    train_entries.append(entry)

        print(f"\nTotal entries:")
        print(f"  Train: {len(train_entries)}")
        print(f"  Val: {len(val_entries)}")

        return self._save_manifests(train_entries, val_entries)

    def build_flat(
        self,
        recursive: bool = False,
        path_as_id: bool = False,
    ) -> tuple:
        """
        从扁平目录结构构建 manifest（无子目录分类）。

        Args:
            recursive: 是否递归搜索子目录。
            path_as_id: 是否使用完整路径作为 ID。

        Returns:
            (train_manifest_path, val_manifest_path)
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 查找所有音频文件
        audio_files = find_audio_files(
            self.dataset_dir,
            extensions=self.file_extensions,
            recursive=recursive,
        )
        print(f"Found {len(audio_files)} audio files")

        if self.max_samples:
            audio_files = audio_files[:self.max_samples]
            print(f"Limited to {len(audio_files)} samples")

        # 处理文件
        entries = []
        print("Processing files...")

        if self.num_workers > 1:
            # 并行处理
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for audio_path in audio_files:
                    prefix = self.id_prefix
                    if path_as_id:
                        # 使用相对路径作为 ID 前缀
                        rel_path = audio_path.relative_to(self.dataset_dir)
                        prefix = f"{prefix}_{rel_path.parent}" if prefix else str(rel_path.parent)

                    future = executor.submit(
                        create_entry,
                        audio_path=audio_path,
                        prefix=prefix,
                        compute_duration=self.compute_duration,
                    )
                    futures.append(future)

                for future in futures:
                    entry = future.result()
                    if entry:
                        entries.append(entry)
        else:
            # 单线程处理
            for audio_path in audio_files:
                entry = create_entry(
                    audio_path=audio_path,
                    prefix=self.id_prefix,
                    compute_duration=self.compute_duration,
                )
                if entry:
                    entries.append(entry)

        # 划分训练集和验证集
        train_entries = []
        val_entries = []

        for i, entry in enumerate(entries):
            if self.val_split > 0 and i % int(1 / self.val_split) == 0:
                val_entries.append(entry)
            else:
                train_entries.append(entry)

        print(f"\nTotal entries:")
        print(f"  Train: {len(train_entries)}")
        print(f"  Val: {len(val_entries)}")

        return self._save_manifests(train_entries, val_entries)

    def _process_files_parallel(
        self,
        audio_files: List[Path],
        prefix: str,
        speaker_name: Optional[str] = None,
        language_name: Optional[str] = None,
    ) -> List[Optional[ManifestEntry]]:
        """并行处理文件列表。"""
        entries = [None] * len(audio_files)

        if self.num_workers > 1:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {}
                for i, audio_path in enumerate(audio_files):
                    future = executor.submit(
                        process_file_with_metadata,
                        audio_path=audio_path,
                        prefix=prefix,
                        speaker_name=speaker_name,
                        language_name=language_name,
                        compute_duration=self.compute_duration,
                    )
                    futures[future] = i

                for future in futures:
                    idx = futures[future]
                    try:
                        entries[idx] = future.result()
                    except Exception as e:
                        print(f"Error processing {audio_files[idx]}: {e}")
        else:
            # 单线程处理
            for i, audio_path in enumerate(audio_files):
                entries[i] = process_file_with_metadata(
                    audio_path=audio_path,
                    prefix=prefix,
                    speaker_name=speaker_name,
                    language_name=language_name,
                    compute_duration=self.compute_duration,
                )

        return entries

    def _save_manifests(
        self,
        train_entries: List[ManifestEntry],
        val_entries: List[ManifestEntry],
    ) -> tuple:
        """保存 manifest 文件。"""
        train_path = self.output_dir / "train_manifest.json"
        val_path = self.output_dir / "val_manifest.json"

        # 转换为字典列表
        train_data = [entry.to_dict() for entry in train_entries]
        val_data = [entry.to_dict() for entry in val_entries]

        # 保存
        print(f"\nSaving manifests...")
        print(f"  Train: {train_path} ({len(train_data)} entries)")
        print(f"  Val: {val_path} ({len(val_data)} entries)")

        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)

        print("Done!")
        return str(train_path), str(val_path)


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="通用音频数据集 manifest 生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 自动扫描子目录（适用于多说话人/多语言数据集）
  python scripts/prepare_manifest.py --dataset-dir /path/to/dataset

  # 指定子目录
  python scripts/prepare_manifest.py --dataset-dir /path/to/dataset --subdirs spk1 spk2 spk3

  # 子目录作为 speaker 字段
  python scripts/prepare_manifest.py --dataset-dir /path/to/dataset --subdir-as-speaker

  # 子目录作为 language 字段
  python scripts/prepare_manifest.py --dataset-dir /path/to/dataset --subdir-as-language

  # 扁平目录结构（无子目录分类）
  python scripts/prepare_manifest.py --dataset-dir /path/to/dataset --flat

  # 递归搜索子目录
  python scripts/prepare_manifest.py --dataset-dir /path/to/dataset --flat --recursive

  # 限制样本数量（测试用）
  python scripts/prepare_manifest.py --dataset-dir /path/to/dataset --max-samples 100

  # 指定验证集比例
  python scripts/prepare_manifest.py --dataset-dir /path/to/dataset --val-split 0.1

  # 多进程加速
  python scripts/prepare_manifest.py --dataset-dir /path/to/dataset --num-workers 8

  # 跳过时长计算（快速生成，duration 字段为 0）
  python scripts/prepare_manifest.py --dataset-dir /path/to/dataset --no-compute-duration
        """
    )

    # 必需参数
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="数据集根目录"
    )

    # 输出参数
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认为 dataset-dir/manifests）"
    )

    # 目录结构参数
    parser.add_argument(
        "--subdirs",
        type=str,
        nargs="+",
        default=None,
        help="要处理的子目录列表（默认自动扫描所有子目录）"
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="扁平模式：忽略子目录分类，将所有文件放在一个 manifest 中"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归搜索子目录（仅在 flat 模式下有效）"
    )

    # 元数据参数
    parser.add_argument(
        "--subdir-as-speaker",
        action="store_true",
        help="将子目录名作为 speaker 字段"
    )
    parser.add_argument(
        "--subdir-as-language",
        action="store_true",
        help="将子目录名作为 language 字段"
    )
    parser.add_argument(
        "--id-prefix",
        type=str,
        default="",
        help="manifest ID 的前缀"
    )

    # 处理参数
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.05,
        help="验证集比例（默认 0.05 = 5%%）"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="每个子目录的最大样本数（默认无限制）"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="并行处理的工作进程数（默认 4）"
    )
    parser.add_argument(
        "--no-compute-duration",
        action="store_true",
        help="跳过时长计算（快速生成，duration 字段为 0）"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=None,
        help="音频文件扩展名（默认：.wav .flac .mp3 .ogg .m4a）"
    )

    return parser.parse_args()


def main():
    """主函数。"""
    args = parse_args()

    # 设置输出目录
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(args.dataset_dir, "manifests")

    # 创建构建器
    builder = ManifestBuilder(
        dataset_dir=args.dataset_dir,
        output_dir=output_dir,
        val_split=args.val_split,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        compute_duration=not args.no_compute_duration,
        file_extensions=args.extensions,
        id_prefix=args.id_prefix,
    )

    # 构建 manifest
    if args.flat:
        # 扁平模式
        train_path, val_path = builder.build_flat(
            recursive=args.recursive,
            path_as_id=bool(args.id_prefix),
        )
    else:
        # 子目录模式
        train_path, val_path = builder.build_from_subdirectories(
            subdirs=args.subdirs,
            subdir_as_speaker=args.subdir_as_speaker,
            subdir_as_language=args.subdir_as_language,
        )

    print(f"\nManifest files created:")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")


if __name__ == "__main__":
    main()
