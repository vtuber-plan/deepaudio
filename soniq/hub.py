# coding=utf-8
"""
Hugging Face Hub integration for Soniq.

This module provides utilities for loading models from Hugging Face Hub.
For uploading models, use the `huggingface-cli` command directly.
"""

import os
from pathlib import Path
from typing import Optional
import logging

from huggingface_hub import snapshot_download, hf_hub_download


logger = logging.getLogger(__name__)


def load_from_hub(
    repo_id: str,
    local_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: str = "main",
    token: Optional[str] = None,
    allow_patterns: Optional[list] = None,
) -> str:
    """
    Download a model from Hugging Face Hub and return local path.

    Args:
        repo_id: Repository ID (e.g., "username/model-name").
        local_dir: Local directory to save the model. If None, uses cache.
        cache_dir: Cache directory. Only used if local_dir is None.
        revision: Branch or revision.
        token: Hugging Face API token for private models.
        allow_patterns: Patterns of files to download.

    Returns:
        Path to the downloaded model directory.

    Example:
        ```python
        from soniq.hub import load_from_hub

        # Download to cache
        model_path = load_from_hub("soniq/hifigan-base")

        # Download to specific directory
        model_path = load_from_hub(
            "soniq/hifigan-base",
            local_dir="./models/hifigan",
        )
        ```
    """
    if allow_patterns is None:
        allow_patterns = [
            "*.bin",
            "*.safetensors",
            "*.json",
            "*.txt",
            "*.md",
            "*.yaml",
            "*.yml",
            "*.pt",
            "*.pth",
        ]

    model_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    )

    logger.info(f"Model downloaded to: {model_path}")
    return model_path


def download_file_from_hub(
    repo_id: str,
    filename: str,
    local_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: str = "main",
    token: Optional[str] = None,
) -> str:
    """
    Download a specific file from Hugging Face Hub.

    Args:
        repo_id: Repository ID.
        filename: Filename to download.
        local_dir: Local directory to save the file.
        cache_dir: Cache directory.
        revision: Branch or revision.
        token: Hugging Face API token.

    Returns:
        Path to the downloaded file.
    """
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_dir=local_dir,
    )

    logger.info(f"File downloaded to: {file_path}")
    return file_path
