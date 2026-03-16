# coding=utf-8
"""
Hugging Face Hub utilities for Soniq.

Simple utilities for downloading models from Hugging Face Hub.
For uploading models, use the `huggingface-cli` command directly.
"""

import os
import logging
from typing import List, Optional


logger = logging.getLogger(__name__)


def get_cache_dir() -> str:
    """
    Get the Hugging Face cache directory.

    Returns:
        Path to the cache directory.
    """
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def download_file_from_hub(
    repo_id: str,
    filename: str,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    revision: str = "main",
    token: Optional[str] = None,
) -> str:
    """
    Download a single file from Hugging Face Hub.

    Args:
        repo_id: Repository ID.
        filename: Filename to download.
        cache_dir: Cache directory.
        local_dir: Local directory to save the file.
        revision: Branch or revision.
        token: Hugging Face API token.

    Returns:
        Path to the downloaded file.
    """
    from huggingface_hub import hf_hub_download

    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        local_dir=local_dir,
        revision=revision,
        token=token,
    )
    logger.info(f"File downloaded: {file_path}")
    return file_path
