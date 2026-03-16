# coding=utf-8
"""Manifest utilities."""

import json
import os
from typing import Any, Dict, List


def load_manifest(path: str) -> List[Dict[str, Any]]:
    """Load manifest from JSON or JSONL file."""
    with open(path, 'r', encoding='utf-8') as f:
        if path.endswith('.jsonl'):
            return [json.loads(line) for line in f]
        return json.load(f)


def save_manifest(data: List[Dict[str, Any]], path: str) -> None:
    """Save manifest to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        if path.endswith('.jsonl'):
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)
