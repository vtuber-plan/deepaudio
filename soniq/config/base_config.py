# coding=utf-8
"""
Base configuration class for Soniq.
"""

import json
import os
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict
import yaml


@dataclass
class BaseConfig:
    """
    Base configuration class for Soniq.

    This class provides common configuration loading and saving functionality.

    Example:
        ```python
        config = BaseConfig.load("config.json")
        config.save("config_out.json")
        ```
    """

    # Model configuration
    model: Dict[str, Any] = field(default_factory=dict)

    # Training configuration
    training: Dict[str, Any] = field(default_factory=dict)

    # Data configuration
    data: Dict[str, Any] = field(default_factory=dict)

    # Logging configuration
    logging: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseConfig":
        """
        Create config from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            BaseConfig instance.
        """
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> "BaseConfig":
        """
        Load config from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            BaseConfig instance.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, path: str) -> "BaseConfig":
        """
        Load config from YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            BaseConfig instance.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.

        Returns:
            Configuration dictionary.
        """
        return asdict(self)

    def save(self, path: str, format: str = "json"):
        """
        Save config to file.

        Args:
            path: Path to save file.
            format: File format ("json" or "yaml").
        """
        data = self.to_dict()

        os.makedirs(os.path.dirname(path), exist_ok=True)

        if format == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format == "yaml":
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False)

    def update(self, **kwargs):
        """
        Update configuration with keyword arguments.

        Args:
            **kwargs: Configuration updates.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif "." in key:
                # Handle nested updates
                keys = key.split(".")
                obj = self
                for k in keys[:-1]:
                    obj = getattr(obj, k, {})
                if hasattr(obj, keys[-1]):
                    setattr(obj, keys[-1], value)


def load_config(path: str) -> BaseConfig:
    """
    Load configuration from file.

    Args:
        path: Path to configuration file.

    Returns:
        BaseConfig instance.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in [".json"]:
        return BaseConfig.from_json(path)
    elif ext in [".yaml", ".yml"]:
        return BaseConfig.from_yaml(path)
    else:
        raise ValueError(f"Unsupported config file extension: {ext}")


def save_config(config: BaseConfig, path: str, format: str = "json"):
    """
    Save configuration to file.

    Args:
        config: Configuration to save.
        path: Path to save file.
        format: File format ("json" or "yaml").
    """
    config.save(path, format=format)
