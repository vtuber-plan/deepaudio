"""
Soniq - State-of-the-art Audio Machine Learning Library

Soniq is a PyTorch-based audio machine learning library that provides
state-of-the-art speech and audio models with a Transformers-style API.
"""

__version__ = "0.1.0"
__author__ = "Soniq Team"

from .utils import audio_utils, model_utils, data_utils, hub_utils
from .pipelines import AudioPipeline, MelPipeline
from .hub import load_from_hub, download_file_from_hub
