# coding=utf-8
"""Output dataclasses for Soniq models."""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch


@dataclass
class ModelOutput:
    """Base class for model outputs."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None


@dataclass
class VocoderOutput(ModelOutput):
    """Output for vocoder models."""
    waveform: Optional[torch.Tensor] = None
    mel_reconstruction: Optional[torch.Tensor] = None


@dataclass
class TTSOutput(ModelOutput):
    """Output for TTS models."""
    waveform: Optional[torch.Tensor] = None
    mel_spectrogram: Optional[torch.Tensor] = None
    durations: Optional[torch.Tensor] = None
    alignments: Optional[torch.Tensor] = None


@dataclass
class CodecOutput(ModelOutput):
    """Output for codec models."""
    reconstructed: Optional[torch.Tensor] = None
    codes: Optional[torch.Tensor] = None
    quantized: Optional[torch.Tensor] = None


@dataclass
class SVCOutput(ModelOutput):
    """Output for SVC models."""
    waveform: Optional[torch.Tensor] = None
    content_features: Optional[torch.Tensor] = None
    speaker_embedding: Optional[torch.Tensor] = None


@dataclass
class VCOutput(ModelOutput):
    """Output for voice conversion models."""
    waveform: Optional[torch.Tensor] = None
    converted_features: Optional[torch.Tensor] = None
    source_content: Optional[torch.Tensor] = None
    target_speaker: Optional[torch.Tensor] = None


@dataclass
class ASROutput(ModelOutput):
    """Output for ASR models."""
    transcription: Optional[str] = None
    logits: Optional[torch.Tensor] = None
    token_ids: Optional[torch.Tensor] = None
    alignments: Optional[torch.Tensor] = None
