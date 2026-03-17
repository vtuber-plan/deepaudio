# coding=utf-8
"""
Speaker Encoder modules for Soniq.

Provides speaker embedding extraction for:
- Speaker verification (ECAPA-TDNN)
- Voice conversion (StyleEncoder, ReferenceEncoder)
- Multi-speaker TTS (SpeakerIDEncoder)

Example:
    ```python
    from soniq.models.speaker_encoder import ECAPA_TDNN, StyleEncoder

    # ECAPA-TDNN for speaker verification
    encoder = ECAPA_TDNN(input_channels=80, emb_size=192)
    embedding = encoder(mel_spectrogram)

    # Style encoder for VC
    style_encoder = StyleEncoder(in_dim=80, out_dim=256)
    style_emb = style_encoder(mel_spectrogram)
    ```
"""

from .modeling_speaker_encoder import (
    # Building blocks
    SEModule,
    Res2Conv1dBlock,
    Conv1dBlock,
    Mish,
    Conv1dGLU,
    AttentiveStatsPool,
    # Encoders
    ECAPA_TDNN,
    ECAPA_TDNN_Block,
    StyleEncoder,
    ReferenceEncoder,
    SpeakerIDEncoder,
)

__all__ = [
    # Blocks
    "SEModule",
    "Res2Conv1dBlock",
    "Conv1dBlock",
    "Mish",
    "Conv1dGLU",
    "AttentiveStatsPool",
    # Encoders
    "ECAPA_TDNN",
    "ECAPA_TDNN_Block",
    "StyleEncoder",
    "ReferenceEncoder",
    "SpeakerIDEncoder",
]