# coding=utf-8
"""
VALL-E configuration.

VALL-E is a zero-shot TTS model that uses:
- Autoregressive (AR) Transformer decoder for first quantizer
- Non-Autoregressive (NAR) Transformer decoder for remaining quantizers
- Neural codec audio tokens as representation
"""

from typing import Optional
from transformers.utils import logging
from soniq.models.base.configuration_base import SoniqModelConfig


logger = logging.get_logger(__name__)


class VALLEConfig(SoniqModelConfig):
    """
    Configuration class for VALL-E.

    VALL-E uses an AR decoder to generate the first audio quantizer layer,
    then uses NAR decoders to generate the remaining quantizer layers in parallel.

    Args:
        decoder_dim: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_decoder_layers: Number of AR decoder layers.
        nar_scale_factor: Scaling factor for NAR decoder dimensions.
        num_quantizers: Number of audio quantization layers.
        text_token_num: Vocabulary size for text tokens.
        audio_token_num: Number of audio codebook entries.
        prepend_bos: Whether to prepend BOS token to audio input.
        add_prenet: Whether to add prenet layers before transformer.
        norm_first: Use pre-norm vs post-norm in Transformer.
        prefix_mode: Prompt prefix strategy (0: none, 1: beginning, 2: random).
        share_embedding: Share weights between predictor and embedding.
        dropout: Dropout rate.
        max_seq_len: Maximum sequence length.
    """

    model_type = "valle"

    def __init__(
        self,
        decoder_dim: int = 1024,
        nhead: int = 16,
        num_decoder_layers: int = 12,
        nar_scale_factor: float = 1.0,
        num_quantizers: int = 8,
        text_token_num: int = 512,
        audio_token_num: int = 1024,
        prepend_bos: bool = False,
        add_prenet: bool = False,
        norm_first: bool = True,
        prefix_mode: int = 0,
        share_embedding: bool = True,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        initializer_range: float = 0.02,
        **kwargs
    ):
        self.decoder_dim = decoder_dim
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers
        self.nar_scale_factor = nar_scale_factor
        self.num_quantizers = num_quantizers
        self.text_token_num = text_token_num
        self.audio_token_num = audio_token_num
        self.prepend_bos = prepend_bos
        self.add_prenet = add_prenet
        self.norm_first = norm_first
        self.prefix_mode = prefix_mode
        self.share_embedding = share_embedding
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        # Validate configuration
        if num_quantizers < 1:
            raise ValueError("num_quantizers must be at least 1")
        if num_decoder_layers < 1:
            raise ValueError("num_decoder_layers must be at least 1")

        super().__init__(initializer_range=initializer_range, **kwargs)
