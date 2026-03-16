# coding=utf-8
"""
SpeechTokenizer: Neural Audio Codec with Residual Vector Quantization

SpeechTokenizer uses:
- SEANet encoder/decoder for audio encoding/decoding
- Residual Vector Quantization (RVQ) for discrete code generation
- Factorized semantic and acoustic feature representation
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Any, Optional, List, Tuple
from einops import rearrange

from transformers.utils import logging
from soniq.models.base.outputs import CodecOutput
from soniq.models.codec.base import BaseCodecModel
from soniq.models.codec.speechtokenizer.configuration_speechtokenizer import SpeechTokenizerConfig
from soniq.models.codec.speechtokenizer.speechtokenizer_components import (
    SEANetEncoder,
    SEANetDecoder,
    ResidualVectorQuantizer,
)


logger = logging.get_logger(__name__)


class SpeechTokenizer(BaseCodecModel):
    """
    SpeechTokenizer: Neural Audio Codec with Residual Vector Quantization.

    This model encodes audio into discrete codes using residual vector quantization
    and decodes them back to audio. It supports factorized semantic and acoustic
    feature representation.

    Example:
        ```python
        config = SpeechTokenizerConfig()
        model = SpeechTokenizer(config)

        # Training
        batch = {"audio": audio, "audio_lengths": audio_lengths}
        output = model(batch)

        # Inference
        codes = model.encode(audio)
        reconstructed = model.decode(codes)

        # Extract semantic features
        semantic_features = model.extract_semantic_features(audio)
        ```
    """

    config_class = SpeechTokenizerConfig
    base_model_prefix = "speechtokenizer"
    supports_gradient_checkpointing = True

    def __init__(self, config: SpeechTokenizerConfig):
        super().__init__(config)
        self.config = config

        # Encoder
        self.encoder = SEANetEncoder(
            n_filters=config.n_filters,
            dimension=config.dimension,
            ratios=config.strides,
            lstm_layers=config.lstm_layers,
            bidirectional=config.bidirectional,
            dilation_base=config.dilation_base,
            residual_kernel_size=config.residual_kernel_size,
            n_residual_layers=config.n_residual_layers,
            activation=config.activation,
        )

        # Dimension transformation if needed
        if config.dimension != config.semantic_dimension:
            self.transform = nn.Linear(config.dimension, config.semantic_dimension)
        else:
            self.transform = nn.Identity()

        # Residual Vector Quantizer
        self.quantizer = ResidualVectorQuantizer(
            dimension=config.dimension,
            n_q=config.n_q,
            bins=config.codebook_size,
        )

        # Decoder
        self.decoder = SEANetDecoder(
            n_filters=config.n_filters,
            dimension=config.dimension,
            ratios=config.strides,
            lstm_layers=0,  # Decoder typically doesn't need LSTM
            bidirectional=False,
            dilation_base=config.dilation_base,
            residual_kernel_size=config.residual_kernel_size,
            n_residual_layers=config.n_residual_layers,
            activation=config.activation,
        )

        # Initialize weights
        self.apply(self._init_weights)

    @property
    def sample_rate(self) -> int:
        """Get the audio sample rate."""
        return self.config.sample_rate

    @property
    def hop_length(self) -> int:
        """Get the hop length (stride) of the model."""
        return self.config.downsample_rate

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass for training.

        Args:
            data: Batch dictionary containing:
                - audio: Input audio waveform (batch, 1, seq_len)
                - audio_lengths: Audio lengths (batch,)

        Returns:
            Dictionary containing:
                - reconstructed: Reconstructed audio
                - codes: Quantization codes
                - commit_loss: Commitment loss from RVQ
                - feature: Semantic features from first quantizer
        """
        audio = data["audio"]
        audio_lengths = data.get("audio_lengths", None)

        # Encode
        encoded = self.encoder(audio)

        # Quantize
        quantized, codes, commit_loss, quantized_list = self.quantizer(encoded, n_q=self.config.n_q)

        # Transform to semantic dimension
        feature = rearrange(quantized_list[0], "b d t -> b t d")
        feature = self.transform(feature)

        # Decode
        reconstructed = self.decoder(quantized)

        # Create mask if lengths provided
        # Account for downsampling: reconstructed length is audio_length / hop_length
        if audio_lengths is not None:
            max_len = reconstructed.shape[2]
            # Downsample the audio lengths by the hop factor
            reconstructed_lengths = (audio_lengths / self.hop_length).ceil().long()
            mask = torch.arange(max_len, device=audio.device).unsqueeze(0) < reconstructed_lengths.unsqueeze(1)
            reconstructed = reconstructed * mask.unsqueeze(1).float()

        return {
            "reconstructed": reconstructed,
            "codes": codes,
            "commit_loss": commit_loss,
            "feature": feature,
            "quantized_list": quantized_list,
        }

    @torch.no_grad()
    def encode(
        self,
        audio: torch.Tensor,
        n_q: Optional[int] = None,
        st: int = 0,
    ) -> torch.Tensor:
        """
        Encode audio to discrete codes.

        Args:
            audio: Input audio waveform (batch, 1, seq_len).
            n_q: Number of quantizers to use (default: all).
            st: Start quantizer index.

        Returns:
            codes: Quantization codes (n_q, batch, timesteps).
        """
        encoded = self.encoder(audio)
        codes = self.quantizer.encode(encoded, n_q=n_q, st=st)
        return codes

    @torch.no_grad()
    def decode(
        self,
        codes: torch.Tensor,
        st: int = 0,
    ) -> torch.Tensor:
        """
        Decode codes to audio.

        Args:
            codes: Quantization codes (n_q, batch, timesteps).
            st: Start quantizer index.

        Returns:
            reconstructed: Reconstructed audio (batch, 1, seq_len).
        """
        quantized = self.quantizer.decode(codes, st=st)
        reconstructed = self.decoder(quantized)
        return reconstructed

    @torch.no_grad()
    def extract_semantic_features(
        self,
        audio: torch.Tensor,
        layer: int = 0,
    ) -> torch.Tensor:
        """
        Extract semantic features from audio.

        Args:
            audio: Input audio waveform (batch, 1, seq_len).
            layer: Quantizer layer to extract features from.

        Returns:
            features: Semantic features (batch, timesteps, semantic_dim).
        """
        encoded = self.encoder(audio)
        _, _, _, quantized_list = self.quantizer(encoded, n_q=layer + 1)
        feature = rearrange(quantized_list[layer], "b d t -> b t d")
        feature = self.transform(feature)
        return feature

    def synthesize(
        self,
        codes: torch.Tensor,
        **kwargs,
    ) -> CodecOutput:
        """
        Synthesize audio from codes.

        Args:
            codes: Quantization codes (n_q, batch, timesteps).
            **kwargs: Additional arguments.

        Returns:
            CodecOutput with reconstructed audio.
        """
        reconstructed = self.decode(codes)
        return CodecOutput(reconstructed=reconstructed, codes=codes)

    def reconstruct(
        self,
        audio: torch.Tensor,
        n_q: Optional[int] = None,
        **kwargs,
    ) -> CodecOutput:
        """
        Reconstruct audio from input audio.

        Args:
            audio: Input audio waveform (batch, 1, seq_len).
            n_q: Number of quantizers to use.
            **kwargs: Additional arguments.

        Returns:
            CodecOutput with reconstructed audio and codes.
        """
        codes = self.encode(audio, n_q=n_q)
        reconstructed = self.decode(codes)
        return CodecOutput(reconstructed=reconstructed, codes=codes)
