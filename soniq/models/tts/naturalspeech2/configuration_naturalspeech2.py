# coding=utf-8
"""
NaturalSpeech2 configuration.

NaturalSpeech2 is a TTS model that uses:
- Prior encoder with duration and pitch predictors
- Diffusion decoder for high-quality synthesis
- Query-based speaker embedding
"""

from typing import Optional
from transformers.utils import logging
from soniq.models.base.configuration_base import SoniqModelConfig


logger = logging.get_logger(__name__)


class NaturalSpeech2Config(SoniqModelConfig):
    """
    Configuration class for NaturalSpeech2.

    NaturalSpeech2 uses a prior encoder to generate conditions from text,
    and a diffusion model to synthesize latent representations.

    Args:
        latent_dim: Latent representation dimension (from codec).
        encoder_hidden: Transformer encoder hidden dimension.
        encoder_layers: Number of encoder layers.
        encoder_heads: Number of attention heads.
        encoder_dropout: Encoder dropout rate.
        duration_predictor_hidden: Duration predictor hidden dimension.
        duration_predictor_layers: Number of duration predictor layers.
        pitch_predictor_hidden: Pitch predictor hidden dimension.
        pitch_predictor_layers: Number of pitch predictor layers.
        pitch_min: Minimum pitch value (Hz).
        pitch_max: Maximum pitch value (Hz).
        pitch_bins_num: Number of pitch bins.
        diffusion_hidden: Diffusion model hidden dimension.
        diffusion_layers: Number of diffusion model layers.
        diffusion_type: "diffusion" or "flow".
        beta_min: Diffusion schedule minimum.
        beta_max: Diffusion schedule maximum.
        sigma: Noise scale.
        ode_solver: "euler" or "midpoint".
        query_token_num: Number of speaker query tokens.
        query_hidden: Query embedding dimension.
        vocab_size: Text vocabulary size.
        max_seq_len: Maximum sequence length.
    """

    model_type = "naturalspeech2"

    def __init__(
        self,
        latent_dim: int = 128,
        encoder_hidden: int = 512,
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        encoder_dropout: float = 0.1,
        duration_predictor_hidden: int = 256,
        duration_predictor_layers: int = 2,
        pitch_predictor_hidden: int = 256,
        pitch_predictor_layers: int = 2,
        pitch_min: float = 50.0,
        pitch_max: float = 1100.0,
        pitch_bins_num: int = 256,
        diffusion_hidden: int = 512,
        diffusion_layers: int = 20,
        diffusion_type: str = "diffusion",
        beta_min: float = 0.0001,
        beta_max: float = 0.02,
        sigma: float = 1.0,
        noise_factor: float = 1.0,
        ode_solver: str = "euler",
        query_token_num: int = 32,
        query_hidden: int = 512,
        vocab_size: int = 512,
        max_seq_len: int = 4096,
        cross_attn_per_layer: int = 2,
        dilation_cycle: int = 5,
        initializer_range: float = 0.02,
        **kwargs
    ):
        self.latent_dim = latent_dim
        self.encoder_hidden = encoder_hidden
        self.encoder_layers = encoder_layers
        self.encoder_heads = encoder_heads
        self.encoder_dropout = encoder_dropout
        self.duration_predictor_hidden = duration_predictor_hidden
        self.duration_predictor_layers = duration_predictor_layers
        self.pitch_predictor_hidden = pitch_predictor_hidden
        self.pitch_predictor_layers = pitch_predictor_layers
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.pitch_bins_num = pitch_bins_num
        self.diffusion_hidden = diffusion_hidden
        self.diffusion_layers = diffusion_layers
        self.diffusion_type = diffusion_type
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.sigma = sigma
        self.noise_factor = noise_factor
        self.ode_solver = ode_solver
        self.query_token_num = query_token_num
        self.query_hidden = query_hidden
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.cross_attn_per_layer = cross_attn_per_layer
        self.dilation_cycle = dilation_cycle

        super().__init__(initializer_range=initializer_range, **kwargs)
