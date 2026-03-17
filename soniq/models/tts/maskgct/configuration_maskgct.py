# coding=utf-8
"""MaskGCT configuration."""

from dataclasses import dataclass, field
from typing import List, Optional
from transformers import PretrainedConfig


class MaskGCTConfig(PretrainedConfig):
    """
    Configuration for MaskGCT model.

    MaskGCT is a fully non-autoregressive TTS model with two stages:
    - T2S: Text to Semantic tokens
    - S2A: Semantic to Acoustic tokens
    """

    model_type = "maskgct"

    def __init__(
        self,
        # T2S model config
        t2s_hidden_size: int = 1024,
        t2s_num_layers: int = 16,
        t2s_num_heads: int = 16,
        t2s_cfg_scale: float = 0.2,
        # S2A model config
        s2a_hidden_size: int = 1024,
        s2a_num_layers: int = 16,
        s2a_num_heads: int = 16,
        s2a_num_quantizers: int = 12,
        s2a_cfg_scale: float = 0.15,
        # Codebook config
        semantic_codebook_size: int = 8192,
        acoustic_codebook_size: int = 1024,
        # Phone config
        phone_vocab_size: int = 1024,
        use_phone_cond: bool = True,
        # Sampling
        n_timesteps: int = 40,
        temperature: float = 0.9,
        filter_threshold: float = 0.98,
        cfg_weight: float = 1.0,
        # Audio
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # T2S
        self.t2s_hidden_size = t2s_hidden_size
        self.t2s_num_layers = t2s_num_layers
        self.t2s_num_heads = t2s_num_heads
        self.t2s_cfg_scale = t2s_cfg_scale

        # S2A
        self.s2a_hidden_size = s2a_hidden_size
        self.s2a_num_layers = s2a_num_layers
        self.s2a_num_heads = s2a_num_heads
        self.s2a_num_quantizers = s2a_num_quantizers
        self.s2a_cfg_scale = s2a_cfg_scale

        # Codebook
        self.semantic_codebook_size = semantic_codebook_size
        self.acoustic_codebook_size = acoustic_codebook_size

        # Phone
        self.phone_vocab_size = phone_vocab_size
        self.use_phone_cond = use_phone_cond

        # Sampling
        self.n_timesteps = n_timesteps
        self.temperature = temperature
        self.filter_threshold = filter_threshold
        self.cfg_weight = cfg_weight

        self.sample_rate = sample_rate


class MaskGCT_T2S_Config(PretrainedConfig):
    """Configuration for MaskGCT T2S model."""

    model_type = "maskgct_t2s"

    def __init__(
        self,
        hidden_size: int = 1024,
        num_layers: int = 16,
        num_heads: int = 16,
        cfg_scale: float = 0.2,
        cond_codebook_size: int = 8192,
        cond_dim: int = 1024,
        use_phone_cond: bool = True,
        phone_vocab_size: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cfg_scale = cfg_scale
        self.cond_codebook_size = cond_codebook_size
        self.cond_dim = cond_dim
        self.use_phone_cond = use_phone_cond
        self.phone_vocab_size = phone_vocab_size


class MaskGCT_S2A_Config(PretrainedConfig):
    """Configuration for MaskGCT S2A model."""

    model_type = "maskgct_s2a"

    def __init__(
        self,
        num_quantizers: int = 12,
        hidden_size: int = 1024,
        num_layers: int = 16,
        num_heads: int = 16,
        codebook_size: int = 1024,
        cfg_scale: float = 0.15,
        mask_layer_schedule: str = "linear",
        cond_codebook_size: int = 8192,
        predict_layer_1: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_quantizers = num_quantizers
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.codebook_size = codebook_size
        self.cfg_scale = cfg_scale
        self.mask_layer_schedule = mask_layer_schedule
        self.cond_codebook_size = cond_codebook_size
        self.predict_layer_1 = predict_layer_1