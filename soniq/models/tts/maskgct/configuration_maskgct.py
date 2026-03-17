# coding=utf-8
"""MaskGCT model configuration."""

from transformers import PretrainedConfig
from typing import Optional


class MaskGCTT2SConfig(PretrainedConfig):
    """
    Configuration class for MaskGCT Text-to-Semantic (T2S) model.

    Args:
        hidden_size: Hidden dimension size (default: 1536)
        num_hidden_layers: Number of transformer layers (default: 16)
        num_attention_heads: Number of attention heads (default: 16)
        intermediate_size: FFN intermediate size (default: 6144)
        vocab_size: Phone vocabulary size (default: 1024)
        codebook_size: Semantic codebook size (default: 8192)
        cond_dim: Conditioning dimension for semantic tokens (default: 1024)
        max_seq_len: Maximum sequence length (default: 2048)
        dropout: Dropout probability (default: 0.1)
        mask_prob_min: Minimum mask probability (default: 0.6)
        mask_prob_max: Maximum mask probability (default: 1.0)
        prompt_ratio: Ratio of prompt tokens (default: 0.4)

    Example:
        ```python
        config = MaskGCTT2SConfig(
            hidden_size=1536,
            num_hidden_layers=16,
            num_attention_heads=16,
            codebook_size=8192,
        )
        ```
    """

    model_type = "maskgct_t2s"

    def __init__(
        self,
        hidden_size: int = 1536,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 16,
        intermediate_size: int = 6144,
        vocab_size: int = 1024,
        codebook_size: int = 8192,
        cond_dim: int = 1024,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        mask_prob_min: float = 0.6,
        mask_prob_max: float = 1.0,
        prompt_ratio: float = 0.4,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.codebook_size = codebook_size
        self.cond_dim = cond_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.mask_prob_min = mask_prob_min
        self.mask_prob_max = mask_prob_max
        self.prompt_ratio = prompt_ratio
        super().__init__(**kwargs)


class MaskGCTS2AConfig(PretrainedConfig):
    """
    Configuration class for MaskGCT Semantic-to-Acoustic (S2A) model.

    Args:
        hidden_size: Hidden dimension size (default: 1024)
        num_hidden_layers: Number of transformer layers (default: 16)
        num_attention_heads: Number of attention heads (default: 16)
        intermediate_size: FFN intermediate size (default: 4096)
        num_quantizers: Number of RVQ quantizers (default: 12)
        codebook_size: Codebook size per quantizer (default: 1024)
        semantic_codebook_size: Semantic token codebook size for conditioning (default: 8192)
        max_seq_len: Maximum sequence length (default: 2048)
        dropout: Dropout probability (default: 0.1)
        mask_layer_schedule: How to schedule mask layers ("linear", "cosine")

    Example:
        ```python
        config = MaskGCTS2AConfig(
            hidden_size=1024,
            num_hidden_layers=16,
            num_quantizers=12,
            codebook_size=1024,
        )
        ```
    """

    model_type = "maskgct_s2a"

    def __init__(
        self,
        hidden_size: int = 1024,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        num_quantizers: int = 12,
        codebook_size: int = 1024,
        semantic_codebook_size: int = 8192,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        mask_layer_schedule: str = "linear",
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.semantic_codebook_size = semantic_codebook_size
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.mask_layer_schedule = mask_layer_schedule
        super().__init__(**kwargs)


class MaskGCTConfig(PretrainedConfig):
    """
    Configuration class for the full MaskGCT model.

    Args:
        t2s_config: Configuration for Text-to-Semantic model
        s2a_config: Configuration for Semantic-to-Acoustic model
        sample_rate: Audio sample rate in Hz (default: 24000)
        hop_length: Hop length for audio synthesis (default: 480)

    Example:
        ```python
        t2s_config = MaskGCTT2SConfig()
        s2a_config = MaskGCTS2AConfig()
        config = MaskGCTConfig(t2s_config=t2s_config, s2a_config=s2a_config)
        ```
    """

    model_type = "maskgct"

    def __init__(
        self,
        t2s_config: Optional[MaskGCTT2SConfig] = None,
        s2a_config: Optional[MaskGCTS2AConfig] = None,
        sample_rate: int = 24000,
        hop_length: int = 480,
        **kwargs,
    ):
        self.t2s_config = t2s_config or MaskGCTT2SConfig()
        self.s2a_config = s2a_config or MaskGCTS2AConfig()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        super().__init__(**kwargs)
