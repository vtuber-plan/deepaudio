# coding=utf-8
"""Vocos vocoder configuration."""

from transformers import PretrainedConfig


class VocosConfig(PretrainedConfig):
    """
    Configuration class for the Vocos neural vocoder.

    Vocos is a efficient neural vocoder that uses Fourier-based
    synthesis with ISTFT for high-quality audio generation.

    Args:
        sample_rate: Audio sample rate in Hz.
        hop_length: Hop length for audio synthesis.
        n_mel: Number of mel filterbanks (input features).
        n_fft: FFT size for ISTFT.
        dim: Model dimension.
        intermediate_dim: Intermediate dimension in feedforward layers.
        num_layers: Number of transformer layers.
        n_codebooks: Number of codebooks for RVQ input.
        codebook_size: Size of each codebook.

    Example:
        ```python
        config = VocosConfig(
            sample_rate=24000,
            hop_length=256,
            n_mel=100,
            dim=512,
            num_layers=4,
        )
        ```
    """

    model_type = "vocos"

    def __init__(
        self,
        sample_rate: int = 24000,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mel: int = 100,
        n_fft: int = 1024,
        dim: int = 512,
        intermediate_dim: int = 1536,
        num_layers: int = 4,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel = n_mel
        self.n_fft = n_fft
        self.dim = dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        super().__init__(**kwargs)
