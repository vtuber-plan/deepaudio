# coding=utf-8
"""VitsSVC model configuration."""

from transformers import PretrainedConfig


class VitsSVCConfig(PretrainedConfig):
    """
    Configuration class for the VitsSVC singing voice conversion model.

    VitsSVC uses VAE + Flow architecture similar to VITS but with
    content features instead of text input.

    Args:
        sample_rate: Audio sample rate in Hz.
        hop_length: Hop length for audio processing.
        n_mel: Number of mel filterbanks.
        n_fft: FFT size for spectrogram computation.
        hidden_dim: Hidden dimension size.
        inter_channels: Intermediate channels in flow layers.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        kernel_size: Convolution kernel size.
        dropout: Dropout probability.
        n_flow_layers: Number of flow layers.
        n_z_layers: Number of latent variable layers.
        n_prior_layers: Number of prior encoder layers.
        gin_channels: Global conditioning (speaker) channels.

    Example:
        ```python
        config = VitsSVCConfig(
            sample_rate=44100,
            hop_length=512,
            n_mel=128,
            hidden_dim=192,
            inter_channels=192,
        )
        ```
    """

    model_type = "vitsvc"

    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        n_mel: int = 128,
        n_fft: int = 2048,
        hidden_dim: int = 192,
        inter_channels: int = 192,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
        n_flow_layers: int = 8,
        n_z_layers: int = 3,
        n_prior_layers: int = 3,
        gin_channels: int = 256,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mel = n_mel
        self.n_fft = n_fft
        self.hidden_dim = hidden_dim
        self.inter_channels = inter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_flow_layers = n_flow_layers
        self.n_z_layers = n_z_layers
        self.n_prior_layers = n_prior_layers
        self.gin_channels = gin_channels
        super().__init__(**kwargs)
