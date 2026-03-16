# coding=utf-8
"""SpeechTokenizer configuration."""

from soniq.models.base.configuration_base import SoniqModelConfig


class SpeechTokenizerConfig(SoniqModelConfig):
    """
    Configuration class for SpeechTokenizer.

    SpeechTokenizer is a neural audio codec that uses residual vector quantization
    to encode audio into discrete codes. It factorizes semantic and acoustic features.

    Args:
        n_filters: Base number of filters in the encoder/decoder.
        dimension: Latent dimension of the model.
        strides: Stride factors for downsampling.
        lstm_layers: Number of LSTM layers.
        bidirectional: Whether to use bidirectional LSTM.
        dilation_base: Base dilation rate for residual blocks.
        residual_kernel_size: Kernel size for residual convolutions.
        n_residual_layers: Number of residual layers.
        activation: Activation function to use.
        n_q: Number of quantizers in RVQ.
        codebook_size: Size of each codebook.
        semantic_dimension: Dimension for semantic features.
        sample_rate: Audio sample rate.
    """

    model_type = "speechtokenizer"

    def __init__(
        self,
        n_filters: int = 32,
        dimension: int = 512,
        strides: list = None,
        lstm_layers: int = 3,
        bidirectional: bool = False,
        dilation_base: int = 4,
        residual_kernel_size: int = 3,
        n_residual_layers: int = 3,
        activation: str = "ELU",
        n_q: int = 8,
        codebook_size: int = 1024,
        semantic_dimension: int = 512,
        sample_rate: int = 16000,
        initializer_range: float = 0.02,
        **kwargs
    ):
        if strides is None:
            strides = [8, 6, 5, 4]

        self.n_filters = n_filters
        self.dimension = dimension
        self.strides = strides
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.dilation_base = dilation_base
        self.residual_kernel_size = residual_kernel_size
        self.n_residual_layers = n_residual_layers
        self.activation = activation
        self.n_q = n_q
        self.codebook_size = codebook_size
        self.semantic_dimension = semantic_dimension
        self.sample_rate = sample_rate
        self.downsample_rate = 1
        for s in strides:
            self.downsample_rate *= s

        super().__init__(initializer_range=initializer_range, **kwargs)
