# coding=utf-8
"""Vocos vocoder - Modern efficient vocoder based on ConvNeXt + ISTFT."""

from typing import Optional, Dict, Any
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel

from soniq.models.vocoders.vocos.configuration_vocos import VocosConfig


class ISTFT(nn.Module):
    """
    Inverse Short-Time Fourier Transform layer.

    Supports both "center" and "same" padding modes.

    Args:
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length (default: n_fft)
        padding: Padding type ("center" or "same")
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: Optional[int] = None,
        padding: str = "same",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.padding = padding

        # Register window as buffer
        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window)

        # For "same" padding, compute padding amounts
        if padding == "same":
            # Compute left and right padding for overlap-add
            self.pad_left = self.win_length // 2
            self.pad_right = self.win_length - self.pad_left - 1
        else:
            self.pad_left = 0
            self.pad_right = 0

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Inverse STFT.

        Uses PyTorch's istft function for reliable implementation.

        Args:
            spec: Complex spectrogram (batch, n_fft, time)

        Returns:
            Audio waveform (batch, 1, time)
        """
        # Handle different input formats
        if spec.dim() == 4:
            # Complex tensor in real/imag format: (B, F, T, 2)
            spec = torch.view_as_complex(spec.contiguous())

        # spec is now (B, n_fft, T) complex
        # PyTorch istft expects (B, F, T) -> (B, T)
        batch_size = spec.shape[0]

        # Use PyTorch's istft
        audio = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=False,
        )

        return audio.unsqueeze(1)  # (B, 1, T)


class ISTFTHead(nn.Module):
    """
    ISTFT head for Vocos vocoder.

    Converts hidden features to complex spectrogram and applies ISTFT.

    Args:
        dim: Input feature dimension
        n_fft: FFT size
        hop_length: Hop length
        padding: Padding type
    """

    def __init__(
        self,
        dim: int,
        n_fft: int,
        hop_length: int,
        padding: str = "same",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Output: n_fft for magnitude + 2 for phase (cos/sin representation)
        out_dim = n_fft + 2
        self.out = nn.Linear(dim, out_dim)

        # ISTFT layer
        self.istft = ISTFT(n_fft, hop_length, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Hidden features (batch, time, dim)

        Returns:
            Audio waveform (batch, 1, time)
        """
        # Project to spectrogram parameters
        x = self.out(x)  # (B, T, n_fft + 2)
        x = x.transpose(1, 2)  # (B, n_fft + 2, T)

        # Split into magnitude and phase
        mag, p = x.split([self.n_fft, 2], dim=1)

        # Clamp magnitude (from log-scale)
        mag = torch.exp(mag.clamp(max=1e2))

        # Convert phase to complex representation
        # Using cos/sin parameterization for better training stability
        cos_p = torch.cos(p[:, :1, :])
        sin_p = torch.sin(p[:, 1:, :])

        # Create complex spectrogram
        S = mag * (cos_p + 1j * sin_p)

        # ISTFT
        audio = self.istft(S)

        return audio


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block for Vocos backbone.

    Args:
        dim: Hidden dimension
        intermediate_dim: Intermediate dimension in inverted bottleneck
        kernel_size: Depthwise convolution kernel size
        layer_scale_init_value: Initial value for layer scale gamma
        adanorm_num_embeddings: If > 0, use AdaLayerNorm with this many embeddings
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel_size: int = 7,
        layer_scale_init_value: float = 1e-3,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.intermediate_dim = intermediate_dim

        # Depthwise convolution
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )

        # Normalization (applied after dwconv, before pwconv)
        if adanorm_num_embeddings is not None and adanorm_num_embeddings > 0:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim)
        else:
            self.norm = nn.LayerNorm(dim)

        # Inverted bottleneck
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

        # Layer scale
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim),
            requires_grad=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        cond_embedding_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input (batch, dim, time)
            cond_embedding_id: Optional condition embedding ID for AdaLayerNorm

        Returns:
            Output (batch, dim, time)
        """
        residual = x

        # Depthwise conv
        x = self.dwconv(x)

        # Transpose for layer norm: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)

        # Normalize
        if isinstance(self.norm, AdaLayerNorm):
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)

        # Inverted bottleneck
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Layer scale
        x = self.gamma * x

        # Transpose back
        x = x.transpose(1, 2)

        # Residual connection
        return x + residual


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization conditioned on embedding ID.

    Args:
        num_embeddings: Number of embeddings
        embedding_dim: Embedding dimension (should match layer dim)
        eps: Epsilon for numerical stability
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.eps = eps

        # Scale and shift embeddings
        self.scale = nn.Embedding(num_embeddings, embedding_dim)
        self.shift = nn.Embedding(num_embeddings, embedding_dim)

        # Initialize to identity transform
        nn.init.ones_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)

    def forward(
        self,
        x: torch.Tensor,
        cond_embedding_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input (batch, time, dim)
            cond_embedding_id: Condition embedding IDs (batch,) or (batch, time)

        Returns:
            Normalized output (batch, time, dim)
        """
        # Get scale and shift
        if cond_embedding_id.dim() == 1:
            # Same embedding for all time steps
            scale = self.scale(cond_embedding_id).unsqueeze(1)
            shift = self.shift(cond_embedding_id).unsqueeze(1)
        else:
            # Different embeddings per time step
            scale = self.scale(cond_embedding_id)
            shift = self.shift(cond_embedding_id)

        # Apply layer norm
        x = F.layer_norm(x, (self.embedding_dim,), eps=self.eps)

        # Apply adaptive transform
        return x * scale + shift


class VocosBackbone(nn.Module):
    """
    Vocos backbone based on ConvNeXt architecture.

    Args:
        input_channels: Input feature channels (e.g., 128 for mel)
        dim: Hidden dimension
        intermediate_dim: Intermediate dimension in ConvNeXt blocks
        num_layers: Number of ConvNeXt blocks
        kernel_size: ConvNeXt kernel size
        layer_scale_init_value: Layer scale initialization
        adanorm_num_embeddings: If > 0, use AdaLayerNorm
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        kernel_size: int = 7,
        layer_scale_init_value: float = 1e-3,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.dim = dim
        self.num_layers = num_layers

        # Initial embedding
        self.embed = nn.Conv1d(
            input_channels,
            dim,
            kernel_size=7,
            padding=3,
        )

        # Normalization
        if adanorm_num_embeddings is not None and adanorm_num_embeddings > 0:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim)
        else:
            self.norm = nn.LayerNorm(dim)

        # ConvNeXt blocks
        self.convnext = nn.ModuleList([
            ConvNeXtBlock(
                dim=dim,
                intermediate_dim=intermediate_dim,
                kernel_size=kernel_size,
                layer_scale_init_value=layer_scale_init_value,
                adanorm_num_embeddings=adanorm_num_embeddings,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        cond_embedding_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features (batch, input_channels, time)
            cond_embedding_id: Optional condition embedding ID

        Returns:
            Hidden features (batch, time, dim)
        """
        # Initial projection
        x = self.embed(x)  # (B, dim, T)

        # Transpose for norm: (B, dim, T) -> (B, T, dim)
        x = x.transpose(1, 2)

        # Initial norm
        if isinstance(self.norm, AdaLayerNorm):
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)

        # Transpose back for ConvNeXt blocks: (B, T, dim) -> (B, dim, T)
        x = x.transpose(1, 2)

        # ConvNeXt blocks (input/output: B, dim, T)
        for block in self.convnext:
            x = block(x, cond_embedding_id)

        # Transpose for final norm: (B, dim, T) -> (B, T, dim)
        x = x.transpose(1, 2)

        # Final norm
        x = self.final_layer_norm(x)

        return x


class Vocos(PreTrainedModel):
    """
    Vocos vocoder.

    A modern efficient vocoder based on ConvNeXt backbone and ISTFT head.

    Example:
        ```python
        from soniq.models.vocoders.vocos import Vocos, VocosConfig

        config = VocosConfig(
            sample_rate=24000,
            hop_length=256,
            n_mel=100,
            dim=512,
            num_layers=4,
        )
        model = Vocos(config)

        mel = torch.randn(4, 100, 50)  # (batch, n_mel, frames)
        audio = model(mel)  # (batch, 1, samples)
        ```
    """

    config_class = VocosConfig
    base_model_prefix = "vocos"
    supports_gradient_checkpointing = True

    def __init__(self, config: VocosConfig):
        super().__init__(config)
        self.config = config

        # Backbone
        self.backbone = VocosBackbone(
            input_channels=config.n_mel,
            dim=config.dim,
            intermediate_dim=config.intermediate_dim,
            num_layers=config.num_layers,
            adanorm_num_embeddings=None,  # Can be extended for conditional vocoding
        )

        # Head
        self.head = ISTFTHead(
            dim=config.dim,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            padding="same",
        )

        self.post_init()

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def sample_rate(self) -> int:
        """Get the audio sample rate."""
        return self.config.sample_rate

    @property
    def hop_length(self) -> int:
        """Get the hop length."""
        return self.config.hop_length

    @property
    def n_fft(self) -> int:
        """Get the FFT size."""
        return self.config.n_fft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate audio from mel spectrogram.

        Args:
            x: Mel spectrogram (batch, n_mel, time)

        Returns:
            Audio waveform (batch, 1, time)
        """
        # Backbone
        h = self.backbone(x)  # (B, T, dim)

        # Head
        audio = self.head(h)  # (B, 1, T)

        return audio

    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference mode (same as forward, but with eval()).

        Args:
            x: Mel spectrogram (batch, n_mel, time)

        Returns:
            Audio waveform (batch, 1, time)
        """
        self.eval()
        return self.forward(x)

    @torch.no_grad()
    def generate(self, mel: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Generate audio from mel spectrogram.

        Args:
            mel: Mel spectrogram (batch, n_mel, time)
            **kwargs: Additional arguments

        Returns:
            Dictionary with 'waveform' key containing audio
        """
        self.eval()
        waveform = self.forward(mel)
        return {"waveform": waveform}


__all__ = ["Vocos", "VocosBackbone", "ISTFTHead", "ISTFT", "ConvNeXtBlock", "AdaLayerNorm"]
