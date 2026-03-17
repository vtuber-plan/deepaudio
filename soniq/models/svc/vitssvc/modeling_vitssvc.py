# coding=utf-8
"""
VitsSVC: Singing Voice Conversion based on VITS.

VitsSVC uses content features (ContentVec, Whisper, etc.) and F0 as input
instead of text, enabling singing voice conversion.
"""

import copy
import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Any, Optional, Tuple

from transformers import PreTrainedModel
from transformers.utils import logging
from soniq.models.svc.base import BaseSVCModel
from soniq.models.svc.vitssvc.configuration_vitssvc import VitsSVCConfig
from soniq.models.svc.vitssvc.vitssvc_components import (
    ConditionEncoder,
    VitsSVCContentEncoder,
    sequence_mask,
    slice_segments,
    rand_slice_segments,
)
from soniq.models.tts.vits.encoders import PosteriorEncoder
from soniq.models.tts.vits.flows import ResidualCouplingBlock
from soniq.models.vocoders.hifigan.modeling_hifigan import HiFiGANGenerator


logger = logging.get_logger(__name__)


class VitsSVC(BaseSVCModel):
    """
    VitsSVC: Singing Voice Conversion model based on VITS.

    This model converts singing voice using content features and F0 as input.

    Example:
        ```python
        config = VitsSVCConfig()
        model = VitsSVC(config)

        # Training
        batch = {
            "contentvec_feat": content_features,
            "frame_pitch": f0,
            "frame_uv": uv,
            "spk_id": speaker_id,
            "linear": mel_spec,
            "audio": audio,
        }
        output = model(batch)

        # Inference
        converted_audio = model.voice_conversion(source_audio, target_f0, speaker_id=1)
        ```
    """

    config_class = VitsSVCConfig
    base_model_prefix = "vitssvc"
    supports_gradient_checkpointing = True

    def __init__(self, config: VitsSVCConfig):
        super().__init__(config)
        self.config = config

        # Core dimensions
        self.inter_channels = config.inter_channels
        self.hidden_channels = config.hidden_channels
        self.filter_channels = config.filter_channels
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.kernel_size = config.kernel_size
        self.p_dropout = config.p_dropout
        self.n_flow_layer = config.n_flow_layer
        self.gin_channels = config.gin_channels
        self.n_speakers = config.n_speakers

        # F0 parameters
        self.n_bins = config.n_bins_f0
        self.f0_min = config.f0_min
        self.f0_max = config.f0_max

        # Segment size for training
        self.segment_size = config.segment_size

        # Speaker embedding
        self.emb_g = nn.Embedding(self.n_speakers, self.gin_channels)

        # Condition encoder
        self.condition_encoder = ConditionEncoder(
            use_contentvec=config.use_contentvec,
            contentvec_dim=config.contentvec_dim,
            use_whisper=config.use_whisper,
            whisper_dim=config.whisper_dim,
            use_wenet=config.use_wenet,
            wenet_dim=config.wenet_dim,
            use_f0=True,
            n_bins_f0=config.n_bins_f0,
            f0_min=config.f0_min,
            f0_max=config.f0_max,
            use_uv=config.use_uv,
            use_spkid=True,
            n_speakers=config.n_speakers,
            content_encoder_dim=config.hidden_channels,
            output_melody_dim=config.hidden_channels,
            output_singer_dim=config.hidden_channels,  # Same as hidden_channels for add mode
            merge_mode="add",
        )

        # Prior encoder (content encoder for SVC)
        self.enc_p = VitsSVCContentEncoder(
            out_channels=self.inter_channels,
            hidden_channels=self.hidden_channels,
            filter_channels=self.filter_channels,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout,
        )

        # Posterior encoder
        self.enc_q = PosteriorEncoder(
            in_channels=config.spec_channels,
            out_channels=self.inter_channels,
            hidden_channels=self.hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
            gin_channels=self.gin_channels,
        )

        # Flow
        self.flow = ResidualCouplingBlock(
            channels=self.inter_channels,
            hidden_channels=self.hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
            n_flows=self.n_flow_layer,
            gin_channels=self.gin_channels,
        )

        # Decoder (HiFiGAN by default)
        self.dec = HiFiGANGenerator(
            in_channels=self.inter_channels,
            out_channels=1,
            upsample_rates=config.upsample_rates,
            upsample_initial_channel=config.upsample_initial_channel,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            resblock_kernel_sizes=config.resblock_kernel_sizes,
            resblock_dilation_sizes=config.resblock_dilation_sizes,
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, module.embedding_dim ** -0.5)

    @property
    def sample_rate(self) -> int:
        """Get the audio sample rate."""
        return self.config.sample_rate

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass for training.

        Args:
            data: Batch dictionary containing:
                - contentvec_feat: ContentVec features (batch, seq_len, dim)
                - frame_pitch: F0 values (batch, seq_len)
                - frame_uv: UV flags (batch, seq_len)
                - spk_id: Speaker IDs (batch,)
                - linear: Linear spectrogram (batch, seq_len, n_fft//2+1)
                - target_len: Sequence lengths (batch,)

        Returns:
            Dictionary containing:
                - y_hat: Generated audio segment
                - ids_slice: Slice indices
                - x_mask: Condition mask
                - z_mask: Spectrogram mask
                - z: Latent from posterior
                - z_p: Latent after flow
                - m_p: Prior mean
                - logs_p: Prior log variance
                - m_q: Posterior mean
                - logs_q: Posterior log variance
        """
        # Get data
        spec = data["linear"].transpose(1, 2)  # (B, D, T)
        g = data["spk_id"]
        c_lengths = data["target_len"]
        spec_lengths = data["target_len"]
        f0 = data["frame_pitch"]

        # Speaker embedding
        g = self.emb_g(g).unsqueeze(2)  # (B, gin_channels, 1)

        # Condition encoder
        x = self.condition_encoder(data).transpose(1, 2)  # (B, hidden, T)
        x_mask = torch.unsqueeze(sequence_mask(c_lengths, f0.size(1)), 1).to(x.dtype)

        # Prior encoder
        z_p_temp, m_p, logs_p, _ = self.enc_p(x, x_mask)

        # Posterior encoder
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)

        # Flow
        z_p = self.flow(z, spec_mask, g=g)

        # Slice for decoder
        z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)

        # Decode
        o = self.dec(z_slice)

        return {
            "y_hat": o,
            "ids_slice": ids_slice,
            "x_mask": x_mask,
            "z_mask": spec_mask,
            "z": z,
            "z_p": z_p,
            "m_p": m_p,
            "logs_p": logs_p,
            "m_q": m_q,
            "logs_q": logs_q,
        }

    @torch.no_grad()
    def voice_conversion(
        self,
        source: torch.Tensor,
        target_f0: torch.Tensor,
        speaker_id: int = 0,
        content_features: Optional[torch.Tensor] = None,
        noise_scale: float = 0.35,
        **kwargs,
    ) -> torch.Tensor:
        """
        Convert source singing voice to target pitch and speaker.

        Args:
            source: Source audio or features (batch, seq_len) or (batch, 1, seq_len).
            target_f0: Target F0 contour (batch, seq_len).
            speaker_id: Target speaker ID.
            content_features: Pre-extracted content features (batch, seq_len, dim).
            noise_scale: Noise scale for sampling.
            **kwargs: Additional arguments.

        Returns:
            Converted audio (batch, 1, seq_len * hop_length).
        """
        # Prepare data
        batch_size = target_f0.shape[0]
        seq_len = target_f0.shape[1]
        device = target_f0.device

        # Create data dict for condition encoder
        data = {
            "frame_pitch": target_f0,
            "spk_id": torch.full((batch_size,), speaker_id, dtype=torch.long, device=device),
        }

        if content_features is not None:
            data["contentvec_feat"] = content_features

        # Lengths
        c_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)

        # Speaker embedding
        g = self.emb_g(data["spk_id"]).unsqueeze(2)  # (B, gin_channels, 1)

        # Condition encoder
        x = self.condition_encoder(data).transpose(1, 2)
        x_mask = torch.unsqueeze(sequence_mask(c_lengths, seq_len), 1).to(x.dtype)

        # Prior encoder
        z_p, m_p, logs_p, c_mask = self.enc_p(x, x_mask, noise_scale=noise_scale)

        # Flow (reverse)
        z = self.flow(z_p, c_mask, g=g, reverse=True)

        # Decode
        o = self.dec(z * c_mask)

        return o

    @torch.no_grad()
    def infer(
        self,
        data: Dict[str, Any],
        noise_scale: float = 0.35,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference for voice conversion.

        Args:
            data: Input data dictionary with content features, F0, and speaker ID.
            noise_scale: Noise scale for sampling.

        Returns:
            Generated audio and F0.
        """
        f0 = data["frame_pitch"]
        g = data["spk_id"]

        c_lengths = torch.full((f0.size(0),), f0.size(-1), dtype=torch.long, device=f0.device)

        # Speaker embedding
        g = self.emb_g(g).unsqueeze(2)  # (B, gin_channels, 1)

        # Condition encoder
        x = self.condition_encoder(data).transpose(1, 2)
        x_mask = torch.unsqueeze(sequence_mask(c_lengths, f0.size(1)), 1).to(x.dtype)

        # Prior encoder
        z_p, m_p, logs_p, c_mask = self.enc_p(x, x_mask, noise_scale=noise_scale)

        # Flow (reverse)
        z = self.flow(z_p, c_mask, g=g, reverse=True)

        # Decode
        o = self.dec(z * c_mask)

        return o, f0

    def synthesize(
        self,
        source: torch.Tensor,
        target_f0: torch.Tensor,
        speaker_id: int = 0,
        **kwargs,
    ) -> "VCOutput":
        """
        Synthesize converted singing voice.

        Args:
            source: Source audio.
            target_f0: Target F0 contour.
            speaker_id: Target speaker ID.
            **kwargs: Additional arguments.

        Returns:
            VCOutput with converted audio.
        """
        from soniq.models.base.outputs import VCOutput

        converted = self.voice_conversion(source, target_f0, speaker_id, **kwargs)
        return VCOutput(
            waveform=converted,
            converted_features=converted,
        )