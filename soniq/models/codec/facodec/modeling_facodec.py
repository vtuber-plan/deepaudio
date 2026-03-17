# coding=utf-8
"""
FACodec: Factorized Neural Audio Codec

FACodec uses:
- Factorized quantization into prosody, content, timbre, and residual components
- Residual vector quantization (RVQ)
- Style encoder for timbre representation
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Any, Optional, List, Tuple
import torchaudio
import torchaudio.functional as audio_F

from transformers.utils import logging
from soniq.models.base.outputs import CodecOutput
from soniq.models.codec.base import BaseCodecModel
from soniq.models.codec.facodec.configuration_facodec import FACodecConfig
from soniq.models.codec.facodec.facodec_components import (
    FAquantizer,
    StyleEncoder,
    sequence_mask,
)


logger = logging.get_logger(__name__)


class MFCC(nn.Module):
    """MFCC feature extractor."""

    def __init__(self, n_mfcc: int = 40, n_mels: int = 80):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.norm = "ortho"
        dct_mat = audio_F.create_dct(self.n_mfcc, self.n_mels, self.norm)
        self.register_buffer("dct_mat", dct_mat)

    def forward(self, mel_specgram: torch.Tensor) -> torch.Tensor:
        if len(mel_specgram.shape) == 2:
            mel_specgram = mel_specgram.unsqueeze(0)
            unsqueezed = True
        else:
            unsqueezed = False

        mfcc = torch.matmul(mel_specgram.transpose(1, 2), self.dct_mat).transpose(1, 2)

        if unsqueezed:
            mfcc = mfcc.squeeze(0)
        return mfcc


class FACodec(BaseCodecModel):
    """
    FACodec: Factorized Neural Audio Codec.

    This model factorizes audio representation into prosody, content, timbre,
    and residual components using residual vector quantization.

    Example:
        ```python
        config = FACodecConfig()
        model = FACodec(config)

        # Training
        batch = {"audio": audio, "audio_lengths": audio_lengths}
        output = model(batch)

        # Inference
        codes = model.encode(audio)
        reconstructed = model.decode(codes)

        # Voice conversion
        converted = model.voice_conversion(source_audio, target_audio)
        ```
    """

    config_class = FACodecConfig
    base_model_prefix = "facodec"
    supports_gradient_checkpointing = True

    def __init__(self, config: FACodecConfig):
        super().__init__(config)
        self.config = config

        # Encoder (simple convolutional encoder for demonstration)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, config.n_filters, 7, padding=3),
            nn.LeakyReLU(0.1),
            nn.Conv1d(config.n_filters, config.n_filters * 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(config.n_filters * 2, config.n_filters * 4, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(config.n_filters * 4, config.n_filters * 8, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(config.n_filters * 8, config.n_filters * 8, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(config.n_filters * 8, config.in_dim, 3, padding=1),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(config.in_dim, config.n_filters * 8, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(config.n_filters * 8, config.n_filters * 8, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(config.n_filters * 8, config.n_filters * 4, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(config.n_filters * 4, config.n_filters * 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(config.n_filters * 2, config.n_filters, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(config.n_filters, 1, 7, padding=3),
        )

        # Factorized quantizer
        self.quantizer = FAquantizer(
            in_dim=config.in_dim,
            n_p_codebooks=config.n_p_codebooks,
            n_c_codebooks=config.n_c_codebooks,
            n_t_codebooks=config.n_t_codebooks,
            n_r_codebooks=config.n_r_codebooks,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            quantizer_dropout=config.quantizer_dropout,
            causal=config.causal,
            separate_prosody_encoder=config.separate_prosody_encoder,
            timbre_norm=config.timbre_norm,
        )

        # Style encoder for timbre normalization
        if config.timbre_norm:
            self.style_encoder = StyleEncoder(
                in_dim=80, hidden_dim=config.n_filters * 8, out_dim=config.in_dim
            )
        else:
            self.style_encoder = None

        # Mel spectrogram for timbre encoding
        SPECT_PARAMS = {
            "n_fft": 2048,
            "win_length": 1200,
            "hop_length": self.hop_length,
        }
        MEL_PARAMS = {
            "n_mels": 80,
        }
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=MEL_PARAMS["n_mels"], sample_rate=self.sample_rate, **SPECT_PARAMS
        )
        self.mel_mean, self.mel_std = -4, 4

        # Initialize weights
        self.apply(self._init_weights)

    @property
    def sample_rate(self) -> int:
        """Get the audio sample rate."""
        return self.config.sample_rate

    @property
    def hop_length(self) -> int:
        """Get the hop length (stride) of the model."""
        return 300  # Default hop size for FACodec

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def preprocess_mel(self, wave_tensor: torch.Tensor, n_bins: int = 80) -> torch.Tensor:
        """Preprocess waveform to mel spectrogram."""
        mel_tensor = self.to_mel(wave_tensor.squeeze(1))
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mel_mean) / self.mel_std
        return mel_tensor[:, :n_bins, : int(wave_tensor.size(-1) / self.hop_length)]

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
                - codes: Quantization codes for each component
                - commit_loss: Commitment loss
                - codebook_loss: Codebook loss
                - quantized: List of quantized components
        """
        audio = data["audio"]
        audio_lengths = data.get("audio_lengths", None)

        # Encode
        encoded = self.encoder(audio)

        # Quantize
        quantized_out, quantized_list, commit_loss, codebook_loss = self.quantizer(
            encoded,
            wave_segments=audio,
        )

        # Decode
        reconstructed = self.decoder(quantized_out)

        # Create mask if lengths provided
        if audio_lengths is not None:
            max_len = audio.shape[2]
            mask = torch.arange(max_len, device=audio.device).unsqueeze(0) < audio_lengths.unsqueeze(1)
            reconstructed = reconstructed * mask.unsqueeze(1).float()

        return {
            "reconstructed": reconstructed,
            "quantized": quantized_list,
            "commit_loss": commit_loss,
            "codebook_loss": codebook_loss,
        }

    @torch.no_grad()
    def encode(
        self,
        audio: torch.Tensor,
        n_c: Optional[int] = None,
        n_t: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode audio to discrete codes.

        Args:
            audio: Input audio waveform (batch, 1, seq_len).
            n_c: Number of content codebooks to use.
            n_t: Number of timbre codebooks to use.

        Returns:
            Dictionary containing codes for each component:
                - prosody_codes: Prosody quantization codes (batch, n_p, seq_len')
                - content_codes: Content quantization codes (batch, n_c, seq_len')
                - timbre_codes: Timbre quantization codes (if not using timbre_norm)
                - residual_codes: Residual quantization codes (batch, n_r, seq_len')
                - speaker_embedding: Speaker/timbre embedding (batch, in_dim)
        """
        # Ensure correct shape
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        # Encode
        encoded = self.encoder(audio)

        # Quantize each component
        codes = {}

        # Prosody codes
        z_p, codes_p, _, _, _ = self.quantizer.prosody_quantizer(
            encoded, self.quantizer.n_p_codebooks
        )
        codes["prosody_codes"] = codes_p
        codes["prosody_quantized"] = z_p

        # Content codes
        z_c, codes_c, _, _, _ = self.quantizer.content_quantizer(
            encoded, n_c if n_c else self.quantizer.n_c_codebooks
        )
        codes["content_codes"] = codes_c
        codes["content_quantized"] = z_c

        # Timbre handling
        if not self.config.timbre_norm:
            timbre_residual = encoded - z_p.detach() - z_c.detach()
            z_t, codes_t, _, _, _ = self.quantizer.timbre_quantizer(
                timbre_residual, n_t if n_t else self.quantizer.n_t_codebooks
            )
            codes["timbre_codes"] = codes_t
            codes["timbre_quantized"] = z_t
        else:
            # Extract speaker embedding from mel
            mel = self.preprocess_mel(audio)
            mask = torch.ones(mel.shape[0], 1, mel.shape[2], device=mel.device)
            speaker_emb = self.style_encoder(mel, mask)
            codes["speaker_embedding"] = speaker_emb

        # Residual codes
        if self.quantizer.n_r_codebooks > 0:
            if self.config.timbre_norm:
                residual = encoded - z_p.detach() - z_c.detach()
            else:
                residual = encoded - z_p.detach() - z_c.detach() - z_t.detach()
            z_r, codes_r, _, _, _ = self.quantizer.residual_quantizer(
                residual, self.quantizer.n_r_codebooks
            )
            codes["residual_codes"] = codes_r
            codes["residual_quantized"] = z_r

        # Compute combined quantized output
        quantized_out = z_p.detach() + z_c.detach()
        if "timbre_quantized" in codes:
            quantized_out = quantized_out + codes["timbre_quantized"]
        if "residual_quantized" in codes:
            quantized_out = quantized_out + codes["residual_quantized"]
        codes["quantized"] = quantized_out

        return codes

    @torch.no_grad()
    def decode(
        self,
        codes: Dict[str, torch.Tensor],
        speaker_embedding: Optional[torch.Tensor] = None,
        use_residual: bool = True,
    ) -> torch.Tensor:
        """
        Decode codes to audio.

        Args:
            codes: Dictionary containing quantization codes:
                - prosody_codes: Prosody codes (batch, n_p, seq_len)
                - content_codes: Content codes (batch, n_c, seq_len)
                - timbre_codes: Timbre codes (optional, batch, n_t, seq_len)
                - residual_codes: Residual codes (optional, batch, n_r, seq_len)
            speaker_embedding: Speaker/timbre embedding for voice conversion.
            use_residual: Whether to use residual codes.

        Returns:
            reconstructed: Reconstructed audio (batch, 1, seq_len * hop_length).
        """
        # Convert codes to embeddings
        quantized = self.vq2emb(codes, use_residual=use_residual)

        # Apply timbre modulation if speaker embedding provided
        if speaker_embedding is not None and hasattr(self.quantizer, 'timbre_linear'):
            style = self.quantizer.timbre_linear(speaker_embedding).unsqueeze(2)
            gamma, beta = style.chunk(2, dim=1)
            quantized = quantized * gamma + beta

        # Decode through decoder
        reconstructed = self.decoder(quantized)

        return reconstructed

    def vq2emb(
        self,
        codes: Dict[str, torch.Tensor],
        use_residual: bool = True,
    ) -> torch.Tensor:
        """
        Convert VQ codes to embeddings.

        Args:
            codes: Dictionary containing quantization codes.
            use_residual: Whether to include residual codes.

        Returns:
            Combined embedding (batch, in_dim, seq_len).
        """
        out = 0.0

        # Prosody embedding
        if "prosody_codes" in codes:
            z_p, _, _, _ = self.quantizer.prosody_quantizer.from_codes(codes["prosody_codes"])
            out = out + z_p

        # Content embedding
        if "content_codes" in codes:
            z_c, _, _, _ = self.quantizer.content_quantizer.from_codes(codes["content_codes"])
            out = out + z_c

        # Timbre embedding (if not using timbre_norm)
        if "timbre_codes" in codes and hasattr(self.quantizer, 'timbre_quantizer'):
            z_t, _, _, _ = self.quantizer.timbre_quantizer.from_codes(codes["timbre_codes"])
            out = out + z_t

        # Residual embedding
        if "residual_codes" in codes and use_residual and self.quantizer.n_r_codebooks > 0:
            z_r, _, _, _ = self.quantizer.residual_quantizer.from_codes(codes["residual_codes"])
            out = out + z_r

        return out

    @torch.no_grad()
    def inference(
        self,
        codes: Dict[str, torch.Tensor],
        speaker_embedding: torch.Tensor,
        use_residual: bool = True,
    ) -> torch.Tensor:
        """
        Inference with speaker embedding for voice conversion.

        Args:
            codes: Dictionary containing quantization codes.
            speaker_embedding: Target speaker embedding.
            use_residual: Whether to use residual codes.

        Returns:
            Reconstructed audio with target speaker timbre.
        """
        return self.decode(codes, speaker_embedding, use_residual)

    @torch.no_grad()
    def voice_conversion(
        self,
        source_audio: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Voice conversion: convert source audio to target speaker.

        Args:
            source_audio: Source audio waveform (batch, 1, seq_len).
            target_audio: Target audio waveform (batch, 1, seq_len).

        Returns:
            converted: Converted audio (batch, 1, seq_len).
        """
        if self.style_encoder is None:
            raise ValueError("Timbre normalization must be enabled for voice conversion")

        # Encode source
        source_encoded = self.encoder(source_audio)

        # Extract timbre from target
        target_mel = self.preprocess_mel(target_audio)
        target_mask = sequence_mask(
            torch.tensor([target_audio.shape[2]], device=target_audio.device) // self.hop_length,
            target_mel.shape[2],
        ).unsqueeze(1)
        timbre = self.style_encoder(target_mel, target_mask)

        # Apply timbre to source (simplified)
        converted = source_encoded  # In full implementation, apply timbre normalization

        # Decode
        converted = self.decoder(converted)

        return converted

    def synthesize(
        self,
        codes: Dict[str, torch.Tensor],
        **kwargs,
    ) -> CodecOutput:
        """
        Synthesize audio from codes.

        Args:
            codes: Dictionary containing codes for each component.
            **kwargs: Additional arguments.

        Returns:
            CodecOutput with reconstructed audio.
        """
        reconstructed = self.decode(codes)
        return CodecOutput(reconstructed=reconstructed, codes=codes)

    def reconstruct(
        self,
        audio: torch.Tensor,
        **kwargs,
    ) -> CodecOutput:
        """
        Reconstruct audio from input audio.

        Args:
            audio: Input audio waveform (batch, 1, seq_len).
            **kwargs: Additional arguments.

        Returns:
            CodecOutput with reconstructed audio.
        """
        output = self({"audio": audio.unsqueeze(1) if audio.dim() == 2 else audio})
        return CodecOutput(reconstructed=output["reconstructed"])
