# coding=utf-8
"""Jets: End-to-End Non-Autoregressive TTS.

Jets combines FastSpeech2-style encoder-decoder with alignment learning and HiFiGAN vocoder.
Reference: https://arxiv.org/abs/2203.16852
"""

from typing import Dict, Optional, Tuple
import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def get_mask_from_lengths(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Generate mask from lengths.

    Args:
        lengths: Length tensor (B,)
        max_len: Maximum length (default: max of lengths)

    Returns:
        Mask tensor (B, max_len)
    """
    device = lengths.device
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, device=device).unsqueeze(0).expand(batch_size, -1)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def make_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Make pad mask.

    Args:
        lengths: Length tensor (B,)

    Returns:
        Pad mask (B, T)
    """
    max_len = lengths.max().item()
    mask = get_mask_from_lengths(lengths, max_len)
    return mask


def make_non_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Make non-pad mask.

    Args:
        lengths: Length tensor (B,)

    Returns:
        Non-pad mask (B, T)
    """
    return ~make_pad_mask(lengths)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Args:
        d_model: Model dimension
        dropout: Dropout rate
        max_len: Maximum sequence length
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Args:
            x: Input tensor (B, T, D)

        Returns:
            Output with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TextEncoder(nn.Module):
    """Text encoder with transformer blocks.

    Args:
        vocab_size: Vocabulary size
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 2,
        dropout: float = 0.1,
        filter_size: int = 1024,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=filter_size,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        text: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text.

        Args:
            text: Text tokens (B, T)
            src_mask: Source mask (B, T)

        Returns:
            Encoded features (B, T, D)
        """
        x = self.embedding(text)
        x = self.pos_encoding(x)

        # Convert mask for transformer (True = ignore)
        if src_mask is not None:
            src_key_padding_mask = src_mask
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        return x


class VariancePredictor(nn.Module):
    """Variance predictor for duration/pitch/energy.

    Args:
        hidden_dim: Input/output dimension
        filter_size: Filter size for conv layers
        kernel_size: Kernel size
        dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        filter_size: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(hidden_dim, filter_size, kernel_size, padding=padding)
        self.norm1 = nn.LayerNorm(filter_size)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(filter_size, filter_size, kernel_size, padding=padding)
        self.norm2 = nn.LayerNorm(filter_size)
        self.dropout2 = nn.Dropout(dropout)

        self.linear = nn.Linear(filter_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict variance.

        Args:
            x: Input features (B, T, D)
            mask: Optional mask (B, T)

        Returns:
            Predictions (B, T)
        """
        x = x.transpose(1, 2)  # (B, D, T)

        x = self.conv1(x)
        x = x.transpose(1, 2)
        x = F.relu(self.norm1(x))
        x = self.dropout1(x)
        x = x.transpose(1, 2)

        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = F.relu(self.norm2(x))
        x = self.dropout2(x)

        x = self.linear(x).squeeze(-1)  # (B, T)

        if mask is not None:
            x = x.masked_fill(mask, 0.0)

        return x


class LengthRegulator(nn.Module):
    """Length regulator for expanding phoneme features to frame features."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        duration: torch.Tensor,
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expand phoneme features based on duration.

        Args:
            x: Phoneme features (B, T_phone, D)
            duration: Duration for each phone (B, T_phone)
            max_len: Maximum output length

        Returns:
            Expanded features (B, T_frame, D) and mel lengths (B,)
        """
        batch_size = x.shape[0]
        device = x.device

        output = []
        mel_lengths = []

        for b in range(batch_size):
            expanded = []
            for t in range(x.shape[1]):
                dur = max(int(duration[b, t].item()), 0)
                expanded.append(x[b, t:t+1].expand(dur, -1))

            if expanded:
                expanded = torch.cat(expanded, dim=0)
            else:
                expanded = x[b:b+1, :1]  # At least one frame

            output.append(expanded)
            mel_lengths.append(expanded.shape[0])

        # Pad to max length
        max_len = max(mel_lengths) if max_len is None else max_len
        output_padded = torch.zeros(batch_size, max_len, x.shape[-1], device=device)

        for b, expanded in enumerate(output):
            length = min(expanded.shape[0], max_len)
            output_padded[b, :length] = expanded[:length]

        mel_lengths = torch.tensor(mel_lengths, device=device)

        return output_padded, mel_lengths


class Decoder(nn.Module):
    """Decoder with transformer blocks.

    Args:
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        filter_size: Feed-forward filter size
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 2,
        dropout: float = 0.1,
        filter_size: int = 1024,
    ):
        super().__init__()

        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=filter_size,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(
        self,
        x: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode features.

        Args:
            x: Input features (B, T, D)
            tgt_mask: Target mask (B, T)
            memory_mask: Memory mask (encoder mask)

        Returns:
            Decoded features (B, T, D)
        """
        x = self.pos_encoding(x)

        # Create causal mask for autoregressive behavior
        T = x.shape[1]
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        tgt_key_padding_mask = tgt_mask if tgt_mask is not None else None

        x = self.decoder(
            x,
            memory=x,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        return x


class PostNet(nn.Module):
    """PostNet for mel-spectrogram refinement.

    Args:
        n_mel: Number of mel bins
        hidden_dim: Hidden dimension
        num_layers: Number of conv layers
        kernel_size: Kernel size
    """

    def __init__(
        self,
        n_mel: int = 80,
        hidden_dim: int = 256,
        num_layers: int = 5,
        kernel_size: int = 5,
    ):
        super().__init__()
        padding = kernel_size // 2

        layers = []
        for i in range(num_layers):
            in_channels = n_mel if i == 0 else hidden_dim
            out_channels = n_mel if i == num_layers - 1 else hidden_dim

            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding))
            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(out_channels))
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.5))

        self.postnet = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply postnet.

        Args:
            x: Mel spectrogram (B, T, n_mel)

        Returns:
            Refined mel spectrogram
        """
        # Skip postnet for very short sequences
        if x.shape[1] < 5:
            return torch.zeros_like(x)

        x = x.transpose(1, 2)
        x = self.postnet(x)
        x = x.transpose(1, 2)
        return x


class AlignmentModule(nn.Module):
    """Alignment learning module for text-to-mel alignment.

    Args:
        hidden_dim: Attention dimension
        mel_dim: Mel spectrogram dimension
    """

    def __init__(self, hidden_dim: int = 256, mel_dim: int = 80):
        super().__init__()

        # Text encoder
        self.t_conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.t_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

        # Mel encoder
        self.f_conv1 = nn.Conv1d(mel_dim, hidden_dim, kernel_size=3, padding=1)
        self.f_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.f_conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(
        self,
        text: torch.Tensor,
        mel: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_lengths: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate alignment attention.

        Args:
            text: Text features (B, T_text, D)
            mel: Mel spectrogram (B, T_mel, n_mel)
            text_lengths: Text lengths (B,)
            mel_lengths: Mel lengths (B,)
            text_mask: Text mask (B, T_text)

        Returns:
            Log attention probabilities (B, T_mel, T_text)
        """
        # Encode text
        t = text.transpose(1, 2)
        t = F.relu(self.t_conv1(t))
        t = self.t_conv2(t)
        t = t.transpose(1, 2)  # (B, T_text, D)

        # Encode mel
        f = mel.transpose(1, 2)
        f = F.relu(self.f_conv1(f))
        f = F.relu(self.f_conv2(f))
        f = self.f_conv3(f)
        f = f.transpose(1, 2)  # (B, T_mel, D)

        # Compute distance-based attention
        dist = f.unsqueeze(2) - t.unsqueeze(1)  # (B, T_mel, T_text, D)
        dist = torch.norm(dist, p=2, dim=-1)  # (B, T_mel, T_text)
        score = -dist

        # Apply text mask
        if text_mask is not None:
            score = score.masked_fill(text_mask.unsqueeze(1), float('-inf'))

        log_p_attn = F.log_softmax(score, dim=-1)

        return log_p_attn


class Jets(nn.Module):
    """Jets: End-to-End Non-Autoregressive TTS.

    Combines text encoder, variance adaptor, decoder, and vocoder
    for end-to-end text-to-speech.

    Args:
        vocab_size: Vocabulary size
        hidden_dim: Hidden dimension
        n_mel: Number of mel bins
        sample_rate: Audio sample rate
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        n_mel: int = 80,
        sample_rate: int = 22050,
        encoder_layers: int = 4,
        encoder_heads: int = 2,
        decoder_layers: int = 6,
        decoder_heads: int = 2,
        filter_size: int = 1024,
        variance_filter_size: int = 256,
        variance_kernel_size: int = 3,
        variance_dropout: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_mel = n_mel
        self.sample_rate = sample_rate

        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            filter_size=filter_size,
        )

        # Variance predictors
        self.duration_predictor = VariancePredictor(
            hidden_dim=hidden_dim,
            filter_size=variance_filter_size,
            kernel_size=variance_kernel_size,
            dropout=variance_dropout,
        )
        self.pitch_predictor = VariancePredictor(
            hidden_dim=hidden_dim,
            filter_size=variance_filter_size,
            kernel_size=variance_kernel_size,
            dropout=variance_dropout,
        )
        self.energy_predictor = VariancePredictor(
            hidden_dim=hidden_dim,
            filter_size=variance_filter_size,
            kernel_size=variance_kernel_size,
            dropout=variance_dropout,
        )

        # Pitch and energy embeddings
        self.pitch_embed = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.energy_embed = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)

        # Length regulator
        self.length_regulator = LengthRegulator()

        # Decoder
        self.decoder = Decoder(
            hidden_dim=hidden_dim,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            filter_size=filter_size,
        )

        # Mel projection
        self.mel_linear = nn.Linear(hidden_dim, n_mel)
        self.postnet = PostNet(n_mel=n_mel)

        # Alignment module
        self.alignment_module = AlignmentModule(hidden_dim=hidden_dim, mel_dim=n_mel)

        # Vocoder (simple HiFiGAN-like)
        self.vocoder = self._build_vocoder()

    def _build_vocoder(self) -> nn.Module:
        """Build a simple vocoder."""
        # Simple upsampling vocoder
        vocoder = nn.Sequential(
            nn.ConvTranspose1d(self.hidden_dim, 512, 16, 8, 4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(512, 256, 16, 8, 4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(256, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 1, 7, 1, 3),
        )
        return vocoder

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        mel: Optional[torch.Tensor] = None,
        mel_lengths: Optional[torch.Tensor] = None,
        pitch_target: Optional[torch.Tensor] = None,
        energy_target: Optional[torch.Tensor] = None,
        duration_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            text: Text tokens (B, T_text)
            text_lengths: Text lengths (B,)
            mel: Mel spectrogram (B, T_mel, n_mel)
            mel_lengths: Mel lengths (B,)
            pitch_target: Pitch target (B, T_text)
            energy_target: Energy target (B, T_text)
            duration_target: Duration target (B, T_text)

        Returns:
            Dictionary with outputs
        """
        # Text encoding
        text_mask = get_mask_from_lengths(text_lengths)
        encoder_out = self.text_encoder(text, text_mask)  # (B, T_text, D)

        # Alignment and duration
        if mel is not None and mel_lengths is not None:
            # Compute alignment attention
            log_p_attn = self.alignment_module(
                encoder_out, mel, text_lengths, mel_lengths,
                text_mask=make_pad_mask(text_lengths),
            )

            # Viterbi decode for duration
            durations = self._viterbi_decode(log_p_attn, text_lengths, mel_lengths)

            if duration_target is None:
                duration_target = durations
        else:
            log_p_attn = None
            durations = None

        # Predict duration
        log_duration_pred = self.duration_predictor(encoder_out, text_mask)

        # Use target duration for training
        if duration_target is not None:
            durations = duration_target

        # Predict and embed pitch
        pitch_pred = self.pitch_predictor(encoder_out, text_mask)
        if pitch_target is not None:
            pitch_emb = self.pitch_embed(pitch_target.unsqueeze(-1).transpose(1, 2)).transpose(1, 2)
        else:
            pitch_emb = self.pitch_embed(pitch_pred.unsqueeze(-1).transpose(1, 2)).transpose(1, 2)

        # Predict and embed energy
        energy_pred = self.energy_predictor(encoder_out, text_mask)
        if energy_target is not None:
            energy_emb = self.energy_embed(energy_target.unsqueeze(-1).transpose(1, 2)).transpose(1, 2)
        else:
            energy_emb = self.energy_embed(energy_pred.unsqueeze(-1).transpose(1, 2)).transpose(1, 2)

        # Add embeddings
        encoder_out = encoder_out + pitch_emb + energy_emb

        # Length regulation
        if durations is not None:
            expanded, mel_lens = self.length_regulator(encoder_out, durations)
        else:
            # Use predicted durations
            pred_durations = torch.clamp(torch.round(torch.exp(log_duration_pred) - 1), min=0).long()
            expanded, mel_lens = self.length_regulator(encoder_out, pred_durations)

        # Decode
        mel_mask = get_mask_from_lengths(mel_lens) if mel_lens is not None else None
        decoder_out = self.decoder(expanded, mel_mask)

        # Mel projection
        mel_pred = self.mel_linear(decoder_out)
        mel_postnet = mel_pred + self.postnet(mel_pred)

        outputs = {
            'mel_pred': mel_pred,
            'mel_postnet': mel_postnet,
            'log_duration_pred': log_duration_pred,
            'duration_pred': durations if durations is not None else torch.clamp(torch.round(torch.exp(log_duration_pred) - 1), min=0).long(),
            'pitch_pred': pitch_pred,
            'energy_pred': energy_pred,
            'encoder_out': encoder_out,
            'mel_lengths': mel_lens,
        }

        if log_p_attn is not None:
            outputs['log_p_attn'] = log_p_attn

        return outputs

    def _viterbi_decode(
        self,
        log_p_attn: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Decode durations from attention using Viterbi algorithm.

        Args:
            log_p_attn: Log attention probabilities (B, T_mel, T_text)
            text_lengths: Text lengths (B,)
            mel_lengths: Mel lengths (B,)

        Returns:
            Durations (B, T_text)
        """
        batch_size = log_p_attn.shape[0]
        max_text_len = log_p_attn.shape[2]
        device = log_p_attn.device

        durations = torch.zeros(batch_size, max_text_len, device=device)

        for b in range(batch_size):
            T_text = text_lengths[b].item()
            T_mel = mel_lengths[b].item()

            # Simple diagonal alignment as fallback
            # In practice, use proper Viterbi or MAS
            ratio = T_mel / max(T_text, 1)
            for t in range(T_text):
                start = int(t * ratio)
                end = int((t + 1) * ratio)
                durations[b, t] = end - start

        return durations

    def inference(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speed: float = 1.0,
        pitch_control: float = 1.0,
        energy_control: float = 1.0,
    ) -> torch.Tensor:
        """Inference mode.

        Args:
            text: Text tokens (B, T_text)
            text_lengths: Text lengths (B,)
            speed: Speed control factor
            pitch_control: Pitch control factor
            energy_control: Energy control factor

        Returns:
            Generated audio (B, 1, T)
        """
        self.eval()
        with torch.no_grad():
            # Encode text
            text_mask = get_mask_from_lengths(text_lengths)
            encoder_out = self.text_encoder(text, text_mask)

            # Predict variance
            log_duration_pred = self.duration_predictor(encoder_out, text_mask)
            pitch_pred = self.pitch_predictor(encoder_out, text_mask) * pitch_control
            energy_pred = self.energy_predictor(encoder_out, text_mask) * energy_control

            # Apply embeddings
            pitch_emb = self.pitch_embed(pitch_pred.unsqueeze(-1).transpose(1, 2)).transpose(1, 2)
            energy_emb = self.energy_embed(energy_pred.unsqueeze(-1).transpose(1, 2)).transpose(1, 2)
            encoder_out = encoder_out + pitch_emb + energy_emb

            # Length regulation with speed control
            durations = torch.clamp(torch.round(torch.exp(log_duration_pred) - 1) / speed, min=0).long()
            expanded, mel_lens = self.length_regulator(encoder_out, durations)

            # Decode
            decoder_out = self.decoder(expanded)

            # Generate mel
            mel_pred = self.mel_linear(decoder_out)
            mel_postnet = mel_pred + self.postnet(mel_pred)

            # Generate audio using vocoder
            audio = self.vocoder(decoder_out.transpose(1, 2))

        return audio, mel_postnet, durations


__all__ = [
    "Jets",
    "TextEncoder",
    "Decoder",
    "VariancePredictor",
    "LengthRegulator",
    "PostNet",
    "AlignmentModule",
    "get_mask_from_lengths",
]