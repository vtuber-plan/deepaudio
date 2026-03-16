# coding=utf-8
"""
FastSpeech2: Fast and High-Quality Text-to-Speech with Parallel Decoders.

FastSpeech2 is a non-autoregressive TTS model that uses:
- Transformer encoder-decoder architecture
- Duration predictor for length regulation
- Pitch and energy predictors for prosody modeling
- PostNet for refining mel spectrogram predictions
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Any, Optional, Tuple

from transformers.utils import logging
from soniq.models.base.outputs import TTSOutput
from soniq.models.tts.base import BaseTTSModel
from soniq.models.tts.fastspeech2.configuration_fastspeech2 import FastSpeech2Config
from soniq.models.tts.fastspeech2.fastspeech2_components import (
    FS2Encoder,
    FS2Decoder,
    VariancePredictor,
    VarianceEmbedding,
    LengthRegulator,
    PostNet,
    get_mask_from_lengths,
    expand,
)


logger = logging.get_logger(__name__)


class FastSpeech2(BaseTTSModel):
    """
    FastSpeech2: Fast and High-Quality Text-to-Speech with Parallel Decoders.

    This model generates mel spectrograms directly from text using a
    non-autoregressive Transformer architecture with variance prediction.

    Example:
        ```python
        config = FastSpeech2Config()
        model = FastSpeech2(config)

        # Training
        batch = {"texts": text_ids, "text_len": text_lengths,
                 "mel": mel_spec, "target_len": mel_lengths,
                 "pitch": pitch, "energy": energy, "durations": durations}
        output = model(batch)

        # Inference
        output = model.infer(text_ids, text_lengths)
        ```
    """

    config_class = FastSpeech2Config
    base_model_prefix = "fastspeech2"
    supports_gradient_checkpointing = False

    def __init__(self, config: FastSpeech2Config):
        super().__init__(config)
        self.config = config

        # Encoder
        self.encoder = FS2Encoder(
            n_vocab=config.n_vocab,
            hidden_channels=config.hidden_channels,
            n_layers=config.encoder_layers,
            n_heads=config.n_heads,
            filter_channels=config.filter_channels,
            dropout=config.encoder_dropout,
        )

        # Variance Adaptor components
        self.duration_predictor = VariancePredictor(
            in_channels=config.hidden_channels,
            filter_channels=config.variance_predictor_filter_size,
            kernel_size=config.variance_predictor_kernel_size,
            dropout=config.variance_predictor_dropout,
        )

        self.length_regulator = LengthRegulator()

        self.pitch_predictor = VariancePredictor(
            in_channels=config.hidden_channels,
            filter_channels=config.variance_predictor_filter_size,
            kernel_size=config.variance_predictor_kernel_size,
            dropout=config.variance_predictor_dropout,
        )

        self.energy_predictor = VariancePredictor(
            in_channels=config.hidden_channels,
            filter_channels=config.variance_predictor_filter_size,
            kernel_size=config.variance_predictor_kernel_size,
            dropout=config.variance_predictor_dropout,
        )

        # Variance embeddings
        self.pitch_embedding = VarianceEmbedding(config.pitch_n_bins, config.hidden_channels)
        self.energy_embedding = VarianceEmbedding(config.energy_n_bins, config.hidden_channels)

        # Pitch and energy bin boundaries
        self.register_buffer(
            "pitch_bins",
            torch.linspace(config.pitch_min, config.pitch_max, config.pitch_n_bins - 1)
        )
        self.register_buffer(
            "energy_bins",
            torch.linspace(config.energy_min, config.energy_max, config.energy_n_bins - 1)
        )

        # Decoder
        self.decoder = FS2Decoder(
            hidden_channels=config.hidden_channels,
            n_layers=config.decoder_layers,
            n_heads=config.n_heads,
            filter_channels=config.filter_channels,
            dropout=config.decoder_dropout,
        )

        # Mel projection
        self.mel_linear = nn.Linear(config.hidden_channels, config.n_mel)

        # PostNet
        self.postnet = PostNet(n_mel=config.n_mel)

        # Speaker embedding
        if config.n_speakers > 0:
            self.speaker_emb = nn.Embedding(config.n_speakers, config.hidden_channels)
        else:
            self.speaker_emb = None

    def forward(
        self,
        data: Dict[str, Any],
        p_control: float = 1.0,
        e_control: float = 1.0,
        d_control: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Forward pass for training.

        Args:
            data: Batch dictionary containing:
                - texts: Text token IDs (batch, seq_len)
                - text_len: Text sequence lengths (batch,)
                - mel: Mel spectrogram (batch, time, n_mel)
                - target_len: Mel sequence lengths (batch,)
                - pitch: Pitch values (batch, seq_len) or (batch, time)
                - energy: Energy values (batch, seq_len) or (batch, time)
                - durations: Duration targets (batch, seq_len)
                - spk_id: Optional speaker IDs (batch,)
            p_control: Pitch control factor (1.0 = no change).
            e_control: Energy control factor (1.0 = no change).
            d_control: Duration control factor (1.0 = no change).

        Returns:
            Dictionary containing predictions and losses.
        """
        # Extract inputs
        texts = data["texts"]
        text_lengths = data["text_len"]
        mel_targets = data.get("mel")  # (batch, time, n_mel)
        target_lengths = data.get("target_len")

        # Get variance targets if available
        pitch_targets = data.get("pitch")
        energy_targets = data.get("energy")
        duration_targets = data.get("durations")

        # Get speaker embedding
        spk_id = data.get("spk_id")
        spk_emb = None
        if spk_id is not None and self.speaker_emb is not None:
            spk_emb = self.speaker_emb(spk_id).unsqueeze(1)

        # 1. Encoder
        encoder_output, src_mask = self.encoder(texts, text_lengths)

        # Add speaker embedding
        if spk_emb is not None:
            encoder_output = encoder_output + spk_emb

        # 2. Duration prediction
        log_duration_pred = self.duration_predictor(encoder_output, src_mask).squeeze(-1)

        # 3. Get durations (use ground truth if available)
        if duration_targets is not None:
            # Training: use ground truth
            durations = duration_targets
        else:
            # Inference: use predicted (rounded)
            durations = torch.clamp((torch.exp(log_duration_pred) - 1).round(), min=0).long()

        # Apply duration control
        if d_control != 1.0:
            durations = (durations * d_control).long()

        # 4. Length regulation
        max_target_len = target_lengths.max().item() if target_lengths is not None else None
        decoder_input, mel_masks = self._length_regulate(
            encoder_output, durations, max_target_len, src_mask
        )

        # 5. Add pitch and energy embeddings
        if pitch_targets is not None:
            # Discretize and embed pitch
            pitch_embedded = self._embed_variance(
                pitch_targets, self.pitch_bins, self.pitch_embedding, mel_masks
            )
            decoder_input = decoder_input + pitch_embedded

        if energy_targets is not None:
            # Discretize and embed energy
            energy_embedded = self._embed_variance(
                energy_targets, self.energy_bins, self.energy_embedding, mel_masks
            )
            decoder_input = decoder_input + energy_embedded

        # 6. Predict pitch and energy (for loss computation)
        pitch_pred = self.pitch_predictor(decoder_input, mel_masks).squeeze(-1)
        energy_pred = self.energy_predictor(decoder_input, mel_masks).squeeze(-1)

        # 7. Decoder
        decoder_output = self.decoder(decoder_input, mel_masks)

        # 8. Project to mel spectrogram
        mel_output = self.mel_linear(decoder_output)  # (batch, time, n_mel)

        # 9. PostNet refinement
        mel_output_transposed = mel_output.transpose(1, 2)  # (batch, n_mel, time)
        postnet_output = mel_output_transposed + self.postnet(mel_output_transposed)
        postnet_output = postnet_output.transpose(1, 2)  # (batch, time, n_mel)

        result = {
            "mel_output": mel_output,
            "postnet_output": postnet_output,
            "pitch_predictions": pitch_pred,
            "energy_predictions": energy_pred,
            "log_duration_predictions": log_duration_pred,
            "durations": durations,
            "mel_masks": ~mel_masks if mel_masks is not None else None,
        }

        # Compute losses if targets are available
        if mel_targets is not None:
            losses = self._compute_losses(
                mel_output, postnet_output, pitch_pred, energy_pred,
                log_duration_pred, duration_targets, pitch_targets,
                energy_targets, mel_masks
            )
            result.update(losses)

        return result

    def _length_regulate(self, x, durations, max_len, src_mask):
        """Apply length regulation to expand encoder output."""
        # Expand based on durations
        expanded, lengths = self.length_regulator(x, durations, max_len)

        # Create mask
        if lengths is not None:
            mask = get_mask_from_lengths(lengths, max_len)
        else:
            mask = None

        return expanded, mask

    def _embed_variance(self, variance, bins, embedding, mask):
        """Discretize variance and get embedding."""
        # Discretize
        if mask is not None:
            # Only discretize valid positions
            variance_flat = variance[mask]
            indices = torch.bucketize(variance_flat, bins)
            indices = indices.clamp(0, embedding.embedding.num_embeddings - 1)

            # Create full tensor
            full_indices = torch.zeros_like(variance, dtype=torch.long)
            full_indices[mask] = indices
        else:
            indices = torch.bucketize(variance, bins)
            indices = indices.clamp(0, embedding.embedding.num_embeddings - 1)

        # Embed
        return embedding(indices)

    def _compute_losses(
        self, mel_output, postnet_output, pitch_pred, energy_pred,
        log_duration_pred, duration_targets, pitch_targets,
        energy_targets, mel_masks
    ):
        """Compute FastSpeech2 losses."""
        # Mel loss (MAE)
        if mel_masks is not None:
            mel_loss = F.l1_loss(mel_output[mel_masks], mel_output[mel_masks])
            postnet_loss = F.l1_loss(postnet_output[mel_masks], mel_output[mel_masks])
        else:
            mel_loss = F.l1_loss(mel_output, mel_output)
            postnet_loss = F.l1_loss(postnet_output, mel_output)

        # Duration loss (MSE on log domain)
        if duration_targets is not None:
            log_duration_targets = torch.log(duration_targets.float() + 1)
            duration_loss = F.mse_loss(log_duration_pred, log_duration_targets)
        else:
            duration_loss = torch.tensor(0.0, device=log_duration_pred.device)

        # Pitch loss (MSE)
        if pitch_targets is not None:
            pitch_loss = F.mse_loss(pitch_pred, pitch_targets)
        else:
            pitch_loss = torch.tensor(0.0, device=pitch_pred.device)

        # Energy loss (MSE)
        if energy_targets is not None:
            energy_loss = F.mse_loss(energy_pred, energy_targets)
        else:
            energy_loss = torch.tensor(0.0, device=energy_pred.device)

        total_loss = mel_loss + postnet_loss + duration_loss + pitch_loss + energy_loss

        return {
            "total_loss": total_loss,
            "mel_loss": mel_loss,
            "postnet_loss": postnet_loss,
            "duration_loss": duration_loss,
            "pitch_loss": pitch_loss,
            "energy_loss": energy_loss,
        }

    @torch.no_grad()
    def infer(
        self,
        texts: torch.Tensor,
        text_lengths: torch.Tensor,
        speaker_id: Optional[int] = None,
        p_control: float = 1.0,
        e_control: float = 1.0,
        d_control: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Inference for mel spectrogram synthesis.

        Args:
            texts: Text token IDs of shape (batch, seq_len).
            text_lengths: Text sequence lengths of shape (batch,).
            speaker_id: Optional speaker ID.
            p_control: Pitch control factor.
            e_control: Energy control factor.
            d_control: Duration control factor.

        Returns:
            Dictionary containing:
                - mel_output: Generated mel spectrogram
                - postnet_output: Refined mel spectrogram
        """
        # Get speaker embedding
        spk_id = None
        if speaker_id is not None and self.speaker_emb is not None:
            spk_id = torch.tensor([[speaker_id]], device=texts.device)

        data = {
            "texts": texts,
            "text_len": text_lengths,
            "spk_id": spk_id,
        }

        output = self.forward(
            data,
            p_control=p_control,
            e_control=e_control,
            d_control=d_control,
        )

        return output

    def synthesize(
        self,
        text_ids: torch.Tensor,
        text_lengths: torch.Tensor,
        speaker_id: Optional[int] = None,
        **kwargs,
    ) -> TTSOutput:
        """
        Synthesize mel spectrogram from text.

        Args:
            text_ids: Text token IDs of shape (batch, seq_len).
            text_lengths: Text sequence lengths of shape (batch,).
            speaker_id: Optional speaker ID.
            **kwargs: Additional arguments for inference.

        Returns:
            TTSOutput with mel spectrogram.
        """
        output = self.infer(text_ids, text_lengths, speaker_id=speaker_id, **kwargs)
        mel = output["postnet_output"].transpose(1, 2)  # (batch, n_mel, time)
        return TTSOutput(mel=mel)
