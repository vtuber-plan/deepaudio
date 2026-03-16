# coding=utf-8
"""
VALL-E (Neural Codec Language Model for Zero-Shot Text-to-Speech Synthesis)

VALL-E is a zero-shot TTS model that uses:
- Autoregressive (AR) Transformer decoder for first quantizer layer
- Non-Autoregressive (NAR) Transformer decoders for remaining quantizer layers
- Neural codec audio tokens as representation

The model learns to generate audio codes from text prompts, enabling
zero-shot synthesis by using reference audio as a prompt.
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Any, Optional, Tuple

from transformers.utils import logging
from soniq.models.base.outputs import TTSOutput
from soniq.models.tts.base import BaseTTSModel
from soniq.models.tts.valle.configuration_valle import VALLEConfig
from soniq.models.tts.valle.valle_components import (
    TokenEmbedding,
    SinePositionalEmbedding,
    ARDecoder,
    NARDecoder,
    Prenet,
)


logger = logging.get_logger(__name__)


class VALLE(BaseTTSModel):
    """
    VALL-E: Neural Codec Language Model for Zero-Shot Text-to-Speech Synthesis.

    This model uses a two-stage approach:
    1. AR decoder generates first quantizer codes autoregressively
    2. NAR decoders generate remaining quantizer codes in parallel

    Example:
        ```python
        config = VALLEConfig()
        model = VALLE(config)

        # Training
        batch = {"phone_ids": phone_ids, "phone_lengths": phone_lengths,
                 "audio_codes": audio_codes}  # (batch, n_q, seq_len)
        output = model(batch)

        # Inference
        output = model.inference(phone_ids, phone_lengths, prompt_codes=prompt_codes)
        ```
    """

    config_class = VALLEConfig
    base_model_prefix = "valle"
    supports_gradient_checkpointing = True

    def __init__(self, config: VALLEConfig):
        super().__init__(config)
        self.config = config

        # Embeddings
        self.ar_text_embedding = TokenEmbedding(
            config.decoder_dim, config.text_token_num
        )
        self.nar_text_embedding = TokenEmbedding(
            int(config.decoder_dim * config.nar_scale_factor), config.text_token_num
        )

        # Audio embeddings (add 1 for EOS token)
        audio_vocab_size = config.audio_token_num + 1
        if config.prepend_bos:
            audio_vocab_size += 1

        self.ar_audio_embedding = TokenEmbedding(
            config.decoder_dim, audio_vocab_size
        )

        # Prenet (optional)
        if config.add_prenet:
            self.ar_text_prenet = Prenet(config.decoder_dim, config.decoder_dim)
            self.ar_audio_prenet = Prenet(config.decoder_dim, config.decoder_dim)
            self.nar_text_prenet = Prenet(
                int(config.decoder_dim * config.nar_scale_factor),
                int(config.decoder_dim * config.nar_scale_factor),
            )
            self.nar_audio_prenet = Prenet(
                int(config.decoder_dim * config.nar_scale_factor),
                int(config.decoder_dim * config.nar_scale_factor),
            )
        else:
            self.ar_text_prenet = None
            self.ar_audio_prenet = None
            self.nar_text_prenet = None
            self.nar_audio_prenet = None

        # Positional embeddings
        self.ar_text_position = SinePositionalEmbedding(config.decoder_dim)
        self.ar_audio_position = SinePositionalEmbedding(config.decoder_dim)
        self.nar_text_position = SinePositionalEmbedding(
            int(config.decoder_dim * config.nar_scale_factor)
        )
        self.nar_audio_position = SinePositionalEmbedding(
            int(config.decoder_dim * config.nar_scale_factor)
        )

        # AR Decoder
        self.ar_decoder = ARDecoder(
            config.decoder_dim,
            config.nhead,
            config.num_decoder_layers,
            config.dropout,
            config.norm_first,
        )

        # AR Predictor
        self.ar_predict_layer = nn.Linear(
            config.decoder_dim, config.audio_token_num + 1
        )

        # Share embedding weights with predictor (optional)
        if config.share_embedding:
            self.ar_predict_layer.weight = self.ar_audio_embedding.emb.weight

        # NAR components
        nar_decoder_dim = int(config.decoder_dim * config.nar_scale_factor)
        self.nar_audio_embeddings = nn.ModuleList(
            [
                TokenEmbedding(nar_decoder_dim, config.audio_token_num + 1)
                for _ in range(config.num_quantizers - 1)
            ]
        )

        self.nar_decoder = NARDecoder(
            nar_decoder_dim,
            config.nhead * 2,  # Use more heads for NAR
            int(config.num_decoder_layers * config.nar_scale_factor),
            config.dropout,
            config.norm_first,
        )

        self.nar_predict_layers = nn.ModuleList(
            [
                nn.Linear(nar_decoder_dim, config.audio_token_num + 1)
                for _ in range(config.num_quantizers - 1)
            ]
        )

        # Stage embeddings for NAR
        self.nar_stage_embeddings = nn.ModuleList(
            [
                TokenEmbedding(nar_decoder_dim, config.num_quantizers)
                for _ in range(config.num_quantizers - 1)
            ]
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        data: Dict[str, Any],
        train_stage: int = 0,
    ) -> Dict[str, Any]:
        """
        Forward pass for training.

        Args:
            data: Batch dictionary containing:
                - phone_ids: Phone token IDs (batch, seq_len)
                - phone_lengths: Phone sequence lengths (batch,)
                - audio_codes: Audio codes (batch, n_q, seq_len)
            train_stage:
                - 0: Train both AR and NAR
                - 1: Train AR only
                - 2: Train NAR only

        Returns:
            Dictionary containing predictions and losses.
        """
        x = data["phone_ids"]
        x_lens = data["phone_lengths"]
        y = data["audio_codes"]  # (batch, n_q, seq_len)
        y_lens = data.get("audio_lengths", None)

        # Get batch info
        batch_size = x.shape[0]
        device = x.device

        # Training loss storage
        ar_loss = torch.tensor(0.0, device=device)
        nar_loss = torch.tensor(0.0, device=device)
        ar_acc = torch.tensor(0.0, device=device)
        nar_acc = torch.tensor(0.0, device=device)

        # ==================== AR Decoder Training ====================
        if train_stage == 0 or train_stage == 1:
            ar_loss, ar_acc = self._train_ar_decoder(x, x_lens, y, y_lens)

        # ==================== NAR Decoder Training ====================
        if train_stage == 0 or train_stage == 2:
            nar_loss, nar_acc = self._train_nar_decoder(x, x_lens, y, y_lens)

        # Combine losses
        total_loss = ar_loss + nar_loss

        return {
            "loss": total_loss,
            "ar_loss": ar_loss,
            "nar_loss": nar_loss,
            "ar_accuracy": ar_acc,
            "nar_accuracy": nar_acc,
        }

    def _train_ar_decoder(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        y_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train AR decoder."""
        batch_size = x.shape[0]
        device = x.device

        # 1. Text embedding
        x_emb = self.ar_text_embedding(x)  # (B, T_text, D)
        if self.ar_text_prenet is not None:
            x_emb = self.ar_text_prenet(x_emb)
        x_emb = self.ar_text_position(x_emb)

        # 2. Audio embedding (first quantizer only)
        y_first = y[:, 0, :]  # (B, T_audio)

        # Add prefix if using prompt mode
        if self.config.prefix_mode > 0:
            # For training, we can use a random prefix of the audio
            prefix_len = torch.randint(1, 5, (1,)).item()
            y_prefix = y_first[:, :prefix_len]
            y_input = torch.cat([y_prefix, y_first], dim=1)
        else:
            y_input = y_first

        y_emb = self.ar_audio_embedding(y_input)  # (B, T_audio, D)
        if self.ar_audio_prenet is not None:
            y_emb = self.ar_audio_prenet(y_emb)
        y_emb = self.ar_audio_position(y_emb)

        # 3. Concatenate text and audio
        xy_emb = torch.cat([x_emb, y_emb], dim=1)  # (B, T_text+T_audio, D)

        # Create attention mask
        xy_lens = x_lens + (y_input.shape[1] if y_lens is None else y_lens)
        xy_mask = self._create_mask(xy_emb, xy_lens)

        # 4. AR decoder
        ar_out = self.ar_decoder(xy_emb, mask=xy_mask)

        # 5. Get audio portion of output
        ar_audio_out = ar_out[:, x_emb.shape[1] :, :]  # (B, T_audio, D)

        # 6. Predict next token
        ar_logits = self.ar_predict_layer(ar_audio_out)  # (B, T_audio, vocab)

        # 7. Compute loss (shift by 1 for next token prediction)
        y_target = y_first  # Target is the original sequence

        # Shift predictions and targets
        ar_logits_shifted = ar_logits[:, :-1, :]  # (B, T-1, vocab)
        y_target_shifted = y_target[:, 1:].long()  # (B, T-1)

        # Flatten for cross entropy
        ar_logits_flat = ar_logits_shifted.reshape(-1, ar_logits.shape[-1])
        y_target_flat = y_target_shifted.reshape(-1)

        # Cross entropy loss
        ar_loss = F.cross_entropy(ar_logits_flat, y_target_flat, reduction="mean")

        # Compute accuracy
        ar_preds = ar_logits_flat.argmax(dim=-1)
        ar_correct = (ar_preds == y_target_flat).float().mean()

        return ar_loss, ar_correct * 100

    def _train_nar_decoder(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        y_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train NAR decoder."""
        batch_size = x.shape[0]
        num_quantizers = self.config.num_quantizers
        device = x.device

        # 1. Text embedding
        x_emb = self.nar_text_embedding(x)  # (B, T_text, D)
        if self.nar_text_prenet is not None:
            x_emb = self.nar_text_prenet(x_emb)
        x_emb = self.nar_text_position(x_emb)

        # 2. Randomly select a quantizer layer to train (1 to n_q-1)
        nar_layer_idx = torch.randint(1, num_quantizers, (1,)).item()

        # 3. Build audio input by summing embeddings from previous layers
        # For layer i, input is sum of embeddings from layers 0 to i-1
        y_cumulative = self.ar_audio_embedding(y[:, 0, :])  # Layer 0 from AR

        for i in range(1, nar_layer_idx):
            y_cumulative = y_cumulative + self.nar_audio_embeddings[i - 1](y[:, i, :])

        # Add stage embedding
        stage_emb = self.nar_stage_embeddings[nar_layer_idx - 1](
            torch.full((batch_size,), nar_layer_idx, dtype=torch.long, device=device)
        )
        y_input = y_cumulative + stage_emb.unsqueeze(1)

        if self.nar_audio_prenet is not None:
            y_input = self.nar_audio_prenet(y_input)
        y_input = self.nar_audio_position(y_input)

        # 4. Concatenate text and audio
        xy_emb = torch.cat([x_emb, y_input], dim=1)
        xy_lens = x_lens + (y_input.shape[1] if y_lens is None else y_lens)
        xy_mask = self._create_mask(xy_emb, xy_lens)

        # 5. NAR decoder
        nar_out = self.nar_decoder(xy_emb, mask=xy_mask)

        # 6. Get audio portion and predict
        nar_audio_out = nar_out[:, x_emb.shape[1] :, :]
        nar_logits = self.nar_predict_layers[nar_layer_idx - 1](nar_audio_out)

        # 7. Compute loss
        y_target = y[:, nar_layer_idx, :]

        nar_logits_flat = nar_logits.reshape(-1, nar_logits.shape[-1])
        y_target_flat = y_target.reshape(-1).long()

        nar_loss = F.cross_entropy(nar_logits_flat, y_target_flat, reduction="mean")

        # Compute accuracy
        nar_preds = nar_logits_flat.argmax(dim=-1)
        nar_correct = (nar_preds == y_target_flat).float().mean()

        return nar_loss, nar_correct * 100

    def _create_mask(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Create key padding mask from lengths."""
        if lengths is None:
            return None

        batch_size, seq_len = x.shape[:2]
        max_len = lengths.max()
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        return mask

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        prompt_codes: Optional[torch.Tensor] = None,
        top_k: int = -100,
        temperature: float = 1.0,
        max_len: int = 2000,
    ) -> Dict[str, Any]:
        """
        Inference for audio synthesis.

        Args:
            x: Phone token IDs (batch, seq_len).
            x_lengths: Phone sequence lengths (batch,).
            prompt_codes: Optional prompt audio codes (batch, n_q, prompt_len).
            top_k: Top-k sampling (negative = no clipping).
            temperature: Sampling temperature.
            max_len: Maximum output length.

        Returns:
            Dictionary containing generated audio codes.
        """
        batch_size = x.shape[0]
        device = x.device
        num_quantizers = self.config.num_quantizers

        # ==================== AR Decoder: Generate First Quantizer ====================
        # 1. Text embedding
        x_emb = self.ar_text_embedding(x)
        if self.ar_text_prenet is not None:
            x_emb = self.ar_text_prenet(x_emb)
        x_emb = self.ar_text_position(x_emb)

        # 2. Prepare prompt audio embedding
        if prompt_codes is not None:
            prompt_first = prompt_codes[:, 0, :]  # First quantizer
            prompt_emb = self.ar_audio_embedding(prompt_first)
            if self.ar_audio_prenet is not None:
                prompt_emb = self.ar_audio_prenet(prompt_emb)
            prompt_emb = self.ar_audio_position(prompt_emb)
        else:
            prompt_emb = None

        # 3. Concatenate text and prompt
        if prompt_emb is not None:
            xy_emb = torch.cat([x_emb, prompt_emb], dim=1)
            prompt_len = prompt_emb.shape[1]
        else:
            xy_emb = x_emb
            prompt_len = 0

        # Create mask
        xy_lens = x_lengths + (prompt_len if prompt_emb is not None else 0)
        xy_mask = self._create_mask(xy_emb, xy_lens)

        # 4. Autoregressive generation
        generated_codes = []
        current_emb = xy_emb
        current_mask = xy_mask
        current_len = xy_emb.shape[1]

        for _ in range(max_len):
            # Forward pass
            ar_out = self.ar_decoder(current_emb, mask=current_mask)

            # Get last output
            last_out = ar_out[:, -1:, :]  # (B, 1, D)
            logits = self.ar_predict_layer(last_out).squeeze(1)  # (B, vocab)

            # Sample next token
            if top_k > 0:
                # Top-k filtering
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            if temperature != 1.0:
                logits = logits / temperature

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)  # (B, 1)

            # Check for EOS (token ID = audio_token_num)
            is_eos = (next_token == self.config.audio_token_num).all()
            if is_eos:
                break

            generated_codes.append(next_token)

            # Embed and append for next step
            next_emb = self.ar_audio_embedding(next_token)
            if self.ar_audio_prenet is not None:
                next_emb = self.ar_audio_prenet(next_emb)
            next_emb = self.ar_audio_position(next_emb)

            current_emb = torch.cat([current_emb, next_emb], dim=1)

            # Update mask
            current_len += 1
            current_mask = self._create_mask(current_emb, torch.tensor([current_len] * batch_size, device=device))

        if len(generated_codes) == 0:
            # Fallback if generation failed
            return {"audio_codes": torch.zeros(batch_size, num_quantizers, 1, device=device)}

        # Stack generated codes (batch, generated_len)
        first_quantizer = torch.cat(generated_codes, dim=1)

        # ==================== NAR Decoder: Generate Remaining Quantizers ====================
        all_codes = [first_quantizer]

        for nar_idx in range(1, num_quantizers):
            # 1. Build cumulative audio embedding
            y_cumulative = self.ar_audio_embedding(first_quantizer)
            for i in range(1, nar_idx):
                y_cumulative = y_cumulative + self.nar_audio_embeddings[i - 1](all_codes[i])

            # Add stage embedding
            stage_ids = torch.full((batch_size,), nar_idx, dtype=torch.long, device=device)
            stage_emb = self.nar_stage_embeddings[nar_idx - 1](stage_ids)
            y_input = y_cumulative + stage_emb.unsqueeze(1)

            if self.nar_audio_prenet is not None:
                y_input = self.nar_audio_prenet(y_input)
            y_input = self.nar_audio_position(y_input)

            # 2. Concatenate with text
            xy_emb = torch.cat([x_emb, y_input], dim=1)
            xy_mask = self._create_mask(xy_emb, x_lengths + first_quantizer.shape[1])

            # 3. NAR decoder forward
            nar_out = self.nar_decoder(xy_emb, mask=xy_mask)
            nar_audio_out = nar_out[:, x_emb.shape[1] :, :]

            # 4. Predict
            nar_logits = self.nar_predict_layers[nar_idx - 1](nar_audio_out)
            nar_preds = nar_logits.argmax(dim=-1)  # (B, T)

            all_codes.append(nar_preds)

        # Stack all quantizers (batch, n_q, T)
        audio_codes = torch.stack(all_codes, dim=1)

        return {
            "audio_codes": audio_codes,
            "first_quantizer": first_quantizer,
        }

    def synthesize(
        self,
        phone_ids: torch.Tensor,
        phone_lengths: torch.Tensor,
        speaker_id: Optional[int] = None,
        prompt_codes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> TTSOutput:
        """
        Synthesize speech from phone IDs.

        Args:
            phone_ids: Phone token IDs (batch, seq_len).
            phone_lengths: Phone sequence lengths (batch,).
            speaker_id: Optional speaker ID (not used in VALL-E directly).
            prompt_codes: Optional prompt audio codes for zero-shot synthesis.
            **kwargs: Additional arguments for inference.

        Returns:
            TTSOutput with generated audio codes.
        """
        output = self.inference(
            phone_ids, phone_lengths, prompt_codes=prompt_codes, **kwargs
        )
        return TTSOutput(audio_codes=output["audio_codes"])
