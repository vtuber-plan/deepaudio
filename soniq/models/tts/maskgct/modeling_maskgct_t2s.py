# coding=utf-8
"""MaskGCT Text-to-Semantic (T2S) model."""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Any, Optional, Tuple
import math

from transformers import PreTrainedModel
from soniq.models.tts.maskgct.configuration_maskgct import MaskGCTT2SConfig
from soniq.models.tts.maskgct.maskgct_components import (
    MaskGCTBackbone,
    GumbelSampler,
    get_mask_schedule,
)


class MaskGCT_T2S(PreTrainedModel):
    """
    MaskGCT Text-to-Semantic (T2S) model.

    This model predicts semantic tokens from text using masked generative modeling.
    It follows a mask-and-predict paradigm similar to masked language modeling.

    Example:
        ```python
        from soniq.models.tts.maskgct import MaskGCT_T2S, MaskGCTT2SConfig

        config = MaskGCTT2SConfig(
            hidden_size=1536,
            num_hidden_layers=16,
            num_attention_heads=16,
            codebook_size=8192,
        )
        model = MaskGCT_T2S(config)

        # Training
        phone_ids = torch.randint(0, 1024, (4, 50))
        semantic_tokens = torch.randint(0, 8192, (4, 100))
        output = model(phone_ids, semantic_tokens)

        # Inference
        generated = model.generate(phone_ids)
        ```
    """

    config_class = MaskGCTT2SConfig
    base_model_prefix = "maskgct_t2s"
    supports_gradient_checkpointing = True

    def __init__(self, config: MaskGCTT2SConfig):
        super().__init__(config)
        self.config = config

        # Phone embedding
        self.phone_emb = nn.Embedding(config.vocab_size, config.hidden_size)

        # Semantic token conditioning embedding (output to hidden_size)
        self.cond_emb = nn.Embedding(config.codebook_size, config.hidden_size)

        # Learnable mask token
        self.mask_emb = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        nn.init.trunc_normal_(self.mask_emb, std=0.02)

        # Transformer backbone (cond_dim = hidden_size for simplicity)
        self.backbone = MaskGCTBackbone(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            cond_dim=config.hidden_size,  # Match hidden_size for adaptive norms
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
        )

        # Output projection
        self.to_logit = nn.Linear(config.hidden_size, config.codebook_size)

        # Gumbel sampler for inference
        self.sampler = GumbelSampler()

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.phone_emb

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def _create_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create bidirectional attention mask."""
        return torch.ones(seq_len, seq_len, device=device)

    def _sample_mask(
        self,
        target: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample mask for training.

        Args:
            target: Target tokens (batch, seq_len)
            timestep: Diffusion timestep (batch,)

        Returns:
            mask: Boolean mask (batch, seq_len)
            prompt_mask: Mask for prompt region (batch, seq_len)
        """
        batch_size, seq_len = target.shape

        # Compute mask probability from timestep
        # t in [0, 1], mask_prob(t) = sin(t * pi/2)
        t = timestep.float() / 1000.0  # Normalize timestep
        mask_prob = torch.sin(t * math.pi / 2)

        # Each sequence gets a random mask ratio in [mask_prob_min, mask_prob_max]
        # scaled by the diffusion timestep
        r = self.config.mask_prob_min + (self.config.mask_prob_max - self.config.mask_prob_min) * mask_prob

        # Number of tokens to mask (per batch element)
        num_mask = (seq_len * r).to(torch.long)

        # Create mask
        mask = torch.zeros_like(target, dtype=torch.bool)

        for i in range(batch_size):
            # Randomly select positions to mask
            indices = torch.randperm(seq_len, device=target.device)[:num_mask[i]]
            mask[i, indices] = True

        # Prompt region (first prompt_ratio tokens) is never masked
        prompt_len = int(seq_len * self.config.prompt_ratio)
        prompt_mask = torch.zeros_like(mask, dtype=torch.bool)
        if prompt_len > 0:
            prompt_mask[:, :prompt_len] = True
            mask[:, :prompt_len] = False

        return mask, prompt_mask

    def forward(
        self,
        phone_ids: torch.Tensor,
        semantic_tokens: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        return_loss: bool = True,
    ) -> Dict[str, Any]:
        """
        Forward pass for training.

        Args:
            phone_ids: Phone IDs (batch, phone_seq_len)
            semantic_tokens: Target semantic tokens (batch, token_seq_len)
            timestep: Diffusion timestep (batch,) or None for random
            return_loss: Whether to return loss

        Returns:
            Dictionary with logits, loss, etc.
        """
        batch_size, token_seq_len = semantic_tokens.shape

        # Sample timestep if not provided
        if timestep is None:
            timestep = torch.randint(0, 1000, (batch_size,), device=semantic_tokens.device)

        # Sample mask
        mask, prompt_mask = self._sample_mask(semantic_tokens, timestep)

        # Create input sequence
        # Replace masked tokens with mask embedding
        x = self.phone_emb(phone_ids)  # (batch, phone_seq_len, hidden)

        # Prepare semantic token embeddings
        sem_emb = self.cond_emb(semantic_tokens)  # (batch, token_seq_len, hidden_size)

        # Apply mask to semantic tokens
        mask_emb_expanded = self.mask_emb.expand(batch_size, token_seq_len, -1)
        sem_emb_masked = torch.where(mask.unsqueeze(-1), mask_emb_expanded, sem_emb)

        # Add phone conditioning
        phone_emb_expanded = x.mean(dim=1, keepdim=True).expand(-1, token_seq_len, -1)
        hidden_states = sem_emb_masked + phone_emb_expanded

        # Get condition embedding (from unmasked tokens)
        cond_embedding = sem_emb.mean(dim=1)  # Simple pooling

        # Attention mask
        attention_mask = self._create_attention_mask(token_seq_len, semantic_tokens.device)

        # Transformer backbone
        hidden_states = self.backbone(
            hidden_states=hidden_states,
            timestep=timestep.float(),
            cond_embedding=cond_embedding,
            attention_mask=attention_mask,
        )

        # Output logits
        logits = self.to_logit(hidden_states)  # (batch, token_seq_len, codebook_size)

        if return_loss:
            # Compute loss only on masked tokens
            loss = F.cross_entropy(
                logits[mask],
                semantic_tokens[mask],
                reduction='mean',
            )
        else:
            loss = None

        return {
            "logits": logits,
            "loss": loss,
            "mask": mask,
            "prompt_mask": prompt_mask,
        }

    @torch.no_grad()
    def generate(
        self,
        phone_ids: torch.Tensor,
        num_steps: int = 20,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        max_len: int = 200,
    ) -> torch.Tensor:
        """
        Generate semantic tokens from phone IDs.

        Uses iterative refinement with mask-and-predict.

        Args:
            phone_ids: Phone IDs (batch, phone_seq_len)
            num_steps: Number of refinement steps
            temperature: Sampling temperature
            guidance_scale: CFG scale (not implemented, placeholder)
            max_len: Maximum sequence length

        Returns:
            Generated semantic tokens (batch, token_seq_len)
        """
        self.eval()
        batch_size, phone_seq_len = phone_ids.shape
        device = phone_ids.device

        # Initialize all tokens as masked
        seq_len = max_len
        semantic_tokens = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        # Phone embedding for conditioning
        x = self.phone_emb(phone_ids)
        phone_cond = x.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)

        # Iterative refinement
        for step in range(num_steps):
            # Compute current mask ratio
            mask_ratio = 1.0 - (step / num_steps)
            num_masked = int(seq_len * mask_ratio)

            # Prepare input
            mask_emb_expanded = self.mask_emb.expand(batch_size, seq_len, -1)
            sem_emb = self.cond_emb(semantic_tokens)
            sem_emb_masked = torch.where(mask.unsqueeze(-1), mask_emb_expanded, sem_emb)
            hidden_states = sem_emb_masked + phone_cond

            # Get condition embedding
            # Use unmasked tokens for conditioning
            unmasked = ~mask
            if unmasked.any():
                cond_embedding = (sem_emb * unmasked.unsqueeze(-1).float()).sum(dim=1) / unmasked.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                cond_embedding = sem_emb.mean(dim=1)

            # Attention mask
            attention_mask = self._create_attention_mask(seq_len, device)

            # Forward pass
            hidden_states = self.backbone(
                hidden_states=hidden_states,
                timestep=torch.full((batch_size,), step, device=device, dtype=torch.float),
                cond_embedding=cond_embedding,
                attention_mask=attention_mask,
            )

            # Get logits for masked positions
            logits = self.to_logit(hidden_states)

            # Sample tokens for masked positions
            if step == num_steps - 1:
                # Final step: take argmax
                sampled = logits.argmax(dim=-1)
            else:
                # Sample with Gumbel-softmax
                sampled = self.sampler(logits, temperature=temperature)

            # Update semantic tokens
            semantic_tokens = torch.where(mask, sampled, semantic_tokens)

            # Compute confidence scores (entropy-based)
            probs = F.softmax(logits, dim=-1)
            confidence = -probs.log().sum(dim=-1)  # Lower entropy = higher confidence

            # Keep only the most confident predictions
            if step < num_steps - 1:
                # Sort by confidence
                conf_flat = confidence.view(-1)
                conf_flat = torch.where(mask.view(-1), conf_flat, -1e9)

                # Keep top num_masked confident predictions
                _, top_indices = conf_flat.topk(num_masked, dim=-1)
                new_mask = torch.zeros_like(mask).view(-1)
                new_mask[top_indices] = True
                mask = new_mask.view(batch_size, seq_len)

        return semantic_tokens

    @torch.no_grad()
    def inference(
        self,
        phone_ids: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference method for generating semantic tokens.

        Args:
            phone_ids: Phone IDs (batch, phone_seq_len)
            **kwargs: Additional arguments for generate

        Returns:
            Dictionary with semantic_tokens
        """
        self.eval()
        semantic_tokens = self.generate(phone_ids, **kwargs)
        return {"semantic_tokens": semantic_tokens}
