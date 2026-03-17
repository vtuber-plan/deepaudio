# coding=utf-8
"""MaskGCT Semantic-to-Acoustic (S2A) model."""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Any, Optional, Tuple, List
import math

from transformers import PreTrainedModel
from soniq.models.tts.maskgct.configuration_maskgct import MaskGCTS2AConfig
from soniq.models.tts.maskgct.maskgct_components import (
    MaskGCTBackbone,
    GumbelSampler,
    get_mask_schedule,
)


class MaskGCT_S2A(PreTrainedModel):
    """
    MaskGCT Semantic-to-Acoustic (S2A) model.

    This model predicts acoustic tokens (RVQ codes) conditioned on semantic tokens.
    It uses layer-wise masking where different RVQ quantizer layers are masked
    with different probabilities.

    Example:
        ```python
        from soniq.models.tts.maskgct import MaskGCT_S2A, MaskGCTS2AConfig

        config = MaskGCTS2AConfig(
            hidden_size=1024,
            num_hidden_layers=16,
            num_quantizers=12,
            codebook_size=1024,
        )
        model = MaskGCT_S2A(config)

        # Training
        semantic_tokens = torch.randint(0, 8192, (4, 100))
        acoustic_tokens = torch.randint(0, 1024, (4, 100, 12))
        output = model(semantic_tokens, acoustic_tokens)

        # Inference
        generated = model.generate(semantic_tokens)
        ```
    """

    config_class = MaskGCTS2AConfig
    base_model_prefix = "maskgct_s2a"
    supports_gradient_checkpointing = True

    def __init__(self, config: MaskGCTS2AConfig):
        super().__init__(config)
        self.config = config

        # Layer embedding (for conditioning on which RVQ layer)
        self.layer_emb = nn.Embedding(config.num_quantizers, config.hidden_size)

        # Learnable mask token
        self.mask_emb = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        nn.init.trunc_normal_(self.mask_emb, std=0.02)

        # Token embeddings for each RVQ layer
        self.token_emb = nn.ModuleList([
            nn.Embedding(config.codebook_size, config.hidden_size)
            for _ in range(config.num_quantizers)
        ])

        # Semantic token conditioning (use semantic_codebook_size)
        self.cond_emb = nn.Embedding(config.semantic_codebook_size, config.hidden_size)

        # Transformer backbone (cond_dim = hidden_size)
        self.backbone = MaskGCTBackbone(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            cond_dim=config.hidden_size,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
        )

        # Output projections for each RVQ layer
        self.to_logits = nn.ModuleList([
            nn.Linear(config.hidden_size, config.codebook_size)
            for _ in range(config.num_quantizers)
        ])

        # Gumbel sampler
        self.sampler = GumbelSampler()

        self.post_init()

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

    def _get_mask_layer_schedule(
        self,
        step: int,
        total_steps: int,
    ) -> int:
        """
        Get which RVQ layer to mask based on schedule.

        Args:
            step: Current step
            total_steps: Total steps

        Returns:
            Layer index to mask
        """
        t = step / total_steps
        
        if self.config.mask_layer_schedule == "linear":
            # Linear schedule: mask lower layers first
            layer = int(t * self.config.num_quantizers)
        elif self.config.mask_layer_schedule == "cosine":
            # Cosine schedule
            layer = int((1 - math.cos(t * math.pi)) / 2 * self.config.num_quantizers)
        else:
            # Uniform: random layer
            layer = torch.randint(0, self.config.num_quantizers, (1,)).item()
        
        return min(layer, self.config.num_quantizers - 1)

    def _sample_mask(
        self,
        target: torch.Tensor,
        timestep: torch.Tensor,
        mask_layer: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample mask for training.

        Args:
            target: Target tokens (batch, seq_len, num_quantizers)
            timestep: Diffusion timestep (batch,)
            mask_layer: Which RVQ layer to mask

        Returns:
            mask: Boolean mask (batch, seq_len)
            layer_mask: Which layers are fully masked (num_quantizers,)
        """
        batch_size, seq_len, num_quantizers = target.shape

        # Compute mask probability from timestep
        t = timestep.float() / 1000.0
        mask_prob = torch.sin(t * math.pi / 2)

        # Mask ratio for the target layer - use mean across batch
        r = 0.6 + 0.4 * mask_prob.mean().item()  # [0.6, 1.0]
        num_mask = int(seq_len * r)

        # Create mask for the target layer
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=target.device)

        for i in range(batch_size):
            indices = torch.randperm(seq_len, device=target.device)[:num_mask]
            mask[i, indices] = True

        # Layer mask: lower layers observed, current layer partially masked, upper layers fully masked
        layer_mask = torch.zeros(num_quantizers, dtype=torch.bool, device=target.device)
        layer_mask[mask_layer + 1:] = True  # Upper layers fully masked

        return mask, layer_mask

    def forward(
        self,
        semantic_tokens: torch.Tensor,
        acoustic_tokens: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        mask_layer: Optional[int] = None,
        return_loss: bool = True,
    ) -> Dict[str, Any]:
        """
        Forward pass for training.

        Args:
            semantic_tokens: Conditioning semantic tokens (batch, seq_len)
            acoustic_tokens: Target acoustic tokens (batch, seq_len, num_quantizers)
            timestep: Diffusion timestep (batch,) or None for random
            mask_layer: Which RVQ layer to mask (None for random)
            return_loss: Whether to return loss

        Returns:
            Dictionary with logits, loss, etc.
        """
        batch_size, seq_len, num_quantizers = acoustic_tokens.shape
        device = acoustic_tokens.device

        # Sample timestep if not provided
        if timestep is None:
            timestep = torch.randint(0, 1000, (batch_size,), device=device)

        # Sample or select mask layer
        if mask_layer is None:
            mask_layer = self._get_mask_layer_schedule(
                timestep[0].item() / 1000.0 * 1000,
                1000
            )

        # Sample mask
        mask, layer_mask = self._sample_mask(acoustic_tokens, timestep, mask_layer)

        # Prepare input embeddings
        # Get semantic conditioning
        sem_emb = self.cond_emb(semantic_tokens)  # (batch, seq_len, cond_dim)

        # Prepare acoustic token embeddings
        # Shape: (batch, seq_len, hidden_size)
        acoustic_emb = torch.zeros(batch_size, seq_len, self.config.hidden_size, device=device)
        
        for k in range(num_quantizers):
            token_emb_k = self.token_emb[k](acoustic_tokens[:, :, k])
            
            # Apply layer-specific masking
            if layer_mask[k]:
                # This layer is fully masked - use mask embedding
                acoustic_emb = acoustic_emb + self.mask_emb.expand(batch_size, seq_len, -1)
            else:
                acoustic_emb = acoustic_emb + token_emb_k

        # Add layer embedding
        layer_emb = self.layer_emb(torch.tensor(mask_layer, device=device))
        hidden_states = acoustic_emb + layer_emb.unsqueeze(0).unsqueeze(0)

        # Get condition embedding from semantic tokens
        cond_embedding = sem_emb.mean(dim=1)

        # Attention mask
        attention_mask = self._create_attention_mask(seq_len, device)

        # Transformer backbone
        hidden_states = self.backbone(
            hidden_states=hidden_states,
            timestep=timestep.float(),
            cond_embedding=cond_embedding,
            attention_mask=attention_mask,
        )

        # Output logits for the masked layer
        logits = self.to_logits[mask_layer](hidden_states)

        if return_loss:
            # Compute loss only on masked tokens of the masked layer
            loss = F.cross_entropy(
                logits[mask],
                acoustic_tokens[:, :, mask_layer][mask],
                reduction='mean',
            )
        else:
            loss = None

        return {
            "logits": logits,
            "loss": loss,
            "mask": mask,
            "mask_layer": mask_layer,
            "layer_mask": layer_mask,
        }

    @torch.no_grad()
    def generate(
        self,
        semantic_tokens: torch.Tensor,
        num_steps: int = 20,
        temperature: float = 1.0,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate acoustic tokens from semantic tokens.

        Uses iterative refinement with layer-wise masking.

        Args:
            semantic_tokens: Semantic tokens (batch, seq_len)
            num_steps: Number of refinement steps
            temperature: Sampling temperature
            max_len: Maximum sequence length (default: semantic_tokens.shape[1])

        Returns:
            Generated acoustic tokens (batch, seq_len, num_quantizers)
        """
        self.eval()
        batch_size, seq_len = semantic_tokens.shape
        device = semantic_tokens.device

        if max_len is None:
            max_len = seq_len

        num_quantizers = self.config.num_quantizers

        # Initialize all tokens as masked
        acoustic_tokens = torch.zeros(
            batch_size, max_len, num_quantizers,
            dtype=torch.long, device=device
        )
        mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)

        # Get semantic conditioning
        sem_emb = self.cond_emb(semantic_tokens)
        
        # Pad or truncate semantic conditioning if needed
        if max_len > seq_len:
            sem_emb = F.pad(sem_emb, (0, 0, 0, max_len - seq_len))
        elif max_len < seq_len:
            sem_emb = sem_emb[:, :max_len, :]

        # Generate layer by layer (coarse to fine)
        for layer_idx in range(num_quantizers):
            # Reset mask for this layer
            mask[:] = True

            # Iterative refinement for this layer
            for step in range(num_steps):
                mask_ratio = 1.0 - (step / num_steps)
                num_masked = int(max_len * mask_ratio)

                # Prepare input with current layer's mask embedding
                acoustic_emb = torch.zeros(
                    batch_size, max_len, self.config.hidden_size,
                    device=device
                )

                for k in range(num_quantizers):
                    if k < layer_idx:
                        # Lower layers: use generated tokens
                        token_emb_k = self.token_emb[k](acoustic_tokens[:, :, k])
                        acoustic_emb = acoustic_emb + token_emb_k
                    elif k == layer_idx:
                        # Current layer: apply mask
                        mask_emb_expanded = self.mask_emb.expand(batch_size, max_len, -1)
                        token_emb_k = self.token_emb[k](acoustic_tokens[:, :, k])
                        token_emb_masked = torch.where(
                            mask.unsqueeze(-1),
                            mask_emb_expanded,
                            token_emb_k
                        )
                        acoustic_emb = acoustic_emb + token_emb_masked
                    else:
                        # Upper layers: fully masked
                        acoustic_emb = acoustic_emb + self.mask_emb.expand(batch_size, max_len, -1)

                # Add layer embedding
                layer_emb = self.layer_emb(torch.tensor(layer_idx, device=device))
                hidden_states = acoustic_emb + layer_emb.unsqueeze(0).unsqueeze(0)

                # Get condition embedding
                cond_embedding = sem_emb.mean(dim=1)

                # Attention mask
                attention_mask = self._create_attention_mask(max_len, device)

                # Forward pass
                hidden_states = self.backbone(
                    hidden_states=hidden_states,
                    timestep=torch.full((batch_size,), step, device=device, dtype=torch.float),
                    cond_embedding=cond_embedding,
                    attention_mask=attention_mask,
                )

                # Get logits
                logits = self.to_logits[layer_idx](hidden_states)

                # Sample tokens
                if step == num_steps - 1:
                    sampled = logits.argmax(dim=-1)
                else:
                    sampled = self.sampler(logits, temperature=temperature)

                # Update tokens
                acoustic_tokens[:, :, layer_idx] = torch.where(mask, sampled, acoustic_tokens[:, :, layer_idx])

                # Compute confidence
                probs = F.softmax(logits, dim=-1)
                confidence = -probs.log().sum(dim=-1)

                # Update mask for next iteration
                if step < num_steps - 1:
                    conf_flat = confidence.view(-1)
                    conf_flat = torch.where(mask.view(-1), conf_flat, -1e9)
                    _, top_indices = conf_flat.topk(num_masked, dim=-1)
                    new_mask = torch.zeros_like(mask).view(-1)
                    new_mask[top_indices] = True
                    mask = new_mask.view(batch_size, max_len)

        return acoustic_tokens

    @torch.no_grad()
    def inference(
        self,
        semantic_tokens: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference method for generating acoustic tokens.

        Args:
            semantic_tokens: Semantic tokens (batch, seq_len)
            **kwargs: Additional arguments for generate

        Returns:
            Dictionary with acoustic_tokens
        """
        self.eval()
        acoustic_tokens = self.generate(semantic_tokens, **kwargs)
        return {"acoustic_tokens": acoustic_tokens}
