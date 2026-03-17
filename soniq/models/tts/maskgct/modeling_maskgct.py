# coding=utf-8
"""
MaskGCT: Fully Non-Autoregressive Text-to-Speech.

MaskGCT is a two-stage TTS model:
1. T2S: Text to Semantic tokens
2. S2A: Semantic to Acoustic tokens

Reference: "MaskGCT: Zero-Shot Text-to-Speech with Learned Semantic Token"
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, List, Union

from transformers import PreTrainedModel
from .configuration_maskgct import (
    MaskGCTConfig,
    MaskGCT_T2S_Config,
    MaskGCT_S2A_Config,
)


# ============================================================================
# Utility Functions
# ============================================================================

def log(t: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Safe log."""
    return torch.log(t + eps)


def gumbel_noise(t: torch.Tensor) -> torch.Tensor:
    """Generate Gumbel noise."""
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t: torch.Tensor, temperature: float = 1.0, dim: int = -1) -> torch.Tensor:
    """Gumbel sampling."""
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits: torch.Tensor, thres: float = 0.9) -> torch.Tensor:
    """Top-k filtering."""
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(2, ind, val)
    return probs


# ============================================================================
# Sinusoidal Positional Encoding
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional encoding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :] * 1.0
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ============================================================================
# Adaptive RMS Norm
# ============================================================================

class AdaptiveRMSNorm(nn.Module):
    """Adaptive RMS normalization for conditioning."""

    def __init__(self, hidden_size: int = 1024, eps: float = 1e-6, dim_cond: int = 1024):
        super().__init__()
        self.to_weight = nn.Linear(dim_cond, hidden_size)
        nn.init.zeros_(self.to_weight.weight)
        nn.init.ones_(self.to_weight.bias)
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, cond_embedding: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        weight = self.to_weight(cond_embedding)
        if len(weight.shape) == 2:
            weight = weight.unsqueeze(1)

        return (weight * hidden_states).to(input_dtype)


# ============================================================================
# Diffusion Transformer Block
# ============================================================================

class DiffusionTransformerBlock(nn.Module):
    """Transformer block with adaptive layer norm for diffusion."""

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        intermediate_size: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Adaptive layer norms
        self.input_layernorm = AdaptiveRMSNorm(hidden_size, dim_cond=hidden_size)
        self.post_attention_layernorm = AdaptiveRMSNorm(hidden_size, dim_cond=hidden_size)

        # Self attention
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # MLP
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_embedding: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        # Pre-norm
        hidden_states = self.input_layernorm(hidden_states, cond_embedding)

        # Self attention
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        hidden_states = residual + attn_output

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states, cond_embedding)

        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        hidden_states = self.down_proj(gate * up)

        hidden_states = residual + hidden_states

        return hidden_states


# ============================================================================
# Diffusion Transformer
# ============================================================================

class DiffusionTransformer(nn.Module):
    """Diffusion Transformer backbone for MaskGCT."""

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_layers: int = 16,
        intermediate_size: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            DiffusionTransformerBlock(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])

        self.norm = AdaptiveRMSNorm(hidden_size, dim_cond=hidden_size)

        # Diffusion step embedding
        self.diff_step_embedding = SinusoidalPosEmb(hidden_size)
        self.diff_step_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare attention mask."""
        batch_size, seq_len = attention_mask.size()

        # Expand to [batch, 1, 1, seq_len]
        expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_len, seq_len).to(dtype)

        # Convert 0s to -inf and 1s to 0
        inverted_mask = 1.0 - expanded_mask
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    def forward(
        self,
        x: torch.Tensor,
        diffusion_step: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, T, C).
            diffusion_step: Diffusion timestep (B,).
            x_mask: Attention mask (B, T), 1 for valid, 0 for padding.
            cond: Conditioning tensor (B, T, C).

        Returns:
            Output tensor (B, T, C).
        """
        # Add conditioning
        if cond is not None:
            x = x + cond

        # Diffusion step embedding
        diffusion_step = self.diff_step_embedding(diffusion_step).to(x.device)
        diffusion_step = self.diff_step_mlp(diffusion_step)

        # Prepare attention mask
        if x_mask is not None:
            attention_mask = self._prepare_attention_mask(x_mask, x.dtype)
        else:
            attention_mask = None

        # Apply layers
        hidden_states = x
        for layer in self.layers:
            hidden_states = layer(hidden_states, diffusion_step, attention_mask)

        # Final norm
        hidden_states = self.norm(hidden_states, diffusion_step)

        return hidden_states


# ============================================================================
# MaskGCT T2S (Text to Semantic)
# ============================================================================

class MaskGCT_T2S(PreTrainedModel):
    """
    MaskGCT Text-to-Semantic model.

    Converts text/phone tokens to semantic tokens using
    mask-and-predict diffusion.
    """

    config_class = MaskGCT_T2S_Config
    base_model_prefix = "maskgct_t2s"

    def __init__(self, config: MaskGCT_T2S_Config):
        super().__init__(config)
        self.config = config

        self.hidden_size = config.hidden_size

        # Embeddings
        self.mask_emb = nn.Embedding(1, config.hidden_size)
        self.cond_emb = nn.Embedding(config.cond_codebook_size, config.hidden_size)

        if config.use_phone_cond:
            self.phone_emb = nn.Embedding(config.phone_vocab_size, config.hidden_size, padding_idx=config.phone_vocab_size - 1)

        # Output projection
        self.to_logit = nn.Linear(config.hidden_size, config.cond_codebook_size)

        # Diffusion transformer
        self.diff_estimator = DiffusionTransformer(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
        )

    def mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal mask probability."""
        return torch.sin(t * np.pi / 2).to(t.device)

    def forward_diffusion(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward diffusion process."""
        mask_prob = self.mask_prob(t)
        mask_prob = torch.where(mask_prob < 0.2, torch.ones_like(mask_prob) * 0.2, mask_prob)

        mask_token = self.mask_emb(torch.LongTensor([0]).to(x0.device))  # (1, hidden_size)

        # Random masking
        mask = torch.bernoulli(torch.ones(x0.shape[0], x0.shape[1], device=x0.device) * mask_prob[..., None])
        # mask shape: (B, T)

        # Get conditional embedding
        cond = self.cond_emb(x0)  # (B, T, hidden_size)

        # Apply mask
        xt = mask.unsqueeze(-1) * mask_token.view(1, 1, -1) + (1 - mask).unsqueeze(-1) * cond
        # xt shape: (B, T, hidden_size)

        return xt, t, mask

    def forward(
        self,
        x0: torch.Tensor,
        x_mask: torch.Tensor,
        phone_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass.

        Args:
            x0: Semantic tokens (B, T).
            x_mask: Attention mask (B, T).
            phone_id: Phone token IDs (B, phone_len).

        Returns:
            logits: Predicted logits (B, T, codebook_size).
            mask: Mask used for diffusion.
            x0: Original semantic tokens.
        """
        # Sample timestep
        t = torch.rand(x0.shape[0], device=x0.device, requires_grad=False)
        t = torch.clamp(t, 1e-5, 1.0)

        # Forward diffusion
        xt, t, mask = self.forward_diffusion(x0, t)

        # Get phone embedding
        phone_embedding = None
        if self.config.use_phone_cond and phone_id is not None:
            phone_embedding = self.phone_emb(phone_id)
            # Concat phone embedding to input
            xt = torch.cat([phone_embedding, xt], dim=1)
            x_mask = torch.cat([torch.ones_like(phone_id), x_mask], dim=1)

        # Diffusion estimation
        embeds = self.diff_estimator(xt, t, x_mask)

        # Remove phone part
        if phone_embedding is not None:
            embeds = embeds[:, phone_embedding.shape[1]:]

        logits = self.to_logit(embeds)

        return logits, mask, x0

    @torch.no_grad()
    def synthesize(
        self,
        phone_id: torch.Tensor,
        target_len: int,
        prompt: Optional[torch.Tensor] = None,
        n_timesteps: int = 40,
        temperature: float = 0.9,
        filter_threshold: float = 0.98,
    ) -> torch.Tensor:
        """
        Synthesize semantic tokens from phone IDs.

        Args:
            phone_id: Phone token IDs (B, phone_len).
            target_len: Target sequence length.
            prompt: Optional prompt semantic tokens (B, prompt_len).
            n_timesteps: Number of diffusion steps.
            temperature: Sampling temperature.
            filter_threshold: Top-k filter threshold.

        Returns:
            Semantic tokens (B, target_len).
        """
        batch_size = phone_id.shape[0]
        device = phone_id.device

        # Initialize with mask tokens
        seq = torch.zeros(batch_size, target_len, dtype=torch.long, device=device)
        mask = torch.ones(batch_size, target_len, dtype=torch.bool, device=device)

        phone_embedding = self.phone_emb(phone_id) if self.config.use_phone_cond else None

        h = 1.0 / n_timesteps

        for i in range(n_timesteps):
            t = (1.0 - i * h) * torch.ones(batch_size, device=device)

            # Get embeddings
            token = self.cond_emb(seq)
            cur = mask[..., None] * self.mask_emb(torch.zeros(1, dtype=torch.long, device=device)) + (~mask[..., None]) * token

            if phone_embedding is not None:
                cur = torch.cat([phone_embedding, cur], dim=1)
                cur_mask = torch.cat([torch.ones(batch_size, phone_embedding.shape[1], device=device), mask.float()], dim=1)
            else:
                cur_mask = mask.float()

            # Estimate
            embeds = self.diff_estimator(cur, t, cur_mask)
            if phone_embedding is not None:
                embeds = embeds[:, phone_embedding.shape[1]:]

            logits = self.to_logit(embeds)
            logits = top_k(logits, filter_threshold)

            # Sample
            sampled_ids = gumbel_sample(logits, temperature=max(temperature * (1.0 - i * h), 1e-3))
            seq = torch.where(mask, sampled_ids, seq)

            # Update mask
            if i < n_timesteps - 1:
                next_t = (1.0 - (i + 1) * h) * torch.ones(batch_size, device=device)
                next_mask_num = int(self.mask_prob(next_t)[0].item() * target_len)

                if next_mask_num > 0:
                    # Score-based masking
                    scores = logits.softmax(dim=-1).gather(2, sampled_ids.unsqueeze(-1)).squeeze(-1)
                    scores = 1 - scores
                    scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)

                    mask_indices = scores.topk(min(next_mask_num, target_len), dim=-1).indices
                    mask = torch.zeros_like(scores, dtype=torch.bool).scatter(1, mask_indices, True)
                    seq = seq.masked_fill(mask, 0)
                else:
                    break

        return seq


# ============================================================================
# MaskGCT S2A (Semantic to Acoustic)
# ============================================================================

class MaskGCT_S2A(PreTrainedModel):
    """
    MaskGCT Semantic-to-Acoustic model.

    Converts semantic tokens to acoustic tokens using
    hierarchical mask-and-predict diffusion.
    """

    config_class = MaskGCT_S2A_Config
    base_model_prefix = "maskgct_s2a"

    def __init__(self, config: MaskGCT_S2A_Config):
        super().__init__(config)
        self.config = config

        self.num_quantizers = config.num_quantizers
        self.hidden_size = config.hidden_size

        # Embeddings
        self.layer_emb = nn.Embedding(config.num_quantizers, config.hidden_size)
        self.mask_emb = nn.Embedding(1, config.hidden_size)
        self.cond_emb = nn.Embedding(config.cond_codebook_size, config.hidden_size)

        # Per-layer token embeddings and logits
        self.token_emb = nn.ModuleList([
            nn.Embedding(config.codebook_size, config.hidden_size)
            for _ in range(config.num_quantizers)
        ])
        self.to_logits = nn.ModuleList([
            nn.Linear(config.hidden_size, config.codebook_size)
            for _ in range(config.num_quantizers)
        ])

        # Diffusion transformer
        self.diff_estimator = DiffusionTransformer(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
        )

    def mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal mask probability."""
        return torch.sin(t * np.pi / 2).to(t.device)

    def forward(
        self,
        x0: torch.Tensor,
        x_mask: torch.Tensor,
        cond_code: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass.

        Args:
            x0: Acoustic tokens (B, T, num_quantizers).
            x_mask: Attention mask (B, T).
            cond_code: Semantic tokens (B, T).

        Returns:
            logits: Predicted logits.
            mask_layer: Selected layer for masking.
            mask: Mask used for diffusion.
            x0: Original acoustic tokens.
        """
        batch_size, seq_len, num_q = x0.shape

        # Sample timestep and layer
        t = torch.rand(batch_size, device=x0.device, requires_grad=False)
        t = torch.clamp(t, 1e-5, 1.0)

        mask_layer = torch.randint(0, self.num_quantizers, (1,), device=x0.device)
        mask_prob = self.mask_prob(t)

        mask_token = self.mask_emb(torch.zeros_like(mask_layer))

        # Build input with hierarchical masking
        xt = torch.zeros(batch_size, seq_len, self.hidden_size, device=x0.device)

        for idx in range(self.num_quantizers):
            if idx < mask_layer:
                xt = xt + self.token_emb[idx](x0[:, :, idx])
            elif idx == mask_layer:
                mask = torch.bernoulli(torch.ones(batch_size, seq_len, device=x0.device) * mask_prob[..., None])
                mask_token_expanded = mask_token.view(1, 1, -1)  # (1, 1, hidden_size)
                xt = xt + mask.unsqueeze(-1) * mask_token_expanded + (1 - mask).unsqueeze(-1) * self.token_emb[idx](x0[:, :, idx])
            else:
                mask_token_expanded = mask_token.view(1, 1, -1)
                xt = xt + mask.unsqueeze(-1) * mask_token_expanded

        # Condition
        cond = self.cond_emb(cond_code)
        mask_layer_cond = self.layer_emb(mask_layer).unsqueeze(1)
        cond = cond + mask_layer_cond

        # Estimate
        embeds = self.diff_estimator(xt, t, x_mask, cond)

        logits = self.to_logits[mask_layer.item()](embeds)

        return logits, mask_layer, mask, x0

    @torch.no_grad()
    def synthesize(
        self,
        cond_code: torch.Tensor,
        prompt: Optional[torch.Tensor] = None,
        n_timesteps: Optional[List[int]] = None,
        temperature: float = 1.5,
        filter_threshold: float = 0.98,
    ) -> torch.Tensor:
        """
        Synthesize acoustic tokens from semantic tokens.

        Args:
            cond_code: Semantic tokens (B, T).
            prompt: Optional prompt acoustic tokens (B, prompt_len, num_q).
            n_timesteps: Diffusion steps per layer.
            temperature: Sampling temperature.
            filter_threshold: Top-k filter threshold.

        Returns:
            Acoustic tokens (B, T, num_quantizers).
        """
        if n_timesteps is None:
            n_timesteps = [10] * self.num_quantizers

        batch_size, seq_len = cond_code.shape
        device = cond_code.device

        # Initialize output
        output = torch.zeros(batch_size, seq_len, self.num_quantizers, dtype=torch.long, device=device)

        # Process each layer
        for layer_idx in range(self.num_quantizers):
            steps = n_timesteps[layer_idx]

            seq = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

            # Layer conditioning
            mask_layer = torch.tensor([layer_idx], device=device)
            mask_layer_cond = self.layer_emb(mask_layer).unsqueeze(1)
            cond = self.cond_emb(cond_code) + mask_layer_cond

            # Accumulated embedding from previous layers
            cum = torch.zeros(batch_size, seq_len, self.hidden_size, device=device)
            for prev_idx in range(layer_idx):
                cum = cum + self.token_emb[prev_idx](output[:, :, prev_idx])

            h = 1.0 / steps

            for i in range(steps):
                t = (1.0 - i * h) * torch.ones(batch_size, device=device)

                token = self.token_emb[layer_idx](seq)
                cur = cum + mask[..., None] * self.mask_emb(torch.zeros(1, dtype=torch.long, device=device)) + (~mask[..., None]) * token

                embeds = self.diff_estimator(cur, t, None, cond)

                logits = self.to_logits[layer_idx](embeds)
                logits = top_k(logits, filter_threshold)

                sampled_ids = gumbel_sample(logits, temperature=max(temperature * (1.0 - i * h), 1e-3))
                seq = torch.where(mask, sampled_ids, seq)

                if i < steps - 1:
                    next_t = (1.0 - (i + 1) * h) * torch.ones(batch_size, device=device)
                    next_mask_num = int(self.mask_prob(next_t)[0].item() * seq_len)

                    if next_mask_num > 0:
                        scores = logits.softmax(dim=-1).gather(2, sampled_ids.unsqueeze(-1)).squeeze(-1)
                        scores = 1 - scores
                        scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)

                        mask_indices = scores.topk(min(next_mask_num, seq_len), dim=-1).indices
                        mask = torch.zeros_like(scores, dtype=torch.bool).scatter(1, mask_indices, True)
                        seq = seq.masked_fill(mask, 0)
                    else:
                        break

            output[:, :, layer_idx] = seq

        return output


# ============================================================================
# MaskGCT (Combined Model)
# ============================================================================

class MaskGCT(PreTrainedModel):
    """
    MaskGCT: Fully Non-Autoregressive Text-to-Speech.

    Two-stage TTS model:
    1. T2S: Text -> Semantic tokens
    2. S2A: Semantic tokens -> Acoustic tokens

    Example:
        ```python
        config = MaskGCTConfig()
        model = MaskGCT(config)

        # Synthesize
        semantic_tokens = model.t2s.synthesize(phone_ids, target_len=100)
        acoustic_tokens = model.s2a.synthesize(semantic_tokens)
        ```
    """

    config_class = MaskGCTConfig
    base_model_prefix = "maskgct"

    def __init__(self, config: MaskGCTConfig):
        super().__init__(config)
        self.config = config

        # T2S model
        t2s_config = MaskGCT_T2S_Config(
            hidden_size=config.t2s_hidden_size,
            num_layers=config.t2s_num_layers,
            num_heads=config.t2s_num_heads,
            cfg_scale=config.t2s_cfg_scale,
            cond_codebook_size=config.semantic_codebook_size,
            use_phone_cond=config.use_phone_cond,
            phone_vocab_size=config.phone_vocab_size,
        )
        self.t2s = MaskGCT_T2S(t2s_config)

        # S2A model
        s2a_config = MaskGCT_S2A_Config(
            num_quantizers=config.s2a_num_quantizers,
            hidden_size=config.s2a_hidden_size,
            num_layers=config.s2a_num_layers,
            num_heads=config.s2a_num_heads,
            codebook_size=config.acoustic_codebook_size,
            cond_codebook_size=config.semantic_codebook_size,
        )
        self.s2a = MaskGCT_S2A(s2a_config)

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def forward(
        self,
        semantic_tokens: torch.Tensor,
        acoustic_tokens: torch.Tensor,
        phone_ids: torch.Tensor,
        semantic_mask: Optional[torch.Tensor] = None,
        acoustic_mask: Optional[torch.Tensor] = None,
    ):
        """
        Training forward pass.

        Args:
            semantic_tokens: Target semantic tokens (B, T_sem).
            acoustic_tokens: Target acoustic tokens (B, T_ac, num_q).
            phone_ids: Phone token IDs (B, phone_len).
        """
        if semantic_mask is None:
            semantic_mask = torch.ones_like(semantic_tokens).float()
        if acoustic_mask is None:
            acoustic_mask = torch.ones(acoustic_tokens.shape[:2]).float().to(acoustic_tokens.device)

        # T2S forward
        t2s_logits, t2s_mask, t2s_target = self.t2s(semantic_tokens, semantic_mask, phone_ids)

        # S2A forward
        s2a_logits, s2a_layer, s2a_mask, s2a_target = self.s2a(acoustic_tokens, acoustic_mask, semantic_tokens)

        return {
            "t2s_logits": t2s_logits,
            "t2s_mask": t2s_mask,
            "t2s_target": t2s_target,
            "s2a_logits": s2a_logits,
            "s2a_layer": s2a_layer,
            "s2a_mask": s2a_mask,
            "s2a_target": s2a_target,
        }

    @torch.no_grad()
    def synthesize(
        self,
        phone_ids: torch.Tensor,
        target_len: int,
        n_timesteps_t2s: int = 40,
        n_timesteps_s2a: Optional[List[int]] = None,
        temperature: float = 0.9,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Synthesize speech from phone IDs.

        Args:
            phone_ids: Phone token IDs (B, phone_len).
            target_len: Target sequence length.
            n_timesteps_t2s: T2S diffusion steps.
            n_timesteps_s2a: S2A diffusion steps per layer.
            temperature: Sampling temperature.

        Returns:
            semantic_tokens: Semantic tokens (B, T).
            acoustic_tokens: Acoustic tokens (B, T, num_q).
        """
        # T2S: Phone -> Semantic
        semantic_tokens = self.t2s.synthesize(
            phone_ids,
            target_len,
            n_timesteps=n_timesteps_t2s,
            temperature=temperature,
        )

        # S2A: Semantic -> Acoustic
        acoustic_tokens = self.s2a.synthesize(
            semantic_tokens,
            n_timesteps=n_timesteps_s2a,
            temperature=temperature,
        )

        return semantic_tokens, acoustic_tokens