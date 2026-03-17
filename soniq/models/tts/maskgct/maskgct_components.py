# coding=utf-8
"""MaskGCT model components."""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple
import math


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal position embeddings for diffusion timestep.

    Args:
        dim: Embedding dimension
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Timestep (batch,) or scalar

        Returns:
            Position embedding (batch, dim) or (dim,)
        """
        if x.ndim == 0:
            x = x.unsqueeze(0)
        
        sin_inp = x.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        emb = torch.cat([sin_inp.sin(), sin_inp.cos()], dim=-1)
        return emb


class LlamaAdaptiveRMSNorm(nn.Module):
    """
    Adaptive RMSNorm conditioned on embeddings.

    Similar to LayerNorm but with RMS normalization and
    adaptive scaling from condition embeddings.

    Args:
        hidden_size: Hidden dimension size
        cond_dim: Condition embedding dimension
        eps: Epsilon for numerical stability
    """

    def __init__(self, hidden_size: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.eps = eps
        
        # Project condition to scale weights
        self.to_weight = nn.Sequential(
            nn.Linear(cond_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        # Initialize to identity transform
        nn.init.zeros_(self.to_weight[0].bias)
        nn.init.zeros_(self.to_weight[2].bias)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        cond_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Input (batch, seq_len, hidden_size)
            cond_embedding: Condition (batch, cond_dim)

        Returns:
            Normalized output (batch, seq_len, hidden_size)
        """
        # RMSNorm
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states.to(hidden_states.dtype)
        
        # Get adaptive weights
        weight = self.to_weight(cond_embedding).unsqueeze(1)
        
        return hidden_states * weight


class FeedForward(nn.Module):
    """
    Feedforward network with SwiGLU activation.

    Args:
        dim: Input dimension
        intermediate_dim: Intermediate dimension
        dropout: Dropout probability
    """

    def __init__(self, dim: int, intermediate_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, dim, bias=False)
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        out = self.down_proj(gate * up)
        return self.dropout(out)


class MaskGCTTransformerBlock(nn.Module):
    """
    Transformer block for MaskGCT with adaptive normalization.

    Args:
        hidden_size: Hidden dimension size
        num_attention_heads: Number of attention heads
        intermediate_size: FFN intermediate size
        cond_dim: Condition dimension for adaptive norm
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        cond_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        
        # Self-attention
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Adaptive RMSNorm for attention
        self.attn_norm = LlamaAdaptiveRMSNorm(hidden_size, cond_dim)
        
        # Feedforward
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout)
        
        # Adaptive RMSNorm for FFN
        self.ffn_norm = LlamaAdaptiveRMSNorm(hidden_size, cond_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_embedding: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Input (batch, seq_len, hidden_size)
            cond_embedding: Condition (batch, cond_dim)
            attention_mask: Optional attention mask

        Returns:
            Output (batch, seq_len, hidden_size)
        """
        # Self-attention with adaptive norm
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states, cond_embedding)
        
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention (non-causal, bidirectional)
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        hidden_states = residual + attn_output
        
        # FFN with adaptive norm
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states, cond_embedding)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class MaskGCTBackbone(nn.Module):
    """
    Transformer backbone for MaskGCT.

    Args:
        hidden_size: Hidden dimension size
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: FFN intermediate size
        cond_dim: Condition dimension
        dropout: Dropout probability
        max_seq_len: Maximum sequence length
    """

    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        cond_dim: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        
        # Timestep embedding
        self.time_embed = SinusoidalPosEmb(hidden_size)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            MaskGCTTransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                cond_dim=cond_dim,
                dropout=dropout,
            )
            for _ in range(num_hidden_layers)
        ])
        
        self.norm = LlamaAdaptiveRMSNorm(hidden_size, cond_dim)
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        cond_embedding: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Input (batch, seq_len, hidden_size)
            timestep: Diffusion timestep (batch,)
            cond_embedding: Condition (batch, cond_dim)
            attention_mask: Optional attention mask

        Returns:
            Output (batch, seq_len, hidden_size)
        """
        # Time embedding
        time_emb = self.time_embed(timestep)
        time_emb = self.time_proj(time_emb).unsqueeze(1)
        
        # Add position embeddings
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states + self.pos_embed[:, :seq_len, :]
        
        # Add time embedding
        hidden_states = hidden_states + time_emb
        
        # Transformer blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states, cond_embedding, attention_mask)
        
        hidden_states = self.norm(hidden_states, cond_embedding)
        
        return hidden_states


class GumbelSampler(nn.Module):
    """
    Gumbel-softmax sampler for masked generative modeling.

    Args:
        temperature: Sampling temperature
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        logits: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Sample using Gumbel-softmax trick.

        Args:
            logits: Input logits (batch, seq_len, vocab_size)
            temperature: Optional override temperature

        Returns:
            Sampled indices (batch, seq_len)
        """
        if temperature is None:
            temperature = self.temperature
            
        # Gumbel noise
        u = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(u + 1e-6) + 1e-6)
        
        # Sample
        sampled = (logits + gumbel) / temperature
        return sampled.argmax(dim=-1)


def get_mask_schedule(
    step: int,
    total_steps: int,
    mask_prob_min: float = 0.6,
    mask_prob_max: float = 1.0,
) -> float:
    """
    Compute mask probability for diffusion step.

    Uses sin schedule: mask_prob(t) = sin(t * pi/2)

    Args:
        step: Current step (0 to total_steps)
        total_steps: Total diffusion steps
        mask_prob_min: Minimum mask probability
        mask_prob_max: Maximum mask probability

    Returns:
        Mask probability for this step
    """
    t = step / total_steps
    mask_prob = math.sin(t * math.pi / 2)
    return mask_prob_min + (mask_prob_max - mask_prob_min) * mask_prob
