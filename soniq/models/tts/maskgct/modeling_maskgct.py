# coding=utf-8
"""MaskGCT: Masked Generative Codec Transformer for TTS."""

import torch
from torch import nn
from typing import Dict, Any, Optional, Tuple

from transformers import PreTrainedModel
from soniq.models.tts.maskgct.configuration_maskgct import MaskGCTConfig
from soniq.models.tts.maskgct.modeling_maskgct_t2s import MaskGCT_T2S
from soniq.models.tts.maskgct.modeling_maskgct_s2a import MaskGCT_S2A


class MaskGCT(PreTrainedModel):
    """
    MaskGCT: Masked Generative Codec Transformer for Text-to-Speech.

    This is a fully non-autoregressive TTS model that uses a two-stage
    masked generative approach:
    
    1. Text-to-Semantic (T2S): Predicts semantic tokens from phone IDs
    2. Semantic-to-Acoustic (S2A): Predicts acoustic tokens from semantic tokens

    The model uses a mask-and-predict paradigm similar to masked language
    modeling, where tokens are iteratively refined through multiple steps.

    Example:
        ```python
        from soniq.models.tts.maskgct import MaskGCT, MaskGCTConfig
        from soniq.models.tts.maskgct.configuration_maskgct import MaskGCTT2SConfig, MaskGCTS2AConfig

        # Create config
        t2s_config = MaskGCTT2SConfig(
            hidden_size=1536,
            num_hidden_layers=16,
            codebook_size=8192,
        )
        s2a_config = MaskGCTS2AConfig(
            hidden_size=1024,
            num_hidden_layers=16,
            num_quantizers=12,
            codebook_size=1024,
        )
        config = MaskGCTConfig(t2s_config=t2s_config, s2a_config=s2a_config)

        # Create model
        model = MaskGCT(config)

        # Training
        phone_ids = torch.randint(0, 1024, (4, 50))
        semantic_tokens = torch.randint(0, 8192, (4, 100))
        acoustic_tokens = torch.randint(0, 1024, (4, 100, 12))
        
        t2s_output = model.t2s(phone_ids, semantic_tokens)
        s2a_output = model.s2a(semantic_tokens, acoustic_tokens)

        # Inference
        generated = model.inference(phone_ids)
        # generated['semantic_tokens']: (batch, seq_len)
        # generated['acoustic_tokens']: (batch, seq_len, num_quantizers)
        ```
    """

    config_class = MaskGCTConfig
    base_model_prefix = "maskgct"
    supports_gradient_checkpointing = True

    def __init__(self, config: MaskGCTConfig):
        super().__init__(config)
        self.config = config

        # Text-to-Semantic model
        self.t2s = MaskGCT_T2S(config.t2s_config)

        # Semantic-to-Acoustic model
        self.s2a = MaskGCT_S2A(config.s2a_config)

        self.post_init()

    def forward(
        self,
        phone_ids: torch.Tensor,
        semantic_tokens: torch.Tensor,
        acoustic_tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass for training.

        Args:
            phone_ids: Phone IDs (batch, phone_seq_len)
            semantic_tokens: Semantic tokens (batch, seq_len)
            acoustic_tokens: Optional acoustic tokens (batch, seq_len, num_quantizers)

        Returns:
            Dictionary with t2s_output and s2a_output
        """
        # T2S forward pass
        t2s_output = self.t2s(phone_ids, semantic_tokens, **kwargs)

        # S2A forward pass (if acoustic tokens provided)
        s2a_output = None
        if acoustic_tokens is not None:
            s2a_output = self.s2a(semantic_tokens, acoustic_tokens, **kwargs)

        return {
            "t2s_output": t2s_output,
            "s2a_output": s2a_output,
        }

    @torch.no_grad()
    def generate(
        self,
        phone_ids: torch.Tensor,
        num_t2s_steps: int = 20,
        num_s2a_steps: int = 20,
        temperature: float = 1.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate speech from phone IDs.

        Args:
            phone_ids: Phone IDs (batch, phone_seq_len)
            num_t2s_steps: Number of T2S refinement steps
            num_s2a_steps: Number of S2A refinement steps
            temperature: Sampling temperature
            **kwargs: Additional arguments

        Returns:
            Tuple of (semantic_tokens, acoustic_tokens)
        """
        self.eval()

        # T2S: Generate semantic tokens from phone IDs
        semantic_tokens = self.t2s.generate(
            phone_ids,
            num_steps=num_t2s_steps,
            temperature=temperature,
            **kwargs,
        )

        # S2A: Generate acoustic tokens from semantic tokens
        acoustic_tokens = self.s2a.generate(
            semantic_tokens,
            num_steps=num_s2a_steps,
            temperature=temperature,
            **kwargs,
        )

        return semantic_tokens, acoustic_tokens

    @torch.no_grad()
    def inference(
        self,
        phone_ids: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Full inference pipeline.

        Args:
            phone_ids: Phone IDs (batch, phone_seq_len)
            **kwargs: Additional arguments

        Returns:
            Dictionary with semantic_tokens and acoustic_tokens
        """
        self.eval()
        semantic_tokens, acoustic_tokens = self.generate(phone_ids, **kwargs)
        return {
            "semantic_tokens": semantic_tokens,
            "acoustic_tokens": acoustic_tokens,
        }

    @torch.no_grad()
    def generate_semantic(
        self,
        phone_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate only semantic tokens (T2S only).

        Args:
            phone_ids: Phone IDs (batch, phone_seq_len)
            **kwargs: Additional arguments

        Returns:
            Semantic tokens (batch, seq_len)
        """
        return self.t2s.generate(phone_ids, **kwargs)

    @torch.no_grad()
    def generate_acoustic(
        self,
        semantic_tokens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate only acoustic tokens (S2A only).

        Args:
            semantic_tokens: Semantic tokens (batch, seq_len)
            **kwargs: Additional arguments

        Returns:
            Acoustic tokens (batch, seq_len, num_quantizers)
        """
        return self.s2a.generate(semantic_tokens, **kwargs)
