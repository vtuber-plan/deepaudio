# coding=utf-8
"""
Flow modules for Soniq.
"""

import torch
from torch import nn
from .coupling import AffineCouplingLayer


class NormalizingFlow:
    """
    Normalizing Flow helper class.

    See coupling.py for implementation.
    """
    pass


class FlowSequence(nn.Module):
    """
    Sequence of flow transformations.

    Args:
        in_channels: Number of input channels.
        n_flows: Number of flow layers.
        hidden_channels: Hidden layer channels.
    """

    def __init__(
        self,
        in_channels: int,
        n_flows: int = 4,
        hidden_channels: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_flows = n_flows

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            mask = torch.zeros(1, in_channels, 1)
            mask[:, i % 2::2, :] = 1
            self.flows.append(
                AffineCouplingLayer(in_channels, hidden_channels, mask)
            )

    def forward(
        self,
        x: torch.Tensor,
        reverse: bool = False,
    ) -> tuple:
        log_det_total = 0

        if reverse:
            for flow in reversed(self.flows):
                x, log_det = flow(x, reverse=True)
                log_det_total += log_det
        else:
            for flow in self.flows:
                x, log_det = flow(x, reverse=False)
                log_det_total += log_det

        return x, log_det_total
