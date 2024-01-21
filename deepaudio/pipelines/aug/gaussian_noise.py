import torch
import torchaudio
import torchaudio.transforms as T

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import random

import numpy as np

class GaussianNoise(torch.nn.Module):
    def __init__(self, min_snr=0.0001, max_snr=0.01):
        """
        :param min_snr: Minimum signal-to-noise ratio
        :param max_snr: Maximum signal-to-noise ratio
        """
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr

    def forward(self, audio):
        std = torch.std(audio)
        noise_std = random.uniform(self.min_snr * std, self.max_snr * std)

        norm_dist = torch.distributions.normal.Normal(0.0, noise_std)
        noise = norm_dist.rsample(audio.shape).type(audio.dtype).to(audio.device)

        return audio + noise
