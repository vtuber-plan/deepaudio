import collections
import json
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from deepaudio.samplers.base.base_sampler import BatchSampler


class BaseTrainer(object):
    def __init__(self) -> None:
        pass