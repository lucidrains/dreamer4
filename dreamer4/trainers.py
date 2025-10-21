import torch
from torch.nn import Module

from accelerate import Accelerator

from adam_atan2_pytorch import MuonAdamAtan2

from dreamer4.dreamer4 import (
    VideoTokenizer,
    DynamicsModel
)

class VideoTokenizerTrainer(Module):
    def __init__(
        self,
        model: VideoTokenizer
    ):
        super().__init__()
        raise NotImplementedError
