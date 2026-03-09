# /// script
# dependencies = [
#   "torch",
#   "torchvision",
#   "fire",
#   "tqdm",
#   "numpy",
#   "einops",
#   "moviepy",
#   "imageio",
#   "requests",
#   "accelerate",
#   "adam-atan2-pytorch",
#   "tensorboard",
#   "dreamer4"
# ]
# [tool.uv.sources]
# dreamer4 = { path = "." }
# ///

from math import sqrt
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.utils import save_image

import fire
from tqdm import tqdm
from einops import rearrange, repeat

from dreamer4.dreamer4 import VideoTokenizer

from torch import tensor

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# dataset

class MovingMNISTDataset(Dataset):
    def __init__(
        self,
        root = './data',
        num_frames = 10,
        image_size = 64,
        digit_size = 28,
        min_velocity = -2,
        max_velocity = 3,
        download = True
    ):
        super().__init__()
        from torchvision.datasets import MNIST
        self.mnist = MNIST(root = root, train = True, download = download)
        self.num_frames = num_frames
        self.image_size = image_size
        self.digit_size = digit_size
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        digit, _ = self.mnist[idx]
        digit = T.functional.to_tensor(digit).squeeze(0) # (28, 28)

        seq = torch.arange(self.num_frames)

        # position and velocity

        start_x, start_y = (np.random.randint(0, self.image_size - self.digit_size) for _ in range(2))
        vel_x, vel_y = (np.random.randint(self.min_velocity, self.max_velocity) for _ in range(2))

        x_positions = (start_x + vel_x * seq).clamp(0, self.image_size - self.digit_size)
        y_positions = (start_y + vel_y * seq).clamp(0, self.image_size - self.digit_size)

        # generate video tensor

        video = torch.zeros(self.num_frames, self.image_size, self.image_size)

        for f, x, y in zip(range(self.num_frames), x_positions, y_positions):
            dest = video[f, y:y+self.digit_size, x:x+self.digit_size]
            torch.maximum(dest, digit, out = dest)

        return repeat(video, 'f h w -> c f h w', c = 3)

from dreamer4.trainers import VideoTokenizerTrainer

# main

def main(
    num_frames = 10,
    num_train_steps = 100_000,
    batch_size = 32,
    grad_accum_every = 1,
    min_velocity = -2,
    max_velocity = 3,
    lr = 3e-4,
    dim = 64,
    dim_latent = 32,
    patch_size = 4,
    encoder_depth = 4,
    decoder_depth = 4,
    attn_dim_head = 32,
    attn_heads = 8,
    use_ema = True,
    ema_decay = 0.99,
    log_video_every = 50,
    log_dir = './logs_mnist_tokenizer',
    checkpoint_every = 5000,
    checkpoint_folder = './checkpoints_mnist_tokenizer'
):
    import shutil

    # clear old artifacts

    log_path = Path(log_dir)
    if log_path.exists():
        shutil.rmtree(log_path)

    log_path.mkdir(exist_ok = True, parents = True)

    # dataset

    dataset = MovingMNISTDataset(
        num_frames = num_frames,
        min_velocity = min_velocity,
        max_velocity = max_velocity
    )

    # tokenizer

    tokenizer = VideoTokenizer(
        dim = dim,
        dim_latent = dim_latent,
        patch_size = patch_size,
        channels = 3,
        encoder_depth = encoder_depth,
        decoder_depth = decoder_depth,
        attn_dim_head = attn_dim_head,
        attn_heads = attn_heads,
        encoder_add_decor_aux_loss = True,
        lpips_loss_weight = 0., # disable lpips
    )

    # trainer

    trainer = VideoTokenizerTrainer(
        model = tokenizer,
        dataset = dataset,
        optim_klass = torch.optim.AdamW,
        batch_size = batch_size,
        grad_accum_every = grad_accum_every,
        learning_rate = lr,
        num_train_steps = num_train_steps,
        use_tensorboard_logger = True,
        log_dir = log_dir,
        log_video = True,
        video_fps = 4,
        log_video_every = log_video_every,
        use_ema = use_ema,
        ema_decay = ema_decay,
        checkpoint_every = checkpoint_every,
        checkpoint_folder = checkpoint_folder,
    )

    trainer()

if __name__ == '__main__':
    fire.Fire(main)
