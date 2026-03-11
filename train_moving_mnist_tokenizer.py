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
from adam_atan2_pytorch import MuonAdamAtan2
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

from dataset_moving_mnist import MovingMNISTDataset

from dreamer4.trainers import VideoTokenizerTrainer

# main

def main(
    num_frames = 5,
    image_size = 32,
    digit_size = 14,
    checkpoint_path = None,
    num_train_steps = 100_000,
    num_latents = 32,
    batch_size = 64,
    grad_accum_every = 1,
    min_velocity = -2,
    max_velocity = 3,
    lr = 3e-4,
    dim = 128,
    dim_latent = 32,
    patch_size = 4,
    encoder_depth = 4,
    decoder_depth = 4,
    time_block_every = 4,
    attn_dim_head = 32,
    attn_heads = 8,
    use_ema = True,
    ema_decay = 0.99,
    log_video_every = 50,
    log_dir = './logs_mnist_tokenizer',
    checkpoint_every = 5000,
    checkpoint_folder = './checkpoints_mnist_tokenizer',
    time_decorr_loss_weight = 4e-3,
    space_decorr_loss_weight = 4e-3,
    use_loss_normalization = False,
    encoder_add_decorr_aux_loss = True,
    use_causal_conv3d = True,
    causal_conv3d_kernel_size = 3,
    lpips_loss_weight = 0.
):
    import shutil

    # clear old artifacts

    log_path = Path(log_dir)
    if log_path.exists():
        shutil.rmtree(log_path)

    log_path.mkdir(exist_ok = True, parents = True)

    # dataset

    dataset = MovingMNISTDataset(
        image_size = image_size,
        digit_size = digit_size,
        num_frames = num_frames,
        min_velocity = min_velocity,
        max_velocity = max_velocity
    )

    # tokenizer

    tokenizer = VideoTokenizer(
        dim = dim,
        dim_latent = dim_latent,
        patch_size = patch_size,
        num_latent_tokens = num_latents,
        channels = 3,
        image_height = image_size,
        image_width = image_size,
        encoder_depth = encoder_depth,
        decoder_depth = decoder_depth,
        time_block_every = time_block_every,
        attn_dim_head = attn_dim_head,
        attn_heads = attn_heads,
        encoder_add_decorr_aux_loss = encoder_add_decorr_aux_loss,
        time_decorr_loss_weight = time_decorr_loss_weight,
        space_decorr_loss_weight = space_decorr_loss_weight,
        use_loss_normalization = use_loss_normalization,
        use_causal_conv3d = use_causal_conv3d,
        causal_conv3d_kernel_size = causal_conv3d_kernel_size,
        lpips_loss_weight = lpips_loss_weight,
    )

    # trainer

    trainer = VideoTokenizerTrainer(
        model = tokenizer,
        dataset = dataset,
        checkpoint_path = checkpoint_path,
        optim_klass = MuonAdamAtan2,
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

    if exists(checkpoint_path):
        trainer.load(checkpoint_path)

    trainer()

if __name__ == '__main__':
    fire.Fire(main)
