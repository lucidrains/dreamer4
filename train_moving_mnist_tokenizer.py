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
#   "wandb",
#   "dreamer4"
# ]
# [tool.uv.sources]
# dreamer4 = { path = "." }
# ///

from functools import partial
from math import sqrt
from pathlib import Path
import shutil

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

from dreamer4.trainers import VideoTokenizerTrainer, pixel_shift_aug

# main

def main(
    num_frames = 5,
    image_size = 32,
    digit_size = 14,
    checkpoint_path = None,
    num_train_steps = 100_000,
    num_latents = 32,
    batch_size = 16,
    encoder_moss_layers = (2,),
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
    checkpoint_folder = './logs_mnist_tokenizer/checkpoints',
    time_decorr_loss_weight = 4e-3,
    space_decorr_loss_weight = 4e-3,
    latent_ortho_loss_weight = 0.,
    use_loss_normalization = False,
    encoder_add_decorr_aux_loss = True,
    use_causal_conv3d = True,
    causal_conv3d_kernel_size = 3,
    lpips_loss_weight = 0.,
    decoder_flow_steps = 4,
    latent_ar_loss_weight = 0.,
    latent_ar_sigreg_loss_weight = 0.05,
    latent_ar_placement = 'encoder',
    latent_ar_sigreg_num_slices = 256,
    decoder_v_space_loss = True,
    time_attention_use_pope = False,
    space_attention_use_pope = False,
    restrict_latent_grads_to_noise = True,
    decoder_flow_times_beta_alpha = 1.,
    decoder_flow_times_beta_beta = 1.,
    latent_consistency_loss_weight = 0.,
    use_h_net = False,
    h_net_target_length = 2.,
    clear_runs = False,
    experiment_name = 'dreamer4',
    run_name = None,
    slot_attention_initted_latents = False,
    decoder_slot_attention_initted_spatial_tokens = False,
    slot_attention_inverted = True,
    encoder_slot_spatial_mix = True,
    decoder_slot_spatial_mix = False,
    latent_init_patch_size = None,
    use_pixel_shift_aug = False,
    max_pixel_shift = 3,
    aug_prob = 0.5,
    use_wandb = False,
    separate_flow_decoder = False,
    flow_decoder_train_prob = 0.5,
    encode_temporal_diff = False
):
    # clear old artifacts

    log_path = Path(log_dir)
    ckpt_folder_path = Path(checkpoint_folder)

    if clear_runs and log_path.exists():
        shutil.rmtree(log_path)

    latest_checkpoint = None

    if exists(checkpoint_path):
        latest_checkpoint = Path(checkpoint_path)
    elif ckpt_folder_path.exists():
        checkpoints = list(ckpt_folder_path.glob('tokenizer-*.pt'))
        checkpoints = [ckpt for ckpt in checkpoints if 'ema' not in ckpt.name]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('-')[1]))

    if log_path.exists() and not latest_checkpoint:
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

    # augmentation

    custom_aug_fn = partial(pixel_shift_aug, max_pixel_shift=max_pixel_shift) if use_pixel_shift_aug else None

    # tokenizer

    h_net_kwargs = dict(
        depth = 2,
        heads = attn_heads,
        dim_head = attn_dim_head,
        target_avg_token_length = h_net_target_length
    ) if use_h_net else None

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
        latent_ortho_loss_weight = latent_ortho_loss_weight,
        use_loss_normalization = use_loss_normalization,
        use_causal_conv3d = use_causal_conv3d,
        causal_conv3d_kernel_size = causal_conv3d_kernel_size,
        lpips_loss_weight = lpips_loss_weight,
        decoder_flow_steps = decoder_flow_steps,
        latent_ar_loss_weight = latent_ar_loss_weight,
        latent_ar_sigreg_loss_weight = latent_ar_sigreg_loss_weight,
        latent_ar_placement = latent_ar_placement,
        latent_ar_sigreg_loss_kwargs = dict(num_slices = latent_ar_sigreg_num_slices),
        decoder_v_space_loss = decoder_v_space_loss,
        time_attention_use_pope = time_attention_use_pope,
        space_attention_use_pope = space_attention_use_pope,
        latent_grad_only_at_noise = restrict_latent_grads_to_noise,
        decoder_flow_times_beta_alpha = decoder_flow_times_beta_alpha,
        decoder_flow_times_beta_beta = decoder_flow_times_beta_beta,
        encoder_moss_layers = encoder_moss_layers,
        latent_consistency_loss_weight = latent_consistency_loss_weight,
        h_net_layer = encoder_depth // 2 if use_h_net else None,
        h_net_kwargs = h_net_kwargs,
        slot_attention_initted_latents = slot_attention_initted_latents,
        decoder_slot_attention_initted_spatial_tokens = decoder_slot_attention_initted_spatial_tokens,
        slot_attention_inverted = slot_attention_inverted,
        encoder_slot_spatial_mix = encoder_slot_spatial_mix,
        decoder_slot_spatial_mix = decoder_slot_spatial_mix,
        latent_init_patch_size = latent_init_patch_size,
        has_aug_conditioning = use_pixel_shift_aug,
        separate_flow_decoder = separate_flow_decoder,
        flow_decoder_train_prob = flow_decoder_train_prob,
        encode_temporal_diff = encode_temporal_diff
    )

    # trainer

    trainer = VideoTokenizerTrainer(
        model = tokenizer,
        dataset = dataset,
        checkpoint_path = latest_checkpoint,
        optim_klass = MuonAdamAtan2,
        batch_size = batch_size,
        grad_accum_every = grad_accum_every,
        learning_rate = lr,
        num_train_steps = num_train_steps,
        use_tensorboard = not use_wandb,
        use_wandb = use_wandb,
        log_dir = log_dir,
        log_video = True,
        video_fps = 4,
        log_video_every = log_video_every,
        use_ema = use_ema,
        ema_decay = ema_decay,
        checkpoint_every = checkpoint_every,
        checkpoint_folder = checkpoint_folder,
        project_name = experiment_name,
        run_name = run_name,
        custom_aug_fn = custom_aug_fn,
        aug_prob = aug_prob
    )

    if exists(latest_checkpoint):
        trainer.load(latest_checkpoint)

    trainer()

if __name__ == '__main__':
    fire.Fire(main)
