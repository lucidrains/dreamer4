from __future__ import annotations

import os
import fire
from pathlib import Path
from shutil import rmtree
from adam_atan2_pytorch import MuonAdamAtan2

from dreamer4.dreamer4 import VideoTokenizer
from dreamer4.trainers import VideoTokenizerTrainer, VideoDataset

def exists(val):
    return val is not None

def train(
    data: str,
    # dataset params
    max_num_frames: int = 16,
    image_size: int = 64,
    channels: int = 3,
    exts: tuple[str, ...] = ('gif', 'mp4'),
    # architecture params
    dim: int = 512,
    dim_latent: int = 64,
    patch_size: int = 8,
    depth: int = 4,
    flow_steps: int = 1,
    separate_flow_decoder: bool = False,
    # training params
    batch_size: int = 8,
    grad_accum_every: int = 8,
    num_train_steps: int = 100000,
    learning_rate: float = 1e-4,
    use_ema: bool = True,
    ema_decay: float = 0.999,
    # logging and saving
    name: str = 'default-tokenizer',
    use_wandb: bool = True,
    video_fps: int = 4,
    log_video_every: int = 100,
    checkpoint_every: int = 1000,
    checkpoint_folder: str = './checkpoints',
    log_dir: str = './logs',
    project_name: str = 'dreamer4',
    force_restart: bool = False
):
    data_path = Path(data)
    assert data_path.exists(), f"{data} does not exist"
    assert data_path.is_dir(), f"{data} must be a directory"

    dataset = VideoDataset(
        folder = str(data_path),
        image_size = image_size,
        channels = channels,
        max_num_frames = max_num_frames,
        exts = exts
    )

    tokenizer = VideoTokenizer(
        dim = dim,
        dim_latent = dim_latent,
        patch_size = patch_size,
        image_height = image_size,
        image_width = image_size,
        encoder_depth = depth,
        decoder_depth = depth,
        decoder_flow_steps = flow_steps,
        separate_flow_decoder = separate_flow_decoder
    )

    trainer = VideoTokenizerTrainer(
        model = tokenizer,
        dataset = dataset,
        optim_klass = MuonAdamAtan2,
        batch_size = batch_size,
        grad_accum_every = grad_accum_every,
        learning_rate = learning_rate,
        num_train_steps = num_train_steps,
        use_wandb = use_wandb,
        log_dir = f"{log_dir}/{name}",
        log_video = True,
        video_fps = video_fps,
        log_video_every = log_video_every,
        use_ema = use_ema,
        ema_decay = ema_decay,
        checkpoint_every = checkpoint_every,
        checkpoint_folder = f"{checkpoint_folder}/{name}",
        project_name = project_name,
        run_name = name,
    )

    checkpoint_dir = Path(checkpoint_folder) / name
    log_directory = Path(log_dir) / name

    if force_restart:
        if checkpoint_dir.exists():
            rmtree(checkpoint_dir)
        if log_directory.exists():
            rmtree(log_directory)

    latest_checkpoint = checkpoint_dir / "tokenizer.pt"
    if latest_checkpoint.exists():
        print(f"loading checkpoint from {latest_checkpoint}")
        trainer.load(str(latest_checkpoint))

    trainer()

def main():
    fire.Fire({
        'train-video-tokenizer': train
    })

if __name__ == '__main__':
    main()
