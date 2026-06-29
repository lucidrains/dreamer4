from __future__ import annotations

import os
import fire
from pathlib import Path
from shutil import rmtree
from adam_atan2_pytorch import MuonAdamAtan2

from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel
from dreamer4.trainers import VideoTokenizerTrainer, VideoDataset, VideoTrajectoryDataset, BehaviorCloneTrainer
from dreamer4.env import DynamicsWorldModelWrapper
from dreamer4.web_env.server import WebEnvServer
from dreamer4.web_env import SnakeEnv

import webbrowser

def exists(val):
    return val is not None

def train_tokenizer(
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
    use_lpips_loss: bool = False,
    lpips_loss_weight: float = 0.2,
    encoder_add_decorr_aux_loss: bool = False,
    # logging and saving
    name: str = 'default-tokenizer',
    use_wandb: bool = True,
    log_video: bool = True,
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

    if not use_lpips_loss:
        lpips_loss_weight = 0.

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
        separate_flow_decoder = separate_flow_decoder,
        lpips_loss_weight = lpips_loss_weight,
        encoder_add_decorr_aux_loss = encoder_add_decorr_aux_loss
    )

    checkpoint_dir = Path(checkpoint_folder) / name
    log_directory = Path(log_dir) / name

    if force_restart:
        if checkpoint_dir.exists():
            rmtree(checkpoint_dir)
        if log_directory.exists():
            rmtree(log_directory)

    latest_checkpoint = checkpoint_dir / "tokenizer.pt"
    checkpoint_path_str = str(latest_checkpoint) if latest_checkpoint.exists() else None

    if exists(checkpoint_path_str):
        print(f"loading checkpoint from {checkpoint_path_str}")

    trainer = VideoTokenizerTrainer(
        model = tokenizer,
        dataset = dataset,
        checkpoint_path = checkpoint_path_str,
        optim_klass = MuonAdamAtan2,
        batch_size = batch_size,
        grad_accum_every = grad_accum_every,
        learning_rate = learning_rate,
        num_train_steps = num_train_steps,
        use_wandb = use_wandb,
        log_dir = f"{log_dir}/{name}",
        log_video = log_video,
        video_fps = video_fps,
        log_video_every = log_video_every,
        use_ema = use_ema,
        ema_decay = ema_decay,
        checkpoint_every = checkpoint_every,
        checkpoint_folder = f"{checkpoint_folder}/{name}",
        project_name = project_name,
        run_name = name,
    )

    trainer()

def train_dynamics(
    data: str,
    tokenizer_checkpoint: str,
    # dataset params
    max_num_frames: int = 16,
    image_size: int = 64,
    channels: int = 3,
    exts: tuple[str, ...] = ('gif', 'mp4'),
    # architecture params
    dim: int = 512,
    depth: int = 6,
    attn_dim_head: int = 64,
    attn_heads: int = 8,
    condition_on_actions: bool = False,
    num_continuous_actions: int = 6,
    num_discrete_actions: int = 0,
    # training params
    batch_size: int = 8,
    grad_accum_every: int = 1,
    num_train_steps: int = 100000,
    learning_rate: float = 3e-4,
    # logging and saving
    name: str = 'default',
    use_wandb: bool = True,
    log_video: bool = True,
    video_fps: int = 4,
    log_video_every: int = 100,
    checkpoint_every: int = 5000,
    checkpoint_folder: str = './checkpoints',
    log_dir: str = './logs',
    project_name: str = 'dreamer4',
):
    data_path = Path(data)
    assert data_path.exists() and data_path.is_dir(), f"{data} must be an existing directory"

    tokenizer_path = Path(tokenizer_checkpoint)
    assert tokenizer_path.exists(), f"Tokenizer checkpoint missing at {tokenizer_path}"

    suffix = '-action-conditioned-dynamics' if condition_on_actions else '-dynamics-only'
    run_name = f"{name}{suffix}"

    if condition_on_actions:
        dataset = VideoTrajectoryDataset(
            folder = data_path,
            image_size = image_size,
            channels = channels,
            max_num_frames = max_num_frames,
            exts = exts
        )
    else:
        dataset = VideoDataset(
            folder = data_path,
            image_size = image_size,
            channels = channels,
            max_num_frames = max_num_frames,
            exts = exts
        )

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = VideoTokenizer.init_and_load(tokenizer_path, strict = False)

    if condition_on_actions:
        sample = dataset[0]

        cont_action = sample.get('continuous_actions')
        action_dim = cont_action.shape[-1] if exists(cont_action) else 0
        assert action_dim == num_continuous_actions, f"Expected {num_continuous_actions} continuous actions, but found {action_dim} in dataset"

        disc_action = sample.get('discrete_actions')
        action_dim = disc_action.shape[-1] if exists(disc_action) else 0

    model = DynamicsWorldModel(
        video_tokenizer = tokenizer,
        dim_latent = tokenizer.dim_latent,
        dim = dim,
        depth = depth,
        attn_dim_head = attn_dim_head,
        attn_heads = attn_heads,
        num_continuous_actions = num_continuous_actions if condition_on_actions else 0,
        num_discrete_actions = num_discrete_actions if condition_on_actions else 0,
    )

    trainer = BehaviorCloneTrainer(
        model = model,
        dataset = dataset,
        batch_size = batch_size,
        learning_rate = learning_rate,
        num_train_steps = num_train_steps,
        use_tensorboard = not use_wandb,
        use_wandb = use_wandb,
        project_name = project_name,
        run_name = run_name,
        log_dir = f"{log_dir}/{run_name}",
        video_fps = video_fps,
        log_video_every = log_video_every,
        checkpoint_every = checkpoint_every,
        checkpoint_folder = f"{checkpoint_folder}/{run_name}",
        grad_accum_every = grad_accum_every,
    )

    trainer()

def serve_world_model(
    model_path: str | None = None,
    env_name: str = 'snake',
    port: int = 8000,
    num_generation_steps: int = 4
):
    ENVS = {
        'snake': lambda: SnakeEnv()
    }

    if not exists(model_path):
        assert env_name in ENVS, f"env {env_name} not found"

        print(f"serving ground truth {env_name} env")
        env = ENVS[env_name]()
    else:
        model_file = Path(model_path).resolve()
        assert model_file.exists(), f"World model not found at {model_file}"

        print(f"Loading world model from: {model_file}")
        world_model = DynamicsWorldModel.init_and_load(model_path, strict = False)
        env = DynamicsWorldModelWrapper(world_model, num_generation_steps = num_generation_steps)

    server = WebEnvServer(env, port=port)

    url = f"http://localhost:{port}"
    print(f"Opening browser to {url} ...")
    webbrowser.open(url)

    server.serve()

def main():
    fire.Fire({
        'train-video-tokenizer': train_tokenizer,
        'train-dynamics': train_dynamics,
        'serve-world-model': serve_world_model
    })

if __name__ == '__main__':
    main()
