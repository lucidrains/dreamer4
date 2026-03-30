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
#   "torch-einops-utils",
#   "dreamer4"
# ]
# [tool.uv.sources]
# dreamer4 = { path = "." }
# ///

from pathlib import Path
import fire
import torch
from torch.utils.data import default_collate
from random import random
from einops import repeat

from dataset_moving_mnist import MovingMNISTDataset
from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel, exists
from dreamer4.trainers import BehaviorCloneTrainer, save_video_grid_as_gif

def exists(v):
    return v is not None

# dataset action collation

def random_action_collate(batch):
    use_continuous = random() > 0.5

    for item in batch:
        if 'continuous_actions' in item and 'discrete_actions' in item:
            key_to_drop = 'discrete_actions' if use_continuous else 'continuous_actions'
            item.pop(key_to_drop, None)

    return default_collate(batch)

# custom 3x3 grid sample generation

def custom_3x3_grid_sample(trainer, batch_data):
    device = trainer.device
    dataset = trainer.dataset

    velocities = torch.tensor([
        [-2., -2.], [ 0., -2.], [ 2., -2.],
        [-2.,  0.], [ 0.,  0.], [ 2.,  0.],
        [-2.,  2.], [ 0.,  2.], [ 2.,  2.]
    ], device = device)

    # perfectly branch out from the same static origin
    video = batch_data['video'][0:1, :, :1]
    prompt_video = repeat(video, '1 c f h w -> b c f h w', b = 9)

    bin_size = (dataset.max_velocity - dataset.min_velocity) / dataset.num_action_bins

    for is_continuous in (True, False):
        kwargs = dict()

        if is_continuous:
            actions = repeat(velocities, 'b d -> b t d', t = 15)
            kwargs['prompt_continuous_actions'] = actions
        else:
            discrete_velocities = velocities.clone()
            discrete_velocities.sub_(dataset.min_velocity).div_(bin_size)
            discrete_velocities = discrete_velocities.long().clamp_(max = dataset.num_action_bins - 1)
            actions = repeat(discrete_velocities, 'b d -> b t d', t = 15)
            kwargs['prompt_discrete_actions'] = actions

        generated_video = trainer.unwrap_model(trainer.model).generate(
            prompt = prompt_video,
            time_steps = 16,
            num_steps = 8,
            batch_size = 9,
            image_height = video.shape[-1],
            image_width = video.shape[-1],
            return_decoded_video = True,
            **kwargs
        )

        generated_video = generated_video.clamp(0., 1.)

        if not exists(trainer.results_folder):
            continue

        prefix = 'continuous' if is_continuous else 'discrete'
        gif_path = trainer.results_folder / f'sample-{prefix}-conditioned-{trainer.step.item()}.gif'
        save_video_grid_as_gif(generated_video, gif_path)

def main(
    tokenizer_checkpoint_path: str = './logs_mnist_tokenizer/checkpoints',
    checkpoint_path: str = None,
    num_frames = 5,
    image_size = 32,
    digit_size = 14,
    num_train_steps = 100_000,
    batch_size = 64,
    grad_accum_every = 1,
    lr = 3e-4,
    dim = 512,
    depth = 6,
    attn_dim_head = 64,
    attn_heads = 8,
    use_ema = True,
    video_fps = 4,
    log_video_every = 100,
    log_dir = './logs_mnist_dynamics',
    checkpoint_every = 5000,
    checkpoint_folder = './logs_mnist_dynamics/checkpoints',
    use_loss_normalization = False,
    multi_token_pred_len = 1,
    shortcut_loss_weight = 5e-2,
    sample_prompt_frames = None,
    sample_autoregressive_actions = False,
    condition_on_actions = False,
    num_action_bins = 5,
    latent_ar = False,
    latent_ar_action_conditioned = False,
    latent_ar_layer = 0,
    latent_ar_loss_weight = 0.,
    latent_ar_sigreg_loss_weight = 0.05,
    latent_ar_sigreg_num_slices = 256
):
    import shutil

    if sample_prompt_frames is None:
        sample_prompt_frames = 1 if condition_on_actions else 2

    # clear old artifacts

    log_path = Path(log_dir)
    ckpt_folder_path = Path(checkpoint_folder)
    latest_checkpoint = None

    if exists(checkpoint_path):
        latest_checkpoint = Path(checkpoint_path)
    elif ckpt_folder_path.exists():
        checkpoints = list(ckpt_folder_path.glob('dynamics-*.pt'))
        checkpoints = [ckpt for ckpt in checkpoints if 'ema' not in ckpt.name]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('-')[1]))

    if log_path.exists() and not latest_checkpoint:
        shutil.rmtree(log_path)
    log_path.mkdir(exist_ok = True, parents = True)

    # instantiate the dataset

    dataset = MovingMNISTDataset(
        num_frames = num_frames,
        image_size = image_size,
        digit_size = digit_size,
        condition_on_actions = condition_on_actions,
        action_type = 'both',
        num_action_bins = num_action_bins
    )

    # Load frozen tokenizer

    tok_checkpoint_path = Path(tokenizer_checkpoint_path)

    if tok_checkpoint_path.is_dir():
        ema_checkpoints = list(tok_checkpoint_path.glob('tokenizer-*-ema.pt'))
        assert ema_checkpoints, f"No EMA tokenizer checkpoints found in {tok_checkpoint_path}"

        def get_step(p):
            try:
                return int(p.stem.split('-')[1])
            except ValueError:
                return -1

        tok_checkpoint_path = max(ema_checkpoints, key = get_step)

    assert tok_checkpoint_path.exists(), f"Tokenizer checkpoint missing at {tok_checkpoint_path}"
    print(f"Loading Tokenizer from: {tok_checkpoint_path}")

    tokenizer = VideoTokenizer.init_and_load(str(tok_checkpoint_path), strict=False)
    tokenizer.eval().requires_grad_(False)

    # initialize world model

    model = DynamicsWorldModel(
        video_tokenizer = tokenizer,
        dim_latent = tokenizer.dim_latent,
        dim = dim,
        depth = depth,
        attn_dim_head = attn_dim_head,
        attn_heads = attn_heads,
        use_loss_normalization = use_loss_normalization,
        multi_token_pred_len = multi_token_pred_len,
        shortcut_loss_weight = shortcut_loss_weight,
        num_continuous_actions = 2 if condition_on_actions else 0,
        num_discrete_actions = (num_action_bins, num_action_bins) if condition_on_actions else 0,
        latent_ar = latent_ar,
        latent_ar_action_conditioned = latent_ar_action_conditioned,
        latent_ar_layer = latent_ar_layer,
        latent_ar_loss_weight = latent_ar_loss_weight,
        latent_ar_sigreg_loss_weight = latent_ar_sigreg_loss_weight,
        latent_ar_sigreg_loss_kwargs = dict(num_slices = latent_ar_sigreg_num_slices) if latent_ar else None
    )

    # initialize trainer

    trainer = BehaviorCloneTrainer(
        model = model,
        dataset = dataset,
        batch_size = batch_size,
        learning_rate = lr,
        num_train_steps = num_train_steps,
        log_dir = log_dir,
        video_fps = video_fps,
        log_video_every = log_video_every,
        checkpoint_every = checkpoint_every,
        checkpoint_folder = checkpoint_folder,
        use_ema = use_ema,
        use_tensorboard_logger = True,
        log_video = True,
        sample_prompt_frames = sample_prompt_frames,
        sample_sticky_action = condition_on_actions and not sample_autoregressive_actions,
        sample_autoregressive_actions = sample_autoregressive_actions,
        sample_filename_prefix = 'sample-baseline',
        grad_accum_every = grad_accum_every,
        collate_fn = random_action_collate if condition_on_actions else None,
        custom_sample_fn = custom_3x3_grid_sample if condition_on_actions else None
    )

    # Train dynamics model

    if exists(latest_checkpoint):
        trainer.load(latest_checkpoint)

    trainer()

if __name__ == '__main__':
    fire.Fire(main)
