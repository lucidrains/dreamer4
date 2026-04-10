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
#   "gymnasium[classic-control]",
#   "dreamer4"
# ]
# [tool.uv.sources]
# dreamer4 = { path = "." }
# ///

from pathlib import Path
import fire
import torch
from adam_atan2_pytorch import MuonAdamAtan2

from dataset_gym_rollouts import GymRolloutDataset
from dreamer4.dreamer4 import VideoTokenizer, exists
from dreamer4.trainers import VideoTokenizerTrainer

def main(
    env_id = 'CartPole-v1',
    num_rollouts = 5000,
    num_frames = 5,
    image_height = 60,
    image_width = 90,
    action_repeat = 1,
    checkpoint_path = None,
    num_train_steps = 20_000,
    batch_size = 64,
    grad_accum_every = 1,
    lr = 3e-4,
    dim = 128,
    dim_latent = 32,
    patch_size = 4,
    num_latents = 32,
    encoder_depth = 4,
    decoder_depth = 4,
    time_block_every = 4,
    attn_dim_head = 32,
    attn_heads = 8,
    use_ema = True,
    ema_decay = 0.99,
    log_video_every = 50,
    log_dir = None,
    checkpoint_every = 1000,
    checkpoint_folder = None,
    use_loss_normalization = False,
):
    env_name = env_id.split('-')[0].lower()

    if log_dir is None:
        log_dir = f'./logs_tokenizer_{env_name}'

    if checkpoint_folder is None:
        checkpoint_folder = f'{log_dir}/checkpoints'

    import shutil

    log_path = Path(log_dir)
    ckpt_folder_path = Path(checkpoint_folder)
    latest_checkpoint = None

    if exists(checkpoint_path):
        latest_checkpoint = Path(checkpoint_path)
    elif ckpt_folder_path.exists():
        checkpoints = list(ckpt_folder_path.glob('tokenizer-*.pt'))
        checkpoints = [ckpt for ckpt in checkpoints if 'ema' not in ckpt.name]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key = lambda p: int(p.stem.split('-')[1]))

    if log_path.exists() and not latest_checkpoint:
        shutil.rmtree(log_path)

    log_path.mkdir(exist_ok = True, parents = True)

    # collect rollouts

    print(f'Collecting {num_rollouts} rollouts from {env_id}...')

    dataset = GymRolloutDataset(
        env_id = env_id,
        num_rollouts = num_rollouts,
        num_frames = num_frames,
        image_height = image_height,
        image_width = image_width,
        action_repeat = action_repeat,
    )

    print(f'Collected {len(dataset)} rollouts')

    # tokenizer

    tokenizer = VideoTokenizer(
        dim = dim,
        dim_latent = dim_latent,
        patch_size = patch_size,
        num_latent_tokens = num_latents,
        channels = 3,
        image_height = image_height,
        image_width = image_width,
        encoder_depth = encoder_depth,
        decoder_depth = decoder_depth,
        time_block_every = time_block_every,
        attn_dim_head = attn_dim_head,
        attn_heads = attn_heads,
        use_loss_normalization = use_loss_normalization,
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

    if exists(latest_checkpoint):
        trainer.load(latest_checkpoint)

    trainer()

if __name__ == '__main__':
    fire.Fire(main)
