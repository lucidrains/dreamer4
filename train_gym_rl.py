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
#   "gymnasium[classic-control]",
#   "dreamer4"
# ]
# [tool.uv.sources]
# dreamer4 = { path = "." }
# ///

from pathlib import Path
import fire
import torch

from gym_pixel_env import GymPixelEnv
from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel, exists
from dreamer4.trainers import DreamerTrainer

def main(
    env_id = 'CartPole-v1',
    tokenizer_checkpoint_path: str = None,
    reward_encoding: str = 'hl_gauss',
    num_episodes = 5000,
    image_height = 60,
    image_width = 90,
    action_repeat = 1,
    num_envs = 4,
    dim = 512,
    depth = 6,
    attn_dim_head = 64,
    attn_heads = 8,
    num_bins = 255,
    lr = 3e-4,
    dream_timesteps = 16,
    env_max_timesteps = 500,
    wm_collect_frames = 2048,
    wm_max_frames_per_batch = 128,
    dream_train_steps_per_collect = 12,
    wm_only_steps = 100,
    predict_terminals = True,
    log_dir: str = None,
    checkpoint_path: str = None,
    checkpoint_every = 100,
):
    assert reward_encoding in ('hl_gauss', 'two_hot')

    env_name = env_id.split('-')[0].lower()

    if tokenizer_checkpoint_path is None:
        tokenizer_checkpoint_path = f'./logs_tokenizer_{env_name}/checkpoints'

    if log_dir is None:
        # auto-increment run version
        import glob
        existing = glob.glob(f'./runs/{env_name}_{reward_encoding}_v*')
        version = max([int(p.rstrip('/').split('_v')[-1]) for p in existing] + [0]) + 1
        log_dir = f'./runs/{env_name}_{reward_encoding}_v{version}'

    checkpoint_folder = f'{log_dir}/checkpoints'

    log_path = Path(log_dir)
    log_path.mkdir(exist_ok = True, parents = True)

    # create environment

    env = GymPixelEnv(
        env_id = env_id,
        image_height = image_height,
        image_width = image_width,
        action_repeat = action_repeat,
        num_envs = num_envs,
    ).cuda()

    # load frozen tokenizer

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
    print(f"Loading tokenizer from: {tok_checkpoint_path}")

    tokenizer = VideoTokenizer.init_and_load(str(tok_checkpoint_path), strict = False)
    tokenizer.eval().requires_grad_(False)

    # create world model

    model = DynamicsWorldModel(
        video_tokenizer = tokenizer,
        dim_latent = tokenizer.dim_latent,
        dim = dim,
        depth = depth,
        attn_dim_head = attn_dim_head,
        attn_heads = attn_heads,
        num_discrete_actions = env.num_discrete_actions,
        num_continuous_actions = env.num_continuous_actions,
        reward_encoding = reward_encoding,
        reward_encoder_kwargs = dict(
            num_bins = num_bins,
            reward_range = (-20., 20.),
        ),
        predict_terminals = predict_terminals,
    )

    # load checkpoint if provided

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location = 'cpu', weights_only = False)
        missing, unexpected = model.load_state_dict(ckpt['model'], strict = False)
        start_step = ckpt.get('step', 0)
        print(f"Loaded checkpoint from {checkpoint_path} (step {start_step})")
        if missing:
            print(f"  New params (not in checkpoint): {missing}")
    else:
        start_step = 0

    # create trainer

    trainer = DreamerTrainer(
        model,
        learning_rate = lr,
        start_step = start_step,
        dream_timesteps = dream_timesteps,
        env_max_timesteps = env_max_timesteps,
        wm_collect_frames = wm_collect_frames,
        wm_max_frames_per_batch = wm_max_frames_per_batch,
        dream_train_steps_per_collect = dream_train_steps_per_collect,
        wm_only_steps = wm_only_steps,
        use_tensorboard_logger = True,
        log_dir = log_dir,
        project_name = f'dreamer4_{env_name}_{reward_encoding}',
        checkpoint_every = checkpoint_every,
        checkpoint_folder = checkpoint_folder,
    )

    # train

    print(f'Environment: {env_id}')
    print(f'Reward encoding: {reward_encoding}')
    print(f'Discrete actions: {env.num_discrete_actions}')
    print(f'Continuous actions: {env.num_continuous_actions}')

    trainer(env, num_episodes = num_episodes, env_is_vectorized = num_envs > 1)

    env.close()

if __name__ == '__main__':
    fire.Fire(main)
