# /// script
# dependencies = [
#   "gymnasium",
#   "stable-baselines3",
#   "wandb",
#   "dreamer4"
# ]
# [tool.uv.sources]
# dreamer4 = { path = "." }
# ///

import shutil
from pathlib import Path

import torch
import numpy as np
from einops import rearrange
import imageio

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

import wandb
import fire

from memmap_replay_buffer import ReplayBuffer

from dreamer4.env import RecordToReplayBufferEnvWrapper, RecordToFolderEnvWrapper
from dreamer4.web_env import SnakeEnv

# helpers

def exists(v):
    return v is not None

# classes

class GymWrapper(gym.Env):
    def __init__(self, env):
        self.env = env

        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env

        self.base_env = base_env
        self.grid_size = base_env.grid_size
        self.render_cell_size = base_env.render_cell_size

        self.action_space = spaces.Discrete(env.action_space)
        sz = self.grid_size * self.render_cell_size
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, sz, sz), dtype=np.uint8)

    def _get_obs(self, obs_dict):
        img = obs_dict['image']
        return rearrange(img, 'h w c -> c h w')

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        obs_dict = self.env.reset()
        return self._get_obs(obs_dict), {}

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs_dict, reward, terminated, truncated, info = out
            return self._get_obs(obs_dict), float(reward), bool(terminated), bool(truncated), info

        obs_dict, reward, done, info = out
        return self._get_obs(obs_dict), float(reward), bool(done), False, info

class PrintReturnsCallback(BaseCallback):
    def __init__(
        self,
        video_dir,
        target_apples = 5.0,
        log_every = 50,
        verbose = 0
    ):
        super().__init__(verbose)
        self.target_apples = target_apples

        self.episode_returns = []
        self.episode_apples = []
        self.episode_lengths = []

        self.current_return = 0.
        self.current_apples = 0.
        self.current_length = 0

        self.episodes = 0
        self.log_every = log_every
        self.video_dir = Path(video_dir)

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]

        self.current_return += reward
        self.current_length += 1

        if reward > 0.0:
            self.current_apples += 1

        if done:
            self.episodes += 1
            self.episode_returns.append(self.current_return)
            self.episode_apples.append(self.current_apples)
            self.episode_lengths.append(self.current_length)

            if self.current_apples >= self.target_apples:
                print(f"Reached target of {self.target_apples} apples. Stopping training!")
                return False

            if self.episodes % self.log_every == 0:
                avg_return = np.mean(self.episode_returns[-self.log_every:])
                avg_apples = np.mean(self.episode_apples[-self.log_every:])
                avg_length = np.mean(self.episode_lengths[-self.log_every:])

                print(f"PPO Episode {self.episodes}: Return = {avg_return:.2f}, Apples = {avg_apples:.2f}, Length = {avg_length:.2f}")

                log_dict = {
                    "train/return": avg_return,
                    "train/apples": avg_apples,
                    "train/length": avg_length,
                    "train/episode": self.episodes
                }

                videos = list(self.video_dir.glob('*.mp4'))
                if len(videos) > 0:
                    latest_video = max(videos, key=lambda p: p.stat().st_mtime)
                    try:
                        log_dict["train/video"] = wandb.Video(str(latest_video), fps=4, format="mp4")
                    except Exception:
                        pass

                wandb.log(log_dict)

            self.current_return = 0.
            self.current_apples = 0.
            self.current_length = 0

        return True

class CustomCnnExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )

        c, h, w = observation_space.shape
        n_flatten = 32 * h * w

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, features_dim),
            torch.nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3 automatically normalizes images to [0, 1] floats
        x = self.cnn(observations)
        return self.linear(x)

# main

def main(
    project_name: str = "snake-ppo-agent",
    buffer_folder: str = "./snake_buffer_ppo",
    video_folder: str = "./snake_videos_ppo",
    inspect_replay_buffer: bool = True,
    grid_size: int = 8,
    max_steps: int = 40,
    collision_penalty: float = -10.0,
    apple_reward: float = 5.0,
    aliveness_penalty: float = -0.01,
    ppo_updates_before_collect: int = 500,
    replay_buffer_size: int = 1000,
    target_apples: float = 5.0,
    max_timesteps: int = 41,
    learning_rate: float = 1e-3,
    ent_coef: float = 0.01,
    n_steps: int = 256,
    batch_size: int = 64,
    n_epochs: int = 4,
    log_every: int = 50,
):
    wandb.init(project = project_name)

    test_dir = Path(buffer_folder)
    video_dir = Path(video_folder)

    print(f"Training policy until {target_apples} apples, then populating replay buffer at {test_dir} with {replay_buffer_size} trajectories")

    shutil.rmtree(test_dir, ignore_errors = True)
    test_dir.mkdir(exist_ok = True, parents = True)

    shutil.rmtree(video_dir, ignore_errors = True)
    video_dir.mkdir(exist_ok = True, parents = True)

    env = SnakeEnv(
        grid_size = grid_size,
        max_steps = max_steps,
        collision_penalty = collision_penalty,
        apple_reward = apple_reward,
        aliveness_penalty = aliveness_penalty
    )

    env = RecordToFolderEnvWrapper(
        env,
        folder = str(video_dir),
        fps = 4,
        clear_on_start = True
    )

    gym_env = GymWrapper(env)
    vec_env = DummyVecEnv([lambda: gym_env])

    print("Phase 1: Training PPO...")

    # In SB3, CnnPolicy creates the features extractor and then directly attaches
    # the main policy network (action_net) and value network (value_net) linear heads
    # to the features output. By setting net_arch = dict(pi=[], vf=[]), we ensure
    # no hidden layers are placed between the CNN and the output heads.

    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose = 0,
        device = "cpu",
        learning_rate = learning_rate,
        ent_coef = ent_coef,
        n_steps = n_steps,
        batch_size = batch_size,
        n_epochs = n_epochs,
        policy_kwargs = dict(
            features_extractor_class = CustomCnnExtractor,
            features_extractor_kwargs = dict(features_dim = 128),
            net_arch = dict(pi=[], vf=[])
        )
    )

    callback = PrintReturnsCallback(video_dir = video_dir, target_apples = target_apples, log_every = log_every)

    total_ppo_timesteps = ppo_updates_before_collect * n_steps * vec_env.num_envs
    model.learn(total_timesteps = total_ppo_timesteps, callback = callback)

    print(f"\nPhase 2: Collecting {replay_buffer_size} episodes...")

    buffer = ReplayBuffer(
        folder = str(test_dir),
        max_episodes = replay_buffer_size,
        max_timesteps = max_timesteps,
        fields = dict(
            rewards = 'float',
            terminated = 'bool',
            video = ('uint8', (3, grid_size * 2, grid_size * 2)),
            discrete_actions = 'int'
        ),
        meta_fields = dict(
            cum_reward = 'float',
            total_apples = 'float'
        )
    )

    env_with_buffer = RecordToReplayBufferEnvWrapper(
        env,
        replay_buffer = buffer,
        rewards = True,
        terminated = True
    )

    gym_env_collect = GymWrapper(env_with_buffer)

    episodes_collected = 0
    cum_reward = 0.
    total_apples = 0.

    obs, _ = gym_env_collect.reset()

    while episodes_collected < replay_buffer_size:
        action, _ = model.predict(obs, deterministic = False)
        obs, reward, terminated, truncated, _ = gym_env_collect.step(action)

        cum_reward += float(reward)
        if reward > 0.0:
            total_apples += 1.

        if terminated or truncated:
            buffer.store_meta_datapoint(episodes_collected, 'cum_reward', cum_reward)
            buffer.store_meta_datapoint(episodes_collected, 'total_apples', total_apples)

            episodes_collected += 1
            if episodes_collected % log_every == 0:
                print(f"Collected {episodes_collected} / {replay_buffer_size} episodes")

            obs, _ = gym_env_collect.reset()
            cum_reward = 0.
            total_apples = 0.

    print("Collection complete.")

    if inspect_replay_buffer:
        print("\nStarting replay buffer visualizer...")
        from dreamer4.cli import inspect_replay_buffer as start_visualizer
        start_visualizer(buffer_folder, port = 8081)

if __name__ == "__main__":
    fire.Fire(main)
