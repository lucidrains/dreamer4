import torch
from torch import tensor, Tensor
from torch.nn import Module

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from einops import rearrange

# helpers

def exists(v):
    return v is not None

class GymPixelEnv(Module):
    """Wraps any Gymnasium env to produce pixel observations for dreamer4.

    Handles both discrete and continuous action spaces.
    Supports vectorized (multiple parallel envs) via num_envs > 1.
    """

    def __init__(
        self,
        env_id = 'CartPole-v1',
        image_height = 60,
        image_width = 90,
        action_repeat = 1,
        num_envs = 1,
        seed = None,
        env_kwargs: dict = dict(),
    ):
        super().__init__()

        self.image_height = image_height
        self.image_width = image_width
        self.action_repeat = action_repeat
        self.num_envs = num_envs
        self.vectorized = num_envs > 1
        self.seed = seed

        self.envs = [gym.make(env_id, render_mode = 'rgb_array', **env_kwargs) for _ in range(num_envs)]
        self._needs_reset = [False] * num_envs

        action_space = self.envs[0].action_space

        if isinstance(action_space, spaces.Discrete):
            self.num_discrete_actions = (int(action_space.n),)
            self.num_continuous_actions = 0
            self.is_discrete = True
        elif isinstance(action_space, spaces.Box):
            self.num_discrete_actions = 0
            self.num_continuous_actions = int(np.prod(action_space.shape))
            self.is_discrete = False
        else:
            raise ValueError(f'Unsupported action space: {type(action_space)}')

        self.register_buffer('_dummy', tensor(0))

    @property
    def device(self):
        return self._dummy.device

    def _render_one(self, env):
        frame = env.render()
        frame = torch.from_numpy(frame).float() / 255.
        return rearrange(frame, 'h w c -> 1 c h w')

    def _resize(self, frames):
        # frames: (b, c, h, w)
        frames = torch.nn.functional.interpolate(frames, size = (self.image_height, self.image_width), mode = 'bilinear', align_corners = False)
        return frames.to(self.device)

    def _extract_discrete_action(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().reshape(-1)[0].item()
        elif isinstance(action, np.ndarray):
            action = action.reshape(-1)[0].item()
        elif isinstance(action, (list, tuple)):
            action = np.asarray(action).reshape(-1)[0].item()

        return int(action)

    def reset(self, seed = None):
        self._needs_reset = [False] * self.num_envs
        frames = []
        for i, env in enumerate(self.envs):
            env.reset(seed = (seed or self.seed or 0) + i if exists(seed) or exists(self.seed) else None)
            frames.append(self._render_one(env))

        frames = torch.cat(frames, dim = 0)  # (b, c, h, w)
        frames = self._resize(frames)

        if not self.vectorized:
            frames = frames.squeeze(0)  # (c, h, w)

        return frames

    def step(self, actions):
        discrete = continuous = None

        if self.is_discrete:
            if isinstance(actions, tuple):
                discrete, continuous = actions
            else:
                discrete = actions
        else:
            discrete, continuous = actions

        obs_list = []
        rewards = []
        terminateds = []
        truncateds = []

        for i, env in enumerate(self.envs):
            if self._needs_reset[i]:
                env.reset()
                self._needs_reset[i] = False

            if self.is_discrete:
                action_value = discrete[i] if self.vectorized else discrete
                action = self._extract_discrete_action(action_value)
            else:
                if self.vectorized:
                    action = continuous[i].detach().cpu().numpy().flatten()
                else:
                    action = continuous.detach().cpu().numpy().flatten()

            total_reward = 0.
            terminated = False
            truncated = False

            for _ in range(self.action_repeat):
                _, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break

            obs_list.append(self._render_one(env))
            rewards.append(total_reward)
            terminateds.append(terminated)
            truncateds.append(truncated)

            if terminated or truncated:
                self._needs_reset[i] = True

        obs = self._resize(torch.cat(obs_list, dim = 0))
        reward_t = tensor(rewards, dtype = torch.float32, device = self.device)
        term_t = tensor(terminateds, device = self.device)
        trunc_t = tensor(truncateds, device = self.device)

        if not self.vectorized:
            obs = obs.squeeze(0)

        return obs, reward_t, term_t, trunc_t

    def close(self):
        for env in self.envs:
            env.close()
