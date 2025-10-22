from __future__ import annotations
from random import choice

import torch
from torch import tensor, empty, randn, randint
from torch.nn import Module

from einops import repeat

# mock env

class MockEnv(Module):
    def __init__(
        self,
        image_shape,
        reward_range = (-100, 100),
        num_envs = 1,
        vectorized = False
    ):
        super().__init__()
        self.image_shape = image_shape
        self.reward_range = reward_range

        self.num_envs = num_envs
        self.vectorized = vectorized
        assert not (vectorized and num_envs == 1)

        self.register_buffer('_step', tensor(0))

    def get_random_state(self):
        return randn(3, *self.image_shape)

    def reset(
        self,
        seed = None
    ):
        self._step.zero_()
        state = self.get_random_state()

        if self.vectorized:
            state = repeat(state, '... -> b ...', b = self.num_envs)

        return state

    def step(
        self,
        actions,
    ):
        state = self.get_random_state()

        reward = empty(()).uniform_(*self.reward_range)

        if not self.vectorized:
            return state, reward

        discrete, continuous = actions
        assert discrete.shape[0] == self.num_envs, f'expected batch of actions for {self.num_envs} environments'

        state = repeat(state, '... -> b ...', b = self.num_envs)
        reward = repeat(reward, ' -> b', b = self.num_envs)

        return state, reward
