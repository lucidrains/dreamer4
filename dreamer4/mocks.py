from __future__ import annotations
from random import choice

import torch
from torch import tensor, randn, randint
from torch.nn import Module

from einops import repeat

# mock env

class MockEnv(Module):
    def __init__(
        self,
        image_shape,
        reward_range = (-100., 100.),
        batch_size = 1,
        vectorized = False
    ):
        super().__init__()
        self.image_shape = image_shape
        self.reward_range = reward_range

        self.batch_size = batch_size
        self.vectorized = vectorized
        self.register_buffer('_step', tensor(0))

    def get_random_state(self):
        return randn(3, *self.image_shape)

    def reset(
        self,
        seed = None
    ):
        self._step.zero_()
        return self.get_random_state()

    def step(
        self,
        actions,
    ):
        state = self.get_random_state()
        reward = randint(*self.reward_range, ()).float()

        if self.vectorized:
            state = repeat(state, '... -> b ...', b = self.batch_size)
            reward = repeat(rewardstate, ' -> b', b = self.batch_size)

        return state, reward
