from __future__ import annotations
from random import choice

import torch
from torch import tensor, empty, randn, randint
from torch.nn import Module

from einops import repeat

# helpers

def exists(v):
    return v is not None

# mock env

class MockEnv(Module):
    def __init__(
        self,
        image_shape,
        reward_range = (-100, 100),
        num_envs = 1,
        vectorized = False,
        terminate_after_step = None,
        rand_terminate_prob = 0.05
    ):
        super().__init__()
        self.image_shape = image_shape
        self.reward_range = reward_range

        self.num_envs = num_envs
        self.vectorized = vectorized
        assert not (vectorized and num_envs == 1)

        # mocking termination

        self.can_terminate = exists(terminate_after_step)
        self.terminate_after_step = terminate_after_step
        self.rand_terminate_prob = rand_terminate_prob

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

        if self.vectorized:
            discrete, continuous = actions
            assert discrete.shape[0] == self.num_envs, f'expected batch of actions for {self.num_envs} environments'

            state = repeat(state, '... -> b ...', b = self.num_envs)
            reward = repeat(reward, ' -> b', b = self.num_envs)

        out = (state, reward)

        if self.can_terminate:
            terminate = (
                (torch.rand((self.num_envs)) < self.rand_terminate_prob) &
                (self._step > self.terminate_after_step)
            )

            out = (*out, terminate)

        self._step.add_(1)

        return out
