# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "gymnasium[classic_control]",
#     "memmap-replay-buffer",
#     "tqdm",
#     "x-mlps-pytorch",
#     "ninja",
#     "dreamer4",
#     "opencv-python-headless"
# ]
# [tool.uv.sources]
# dreamer4 = { path = "." }
# ///

import os
import shutil
import pygame
from collections import deque

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch import cat, stack
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical

import gymnasium as gym

from tqdm import tqdm
from einops import rearrange
from x_mlps_pytorch import Feedforwards
from accelerate import Accelerator

from dreamer4.dreamer4 import calc_gae

from memmap_replay_buffer import ReplayBuffer

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# pixel observation wrapper

class PixelFrameStack(gym.ObservationWrapper):
    def __init__(self, env, num_stack = 4):
        super().__init__(env)
        self.num_stack = num_stack

        self.frames = deque(maxlen = num_stack)
        self.observation_space = gym.spaces.Box(
            low = 0, high = 255,
            shape = (num_stack * 3, 64, 64),
            dtype = np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['state'] = obs
        frame = self._render_frame()
        for _ in range(self.num_stack):
            self.frames.append(frame)
        return self._stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['state'] = obs
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, obs):
        self.frames.append(self._render_frame())
        return self._stacked_obs()

    def _stacked_obs(self):
        return np.concatenate(list(self.frames), axis = 0)

    def _render_frame(self):
        frame = self.env.render()
        height, width, _ = frame.shape

        min_dim = min(height, width)
        start_h = (height - min_dim) // 2
        start_w = (width - min_dim) // 2

        cropped = frame[start_h:start_h + min_dim, start_w:start_w + min_dim]
        resized = cv2.resize(cropped, (64, 64), interpolation = cv2.INTER_LINEAR)

        return rearrange(resized, 'h w c -> c h w')

# convolutional encoder

class ConvEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x.float() / 255.)

# actor critic

class ActorCritic(nn.Module):
    def __init__(
        self,
        in_channels,
        num_actions,
        dim_hidden = 256,
        asymmetric_critic = False,
        dim_state = 4
    ):
        super().__init__()
        self.asymmetric_critic = asymmetric_critic
        self.encoder = ConvEncoder(in_channels)
        dim_encoded = 2048

        self.actor = Feedforwards(dim_hidden, depth = 2, dim_in = dim_encoded, dim_out = num_actions)
        
        dim_critic_in = dim_state if asymmetric_critic else dim_encoded
        self.critic = Feedforwards(dim_hidden, depth = 2, dim_in = dim_critic_in, dim_out = 1)

    def forward(self, obs, state = None, action = None):
        features = self.encoder(obs)
        logits = self.actor(features)
        dist = Categorical(logits = logits)

        action = default(action, dist.sample())
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        critic_in = state if self.asymmetric_critic else features
        value = rearrange(self.critic(critic_in), '... 1 -> ...')

        return action, log_prob, entropy, value

# main

def main(
    lr = 3e-4,
    total_timesteps = 512_000,
    num_envs = 4,
    rollout_steps = 128,
    update_epochs = 4,
    gamma = 0.99,
    gae_lambda = 0.95,
    ppo_eps_clip = 0.2,
    entropy_loss_weight = 0.01,
    max_grad_norm = 0.5,
    seed = 1,
    target_return = 100.0,
    save_path = './policy.pt',
    record_every = 500,
    recordings_dir = './recordings',
    data_episodes = 50,
    data_save_dir = './dataset',
    asymmetric_critic = True
):
    accelerator = Accelerator()
    device = accelerator.device
    log = accelerator.print

    batch_size = num_envs * rollout_steps
    minibatch_size = batch_size // 4
    num_updates = total_timesteps // batch_size

    # environments

    if os.path.exists(recordings_dir):
        shutil.rmtree(recordings_dir)

    os.makedirs(recordings_dir, exist_ok = True)

    def make_env(idx):
        def thunk():
            is_first = idx == 0
            env = gym.make('CartPole-v1', render_mode = 'rgb_array')
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = PixelFrameStack(env)
            if is_first:
                env = gym.wrappers.RecordVideo(env, recordings_dir, episode_trigger = lambda ep: ep % record_every == 0)
            return env
        return thunk

    envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(num_envs)])

    obs_shape = envs.single_observation_space.shape
    num_actions = envs.single_action_space.n

    # agent

    agent = ActorCritic(obs_shape[0], num_actions, asymmetric_critic = asymmetric_critic, dim_state = 4)
    optimizer = Adam(agent.parameters(), lr = lr, eps = 1e-5)
    agent, optimizer = accelerator.prepare(agent, optimizer)

    # rollout state

    env_obs, info = envs.reset(seed = seed)
    env_state = info['state']
    env_obs = torch.tensor(env_obs, dtype = torch.uint8, device = device)
    env_state = torch.tensor(env_state, dtype = torch.float32, device = device)
    env_done = torch.zeros(num_envs, dtype = torch.float32, device = device)

    recent_returns = deque(maxlen = 20)

    # training

    pbar = tqdm(range(num_updates), desc = 'ppo updates')

    for _ in pbar:

        # collect rollouts

        all_obs, all_states, all_actions, all_log_probs = [], [], [], []
        all_rewards, all_dones, all_values = [], [], []

        for _ in range(rollout_steps):
            with torch.no_grad():
                action, log_prob, _, value = agent(env_obs, state = env_state)

            all_obs.append(env_obs)
            all_states.append(env_state)
            all_actions.append(action)
            all_log_probs.append(log_prob)
            all_values.append(value)

            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())

            reward_tensor = torch.tensor(reward, dtype = torch.float32, device = device)
            done_tensor = torch.tensor(terminated, dtype = torch.float32, device = device)

            all_rewards.append(reward_tensor)
            all_dones.append(done_tensor)

            env_obs = torch.tensor(next_obs, dtype = torch.uint8, device = device)
            env_state = torch.tensor(infos['state'], dtype = torch.float32, device = device)
            env_done = done_tensor

            if 'episode' in infos:
                for i, finished in enumerate(infos['_episode']):
                    if finished:
                        recent_returns.append(infos['episode']['r'][i])

        if len(recent_returns) > 0:
            avg_ret = np.mean(recent_returns)
            pbar.set_postfix(avg_return = f'{avg_ret:.1f}')

            if avg_ret >= target_return:
                log(f'\nTarget average return of {target_return} reached! Stopping training.')
                break

        # gae

        all_obs = stack(all_obs)
        all_states = stack(all_states)
        all_actions = stack(all_actions)
        all_log_probs = stack(all_log_probs)
        all_rewards = stack(all_rewards)
        all_dones = stack(all_dones)
        all_values = stack(all_values)

        with torch.no_grad():
            next_features = agent.encoder(env_obs)
            critic_in = env_state if agent.asymmetric_critic else next_features
            next_value = rearrange(agent.critic(critic_in), '... 1 -> ...')

            padded_rewards = cat([all_rewards, rearrange(next_value, '... -> 1 ...')])
            padded_values = cat([all_values, rearrange(next_value, '... -> 1 ...')])
            padded_masks = 1. - cat([all_dones, rearrange(torch.ones_like(env_done), '... -> 1 ...')])

            returns = calc_gae(
                rewards         = rearrange(padded_rewards, 't e -> e t'),
                values          = rearrange(padded_values, 't e -> e t'),
                masks           = rearrange(padded_masks, 't e -> e t'),
                gamma           = gamma,
                lam             = gae_lambda,
                use_accelerated = False
            )

            returns = returns[:, :-1]
            advantages = returns - rearrange(all_values, 't e -> e t')

        # flatten for minibatch sampling

        flat_obs        = rearrange(all_obs, 't e ... -> (t e) ...')
        flat_states     = rearrange(all_states, 't e ... -> (t e) ...')
        flat_actions    = rearrange(all_actions, 't e -> (t e)')
        flat_log_probs  = rearrange(all_log_probs, 't e -> (t e)')
        flat_advantages = rearrange(advantages, 'e t -> (t e)')
        flat_returns    = rearrange(returns, 'e t -> (t e)')

        # ppo update

        for _ in range(update_epochs):
            indices = torch.randperm(batch_size, device = device)

            for start in range(0, batch_size, minibatch_size):
                indices_batch = indices[start:start + minibatch_size]

                obs           = flat_obs[indices_batch]
                states        = flat_states[indices_batch]
                actions       = flat_actions[indices_batch]
                old_log_probs = flat_log_probs[indices_batch]
                advantage     = flat_advantages[indices_batch]
                returns       = flat_returns[indices_batch]

                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                _, new_log_prob, entropy, new_value = agent(obs, state = states, action = actions)

                ratio = (new_log_prob - old_log_probs).exp()

                policy_loss = -torch.min(
                    advantage * ratio,
                    advantage * ratio.clamp(1 - ppo_eps_clip, 1 + ppo_eps_clip)
                ).mean()

                value_loss = F.mse_loss(new_value, returns)
                loss = policy_loss - entropy_loss_weight * entropy.mean() + value_loss

                optimizer.zero_grad()
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

    envs.close()
    torch.save(agent.state_dict(), save_path)

    # bootstrap dataset collection

    log('\nGenerating bootstrap dataset...')

    if os.path.exists(data_save_dir):
        shutil.rmtree(data_save_dir)

    dataset_buffer = ReplayBuffer(
        folder = data_save_dir,
        max_episodes = data_episodes,
        max_timesteps = 505,
        fields = dict(
            obs    = ('uint8', (3, 64, 64)),
            action = 'int',
            reward = 'float',
            done   = 'bool'
        )
    )

    env = gym.make('CartPole-v1', render_mode = 'rgb_array')
    env = PixelFrameStack(env)
    agent.eval()

    ep_returns = []
    pbar = tqdm(range(data_episodes), desc = 'collecting data')

    for _ in pbar:
        obs, info = env.reset()
        state_t = rearrange(torch.tensor(info['state'], dtype = torch.float32, device = device), '... -> 1 ...')
        ep_return = 0.
        is_done = False

        while not is_done:
            with torch.no_grad():
                obs_t = rearrange(torch.tensor(obs, dtype = torch.uint8, device = device), '... -> 1 ...')
                
                action, _, _, _ = agent(obs_t, state = state_t)

            action = action.item()

            image_obs = env.frames[-1]

            dataset_buffer.store(
                obs    = image_obs,
                action = action,
                reward = 0.,
                done   = False
            )

            obs, reward, terminated, truncated, info = env.step(action)
            state_t = rearrange(torch.tensor(info['state'], dtype = torch.float32, device = device), '... -> 1 ...')
            is_done = terminated | truncated
            ep_return += reward

            if is_done:
                dataset_buffer.store(
                    obs    = env.frames[-1],
                    action = 0,
                    reward = reward,
                    done   = True
                )

        dataset_buffer.advance_episode(batch_size = 1)
        ep_returns.append(ep_return)
        pbar.set_postfix(avg_return = f'{np.mean(ep_returns):.1f}')

    env.close()

    log(f'\n{data_episodes} episodes for policy with average cumulative reward of {np.mean(ep_returns):.1f} saved to folder {data_save_dir}')

if __name__ == '__main__':
    main()
