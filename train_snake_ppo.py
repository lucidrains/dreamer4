# /// script
# dependencies = [
#   "gymnasium",
#   "dreamer4",
#   "fire",
#   "joblib",
#   "x-ppo>=0.1.0",
#   "torch-einops-utils>=0.2.5",
# ]
# [tool.uv.sources]
# dreamer4 = { path = "." }
# ///

import shutil
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from torch_einops_utils import tree_map_tensor_to_device
from x_ppo import calc_gae, clipped_value_loss_fn, ppo_actor_loss

import gymnasium as gym
from gymnasium import spaces
from joblib import Parallel, delayed
import fire

from memmap_replay_buffer import ReplayBuffer
from dreamer4.env import RecordToReplayBufferEnvWrapper
from dreamer4.web_env import SnakeEnv

# helpers

def image_shape(env):
    size = env.grid_size * env.render_cell_size
    return (3, size, size)

# env wrapper

class GymWrapper(gym.Env):
    def __init__(self, env):
        self.env = env

        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env

        self.action_space = spaces.Discrete(base_env.action_space)
        self.observation_space = spaces.Box(0, 255, shape = image_shape(base_env), dtype = np.uint8)

    def _get_obs(self, obs):
        image = obs['image'] if isinstance(obs, dict) else obs
        return np.clip(image, 0, 255).astype(np.uint8)

    def reset(self, *, seed = None, options = None):
        super().reset(seed = seed)
        obs = self.env.reset(seed = seed, options = options)
        obs = obs[0] if isinstance(obs, tuple) else obs
        return self._get_obs(obs), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._get_obs(obs), float(reward), bool(terminated), bool(truncated), info

# actor critic

class ActorCritic(nn.Module):
    def __init__(
        self,
        num_actions,
        image_shape,
        dim = 128
    ):
        super().__init__()
        input_dim = int(np.prod(image_shape))

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, dim),
            nn.SiLU()
        )

        self.policy = nn.Linear(dim, num_actions)
        self.value = nn.Linear(dim, 1)

    def features(self, obs):
        return self.encoder(obs.float() / 255.)

    def forward(self, obs):
        features = self.features(obs)
        return self.policy(features), self.value(features).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic = False):
        logits, value = self(obs)
        dist = Categorical(logits = logits)
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    @torch.no_grad()
    def value_only(self, obs):
        return self.value(self.features(obs)).squeeze(-1)

# ppo

def collect_rollout(model, envs, obs, n_steps):
    num_envs = len(envs)
    obs_buf = torch.empty((n_steps, num_envs, *obs.shape[1:]), dtype = torch.uint8)
    next_obs_buf = torch.empty_like(obs_buf)

    actions = torch.empty((n_steps, num_envs), dtype = torch.long)
    log_probs = torch.empty((n_steps, num_envs))
    values = torch.empty((n_steps, num_envs))
    rewards = torch.empty((n_steps, num_envs))
    terminated = torch.empty((n_steps, num_envs), dtype = torch.bool)
    truncated = torch.empty_like(terminated)

    episode_return = torch.zeros(num_envs)
    episode_apples = torch.zeros(num_envs)
    episode_length = torch.zeros(num_envs, dtype = torch.long)

    returns, apples, lengths = [], [], []

    for t in range(n_steps):
        obs_buf[t] = obs

        action, log_prob, value = model.act(obs)
        actions[t] = action.cpu()
        log_probs[t] = log_prob.cpu()
        values[t] = value.cpu()

        next_obs = torch.empty_like(obs)

        for env_index, env in enumerate(envs):
            obs_np, reward, terminated_np, truncated_np, _ = env.step(action[env_index].item())
            obs_t = torch.from_numpy(obs_np)

            next_obs_buf[t, env_index] = obs_t
            next_obs[env_index] = obs_t

            rewards[t, env_index] = reward
            terminated[t, env_index] = terminated_np
            truncated[t, env_index] = truncated_np

            episode_return[env_index] += reward
            episode_apples[env_index] += float(reward > 0.)
            episode_length[env_index] += 1

            if terminated_np or truncated_np:
                returns.append(episode_return[env_index].item())
                apples.append(episode_apples[env_index].item())
                lengths.append(episode_length[env_index].item())

                episode_return[env_index] = 0.
                episode_apples[env_index] = 0.
                episode_length[env_index] = 0

                obs_np, _ = env.reset()
                next_obs[env_index] = torch.from_numpy(obs_np)

        obs = next_obs

    flat_next_obs = next_obs_buf.reshape(-1, *obs.shape[1:])
    next_values = model.value_only(flat_next_obs).reshape(n_steps, num_envs)

    return dict(
        obs = obs_buf,
        actions = actions,
        log_probs = log_probs,
        values = values,
        rewards = rewards,
        terminated = terminated,
        truncated = truncated,
        next_values = next_values,
        current_obs = obs,
        returns = returns,
        apples = apples,
        lengths = lengths
    )

def train_ppo(
    model,
    envs,
    num_updates = 200,
    n_steps = 128,
    epochs = 4,
    batch_size = 256,
    gamma = 0.99,
    lam = 0.95,
    eps_clip = 0.2,
    value_clip = 0.2,
    entropy_coef = 0.01,
    value_coef = 0.5,
    max_grad_norm = 0.5,
    learning_rate = 3e-4,
    target_apples = 5.,
    log_every = 10
):
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr = learning_rate)

    obs = torch.stack([torch.from_numpy(env.reset()[0]) for env in envs])
    recent_apples = deque(maxlen = 20)

    for update in range(num_updates):
        rollout = collect_rollout(model, envs, obs, n_steps)
        obs = rollout['current_obs']

        rollout = tree_map_tensor_to_device(rollout, device)

        obs_buf = rollout['obs']
        actions = rollout['actions']
        old_log_probs = rollout['log_probs']
        old_values = rollout['values']
        rewards = rollout['rewards']
        terminated = rollout['terminated']
        truncated = rollout['truncated']
        next_values = rollout['next_values']

        gae = calc_gae(
            rewards.movedim(0, 1),
            old_values.movedim(0, 1),
            terminated = terminated.movedim(0, 1),
            truncated = truncated.movedim(0, 1),
            next_values = next_values.movedim(0, 1),
            gamma = gamma,
            lam = lam,
            return_advantages = True
        )

        returns = gae.returns.movedim(0, 1)
        advantages = gae.advantages.movedim(0, 1)

        total_steps = n_steps * len(envs)
        flat_obs = obs_buf.reshape(total_steps, *obs_buf.shape[2:])
        flat_actions = actions.reshape(total_steps)
        flat_log_probs = old_log_probs.reshape(total_steps)
        flat_values = old_values.reshape(total_steps)
        flat_returns = returns.reshape(total_steps)
        flat_advantages = advantages.reshape(total_steps)

        model.train()

        for _ in range(epochs):
            indices = torch.randperm(total_steps, device = device)

            for start in range(0, total_steps, batch_size):
                batch_indices = indices[start:start + batch_size]

                logits, values = model(flat_obs[batch_indices])
                dist = Categorical(logits = logits)

                log_probs = dist.log_prob(flat_actions[batch_indices])
                entropy = dist.entropy().mean()

                policy_loss = ppo_actor_loss(
                    log_probs,
                    flat_log_probs[batch_indices],
                    flat_advantages[batch_indices],
                    eps_clip = eps_clip,
                    normalize_advantages = True
                ).mean()

                value_loss = clipped_value_loss_fn(
                    values,
                    flat_values[batch_indices],
                    flat_returns[batch_indices],
                    clip = value_clip
                )

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        if len(rollout['apples']) > 0:
            recent_apples.extend(rollout['apples'])

        if update % log_every == 0 and len(rollout['returns']) > 0:
            avg_return = np.mean(rollout['returns'])
            avg_apples = np.mean(rollout['apples'])
            avg_length = np.mean(rollout['lengths'])

            print(f'update {update}: return {avg_return:.2f} | apples {avg_apples:.2f} | length {avg_length:.2f}', flush = True)

        if len(recent_apples) >= 20 and np.mean(recent_apples) >= target_apples:
            print(f'reached {target_apples} apples, stopping ppo', flush = True)
            break

    return model

# replay collection

def collect_episodes(
    collector_id,
    num_episodes,
    folder,
    max_timesteps,
    grid_size,
    max_steps,
    collision_penalty,
    apple_reward,
    aliveness_penalty,
    policy,
    log_every
):
    print(f'[collector {collector_id}] collecting {num_episodes} episodes')

    folder = Path(folder)
    shutil.rmtree(folder, ignore_errors = True)
    folder.mkdir(exist_ok = True, parents = True)

    env = SnakeEnv(
        grid_size = grid_size,
        max_steps = max_steps,
        collision_penalty = collision_penalty,
        apple_reward = apple_reward,
        aliveness_penalty = aliveness_penalty
    )

    buffer = ReplayBuffer(
        folder = str(folder),
        max_episodes = num_episodes,
        max_timesteps = max_timesteps,
        fields = dict(
            rewards = 'float',
            terminated = 'bool',
            video = ('uint8', image_shape(env)),
            discrete_actions = 'int'
        ),
        meta_fields = dict(
            cum_reward = 'float',
            total_apples = 'float'
        )
    )

    env = RecordToReplayBufferEnvWrapper(env, replay_buffer = buffer, rewards = True, terminated = True)
    env = GymWrapper(env)

    policy.eval()
    obs, _ = env.reset()

    episodes = 0
    cum_reward = 0.
    total_apples = 0.

    while episodes < num_episodes:
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        action, _, _ = policy.act(obs_t)

        obs, reward, terminated, truncated, _ = env.step(action.item())

        cum_reward += reward
        total_apples += float(reward > 0.)

        if terminated or truncated:
            buffer.store_meta_datapoint(episodes, 'cum_reward', cum_reward)
            buffer.store_meta_datapoint(episodes, 'total_apples', total_apples)

            episodes += 1
            cum_reward = 0.
            total_apples = 0.

            if episodes % log_every == 0:
                print(f'[collector {collector_id}] {episodes} / {num_episodes}')

            if episodes < num_episodes:
                obs, _ = env.reset()

    buffer.flush()
    print(f'[collector {collector_id}] done')

# main

def main(
    buffer_folder = None,
    inspect_replay_buffer = True,
    grid_size = 4,
    max_steps = 20,
    collision_penalty = -10.0,
    apple_reward = 5.0,
    aliveness_penalty = -0.01,
    num_updates = 200,
    num_envs = 8,
    n_steps = 128,
    epochs = 4,
    batch_size = 256,
    gamma = 0.99,
    lam = 0.95,
    eps_clip = 0.2,
    value_clip = 0.2,
    value_coef = 0.5,
    max_grad_norm = 0.5,
    replay_buffer_size = 1000,
    num_trajectory_collectors = 4,
    target_apples = 4.0,
    learning_rate = 1e-3,
    entropy_coef = 0.02,
    log_every = 10,
    seed = 0
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    buffer_folder = buffer_folder or f'./snake_buffer_ppo_{grid_size}x{grid_size}_{max_steps}steps'

    envs = [
        GymWrapper(SnakeEnv(
            grid_size = grid_size,
            max_steps = max_steps,
            collision_penalty = collision_penalty,
            apple_reward = apple_reward,
            aliveness_penalty = aliveness_penalty
        ))
        for _ in range(num_envs)
    ]

    model = ActorCritic(num_actions = 4, image_shape = envs[0].observation_space.shape)
    model = train_ppo(
        model,
        envs,
        num_updates = num_updates,
        n_steps = n_steps,
        epochs = epochs,
        batch_size = batch_size,
        gamma = gamma,
        lam = lam,
        eps_clip = eps_clip,
        value_clip = value_clip,
        value_coef = value_coef,
        max_grad_norm = max_grad_norm,
        entropy_coef = entropy_coef,
        learning_rate = learning_rate,
        target_apples = target_apples,
        log_every = log_every
    )

    print(f'collecting {replay_buffer_size} episodes with {num_trajectory_collectors} collectors')

    quotient, remainder = divmod(replay_buffer_size, num_trajectory_collectors)
    episodes_per_collector = [quotient + int(i < remainder) for i in range(num_trajectory_collectors)]

    Parallel(n_jobs = num_trajectory_collectors)(
        delayed(collect_episodes)(
            collector_id = i,
            num_episodes = episodes,
            folder = f'{buffer_folder}.{i}',
            max_timesteps = max_steps + 1,
            grid_size = grid_size,
            max_steps = max_steps,
            collision_penalty = collision_penalty,
            apple_reward = apple_reward,
            aliveness_penalty = aliveness_penalty,
            policy = model,
            log_every = 50
        )
        for i, episodes in enumerate(episodes_per_collector) if episodes > 0
    )

    print('collection complete')

    if inspect_replay_buffer:
        from dreamer4.cli import inspect_replay_buffer as start_visualizer
        start_visualizer(f'{buffer_folder}.*', port = 9000)

if __name__ == '__main__':
    fire.Fire(main)