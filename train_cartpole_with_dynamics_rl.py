# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "gymnasium[classic_control]",
#     "tqdm",
#     "ninja",
#     "dreamer4",
#     "fire",
#     "wandb"
# ]
# [tool.uv.sources]
# dreamer4 = { path = "." }
# ///

import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from collections import deque, namedtuple
from functools import partial

import wandb
import numpy as np

import torch
from torch import nn, stack, tensor, is_tensor, zeros
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
pad_sequence = partial(pad_sequence, batch_first = True)

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from tqdm import tqdm
import gymnasium as gym
import fire

from accelerate import Accelerator

from dreamer4.dreamer4 import (
    DynamicsWorldModel,
    Experience,
    Actions,
    divisible_by,
    exists,
    default
)

# memory tuple

Memory = namedtuple('Memory', [
    'eps',
    'critic_state',
    'action',
    'log_prob',
    'action_logits',
    'reward',
    'is_boundary',
    'value',
    'done'
])

# env

def make_env(seed):
    env = gym.make('CartPole-v1')
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env

# agent

class TransformerPPOAgent(nn.Module):
    def __init__(
        self,
        use_asym_critic = False,
        agent_value_gradient_frac = 1.0,
        agent_policy_gradient_frac = 1.0
    ):
        super().__init__()

        self._state_to_latents = nn.Sequential(
            nn.Linear(4, 64 * 2),
            nn.LayerNorm(64 * 2),
            Rearrange('... (num_tokens dim) -> ... num_tokens dim', num_tokens = 2)
        )

        self.dynamics = DynamicsWorldModel(
            dim = 128,
            dim_latent = 64,
            num_latent_tokens = 2,
            num_spatial_tokens = 4,
            num_register_tokens = 1,
            num_discrete_actions = 2,
            use_time_rnn = False,
            transformer_kwargs = dict(
                use_attn_pool = False
            ),
            depth = 3,
            time_block_every = 3,
            policy_head_mlp_depth = 2,
            value_head_mlp_depth = 2,
            gae_discount_factor = 0.99,
            ppo_eps_clip = 0.2,
            agent_value_gradient_frac = agent_value_gradient_frac,
            agent_policy_gradient_frac = agent_policy_gradient_frac,
            normalize_advantages = True,
            use_loss_normalization = False,
            attn_heads = 4,
            attn_dim_head = 16,
            reward_encoder_kwargs = dict(reward_range = (-10., 10.)),
            dim_critic_state = 4 if use_asym_critic else None
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def num_discrete_actions(self):
        return self.dynamics.action_embedder.num_discrete_actions.sum().item()

    def state_to_latents(self, state):
        """Project raw state vector -> (batch, time, num_latent_tokens, dim_latent)."""
        latents = self._state_to_latents(state)
        if latents.ndim == 3:
            latents = rearrange(latents, 'batch tokens dim -> batch 1 tokens dim')
        return latents

    @torch.no_grad()
    def act(self, critic_state, past_action = None, time_cache = None):
        """Single-step inference for rollout collection."""
        device = self.device
        crit = rearrange(critic_state, '... -> 1 ...').to(device)
        latents = self.state_to_latents(crit)

        batch = latents.shape[0]
        step_size = torch.ones((batch,), device = device)

        discrete_actions = None
        if exists(past_action):
            discrete_actions = tensor([[[past_action]]], dtype = torch.long, device = device)

        _, (embeds, next_cache) = self.dynamics(
            latents = latents,
            signal_levels = self.dynamics.max_steps - 1,
            step_sizes = step_size,
            discrete_actions = discrete_actions,
            time_cache = time_cache,
            latent_is_noised = True,
            return_pred_only = True,
            return_intermediates = True
        )

        agent_embed = embeds.agent[:, -1]
        policy_embed = self.dynamics.policy_head(agent_embed)

        # value estimation with optional asymmetric critic state

        value_embed = agent_embed

        if exists(self.dynamics.critic_state_embedder):
            critic_embed = self.dynamics.critic_state_embedder(crit)
            value_embed = value_embed + rearrange(critic_embed, 'batch dim -> batch 1 dim')

        value = self.dynamics.reward_encoder.bins_to_scalar_value(
            self.dynamics.value_head(value_embed)
        )

        # sample action and compute log prob

        action, _ = self.dynamics.action_embedder.sample(policy_embed, pred_head_index = 0, squeeze = False)
        log_prob, _ = self.dynamics.action_embedder.log_probs(policy_embed, pred_head_index = 0, discrete_targets = action)

        # store action logits for PMPO KL divergence

        action_logits, _ = self.dynamics.action_embedder.unembed(policy_embed, pred_head_index = 0)

        # rearrange from single-batch inference to scalar outputs

        return (
            rearrange(action, '1 1 1 -> 1'),
            rearrange(log_prob, '1 1 1 -> 1'),
            rearrange(action_logits, '1 1 num_actions -> num_actions'),
            rearrange(value, '1 1 -> 1'),
            next_cache
        )

# experience processing

def build_experience(memories, episode_lens, is_episode_truncated, device):
    def stack_to_device(tensors):
        return stack(tensors).to(device)

    stacked = [tuple(map(stack_to_device, zip(*ep))) for ep in memories]

    (
        _episodes,
        critic_state_seq,
        actions_seq,
        log_probs_seq,
        action_logits_seq,
        rewards_seq,
        _is_boundaries_seq,
        values_seq,
        dones_seq
    ) = tuple(map(pad_sequence, zip(*stacked)))

    num_envs = critic_state_seq.shape[0]

    # ensure discrete actions and log probs have trailing action dim (envs, time, 1)

    if actions_seq.ndim == 2:
        actions_seq = rearrange(actions_seq, 'envs time -> envs time 1')

    if log_probs_seq.ndim == 2:
        log_probs_seq = rearrange(log_probs_seq, 'envs time -> envs time 1')

    # ensure values are 2D (envs, time)

    if values_seq.ndim == 3:
        values_seq = rearrange(values_seq, 'envs time 1 -> envs time')

    return Experience(
        latents = None,
        critic_state = critic_state_seq,
        rewards = rewards_seq,
        terminals = dones_seq,
        actions = Actions(actions_seq, None),
        log_probs = Actions(log_probs_seq, None),
        old_action_unembeds = (action_logits_seq, None),
        values = values_seq,
        step_size = repeat(tensor([1.], device = device), '1 -> envs', envs = num_envs),
        is_truncated = stack(tuple(is_episode_truncated)).to(device),
        lens = stack(tuple(episode_lens)).to(device)
    )

def slice_experience(exp, idx):
    old_action_unembeds = None

    if exists(exp.old_action_unembeds):
        discrete_logits, continuous_params = exp.old_action_unembeds
        old_action_unembeds = (
            discrete_logits[idx] if exists(discrete_logits) else None,
            continuous_params[idx] if exists(continuous_params) else None
        )

    return Experience(
        latents = None,
        critic_state = exp.critic_state[idx] if exists(exp.critic_state) else None,
        rewards = exp.rewards[idx],
        terminals = exp.terminals[idx],
        actions = Actions(exp.actions.discrete[idx], None),
        log_probs = Actions(exp.log_probs.discrete[idx], None),
        old_action_unembeds = old_action_unembeds,
        values = exp.values[idx],
        step_size = exp.step_size[idx],
        is_truncated = exp.is_truncated[idx],
        lens = exp.lens[idx] if exists(exp.lens) else None
    )

# training loop

def main(
    batch_size = 8,
    grad_accum_every = 1,
    update_episodes = 64,
    update_epochs = 4,
    num_episodes = 5000,
    max_timesteps = 500,
    learning_rate = 3e-4,
    max_grad_norm = 0.5,
    target_return = 70.0,
    use_asym_critic = True,
    max_policy_updates = 250,
    agent_value_gradient_frac = 0.1,
    agent_policy_gradient_frac = 0.1,
    seed = 42,
    use_wandb = False,
    use_pmpo = False
):
    torch.manual_seed(seed)
    assert divisible_by(update_episodes, batch_size)

    if use_wandb:
        wandb.init(project = 'dreamer4-cartpole')

    accelerator = Accelerator(gradient_accumulation_steps = grad_accum_every)
    device = accelerator.device

    def log(msg):
        accelerator.print(msg)

    env = make_env(seed)

    agent = TransformerPPOAgent(
        use_asym_critic = use_asym_critic,
        agent_value_gradient_frac = agent_value_gradient_frac,
        agent_policy_gradient_frac = agent_policy_gradient_frac
    ).to(device)

    optimizer = AdamW(agent.parameters(), lr = learning_rate)
    agent, optimizer = accelerator.prepare(agent, optimizer)

    # rollout state

    recent_returns = deque(maxlen = 20)
    memories = deque(maxlen = update_episodes)
    episode_lens = deque(maxlen = update_episodes)
    is_episode_truncated = deque(maxlen = update_episodes)

    pbar = tqdm(range(num_episodes), desc = 'episodes')
    agent.eval()
    num_policy_updates = 0

    for eps in pbar:
        one_ep = deque([])
        eps_tensor = tensor(eps)

        obs, info = env.reset(seed = seed)
        critic_state = tensor(obs, dtype = torch.float32, device = 'cpu')

        time_cache = None
        past_action = None

        for timestep in range(max_timesteps):
            action, log_prob, action_logits, value, time_cache = agent.act(critic_state, past_action, time_cache)

            discrete_item = int(action.item())
            next_obs, reward, terminated, truncated, infos = env.step(discrete_item)
            next_critic_state = tensor(next_obs, dtype = torch.float32, device = 'cpu')

            done = terminated or truncated

            one_ep.append(Memory(
                eps_tensor,
                critic_state,
                action.cpu(),
                log_prob.cpu(),
                action_logits.cpu(),
                tensor(reward, dtype = torch.float32),
                tensor(False, dtype = torch.bool),
                value.cpu(),
                tensor(terminated, dtype = torch.bool)
            ))

            past_action = discrete_item
            critic_state = next_critic_state

            # bootstrap value for truncated episodes

            if done and not terminated:
                _, _, _, bootstrap_value, _ = agent.act(critic_state, past_action, time_cache)

                one_ep.append(one_ep[-1]._replace(
                    critic_state = critic_state,
                    eps = tensor(-1),
                    action_logits = zeros(agent.num_discrete_actions),
                    is_boundary = tensor(True),
                    value = bootstrap_value.cpu(),
                    reward = tensor(0.),
                    done = tensor(False)
                ))

            if done:
                if exists(infos.get('episode')):
                    ep_ret = infos['episode']['r']
                    ep_ret = ep_ret.item() if is_tensor(ep_ret) else float(ep_ret)
                    recent_returns.append(ep_ret)
                break

        episode_lens.append(tensor(len(one_ep)))
        is_episode_truncated.append(tensor(truncated))
        memories.append(one_ep)

        avg_ret = np.mean(recent_returns) if len(recent_returns) > 0 else 0.0

        if avg_ret >= target_return:
            log(f'\n✅ Target average return of {target_return} reached! (avg={avg_ret:.1f}) Stopping training.')
            break

        loss_metrics = dict(avg_return = f'{avg_ret:.1f}')

        # PPO update

        if divisible_by(eps + 1, update_episodes):
            agent.train()

            exp = build_experience(memories, episode_lens, is_episode_truncated, device)
            total_envs = exp.critic_state.shape[0]

            epoch_policy_loss = 0.
            epoch_value_loss = 0.

            for _ in range(update_epochs):
                batches = torch.randperm(total_envs, device = device).split(batch_size)

                for i, batch_idx in enumerate(batches):
                    micro = slice_experience(exp, batch_idx)
                    micro.latents = agent.state_to_latents(micro.critic_state)

                    policy_loss, value_loss = agent.dynamics.learn_from_experience(
                        experience = micro,
                        only_learn_policy_value_heads = False,
                        use_pmpo = use_pmpo
                    )

                    total_loss = (policy_loss + value_loss) / grad_accum_every
                    accelerator.backward(total_loss)

                    is_last_batch = (i + 1) == len(batches)

                    if divisible_by(i + 1, grad_accum_every) or is_last_batch:
                        accelerator.clip_grad_norm_(agent.parameters(), max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    epoch_policy_loss += policy_loss.item() / len(batches)
                    epoch_value_loss += value_loss.item() / len(batches)

            avg_policy_loss = epoch_policy_loss / update_epochs
            avg_value_loss = epoch_value_loss / update_epochs

            loss_metrics.update(policy_loss = f'{avg_policy_loss:.3f}', value_loss = f'{avg_value_loss:.3f}')

            if use_wandb:
                wandb.log(dict(
                    avg_return = avg_ret,
                    policy_loss = avg_policy_loss,
                    value_loss = avg_value_loss
                ))

            num_policy_updates += 1
            memories.clear()
            episode_lens.clear()
            is_episode_truncated.clear()

            if num_policy_updates >= max_policy_updates:
                log(f'\nReached {num_policy_updates} PPO updates! Stopping training.')
                break

            agent.eval()

        pbar.set_postfix(loss_metrics)

    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    fire.Fire(main)
