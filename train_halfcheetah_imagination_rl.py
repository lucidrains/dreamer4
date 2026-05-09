# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "tqdm",
#     "dreamer4",
#     "fire",
#     "gymnasium[mujoco]",
#     "tensorboard"
# ]
# [tool.uv.sources]
# dreamer4 = { path = "." }
# ///

from __future__ import annotations

import random
import shutil
from copy import deepcopy
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Literal
from math import sqrt

import fire
import gymnasium as gym
import numpy as np
import torch
from einops import rearrange
from adam_atan2_pytorch import MuonAdamAtan2
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dreamer4.dreamer4 import (
    Actions,
    DynamicsWorldModel,
    Experience,
    combine_experiences,
    divisible_by,
    exists,
)


# tokenizer

class ObservationTokenizer(nn.Module):
    def __init__(
        self,
        obs_dim: int = 17,
        num_latent_tokens: int = 4,
        dim_latent: int = 32,
        hidden_dim: int = 256,
        depth: int = 2,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_latent_tokens = num_latent_tokens
        self.dim_latent = dim_latent
        self.eps = eps

        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

        def mlp(dim_in, dim_out):
            layers: list[nn.Module] = []
            dim = dim_in
            for _ in range(depth):
                layers.extend((nn.Linear(dim, hidden_dim), nn.SiLU()))
                dim = hidden_dim
            layers.append(nn.Linear(dim, dim_out))
            return nn.Sequential(*layers)

        self.encoder = mlp(obs_dim, num_latent_tokens * dim_latent)
        self.decoder = mlp(num_latent_tokens * dim_latent, obs_dim)

    @property
    def device(self):
        return self.obs_mean.device

    @torch.no_grad()
    def set_normalization(self, observations: Tensor):
        observations = observations.to(self.device, dtype = torch.float32)
        self.obs_mean.copy_(observations.mean(dim = 0))
        self.obs_std.copy_(observations.std(dim = 0, correction = 0).clamp(min = self.eps))

    def normalize(self, observations: Tensor):
        return (observations - self.obs_mean) / self.obs_std

    def denormalize(self, observations: Tensor):
        return observations * self.obs_std + self.obs_mean

    def encode(self, observations: Tensor):
        shape = observations.shape[:-1]
        observations = self.normalize(observations.float())
        latents = self.encoder(observations.reshape(-1, self.obs_dim)).tanh()
        latents = latents.reshape(*shape, self.num_latent_tokens, self.dim_latent)
        return latents

    def decode(self, latents: Tensor):
        shape = latents.shape[:-2]
        recon = self.decoder(latents.reshape(-1, self.num_latent_tokens * self.dim_latent))
        return self.denormalize(recon.reshape(*shape, self.obs_dim))

    def forward(
        self,
        observations: Tensor,
        *,
        return_latents = False,
        return_recon = False,
    ):
        latents = self.encode(observations)

        if return_latents and not return_recon:
            return latents

        recon = self.decode(latents)
        loss = (self.normalize(recon) - self.normalize(observations.float())).square().mean()

        if return_recon:
            return loss, recon, latents

        return loss


# env helpers

def make_env(
    env_name: str,
    seed: int | None,
    *,
    vectorized = True,
    num_envs = 8,
):
    if vectorized:
        env = gym.make_vec(env_name, num_envs = num_envs)
        env = gym.wrappers.vector.RecordEpisodeStatistics(env)
    else:
        env = gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)

    if exists(seed):
        env.action_space.seed(seed)

    return env


def reset_env(env, seed = None):
    out = env.reset(seed = seed) if exists(seed) else env.reset()
    return out[0] if isinstance(out, tuple) else out


def obs_array(obs):
    obs = np.asarray(obs, dtype = np.float32)
    if obs.ndim == 1:
        obs = obs[None, :]
    return obs


def collect_random_observations(
    env_name: str,
    *,
    num_steps = 4096,
    num_envs = 8,
    seed = 42,
):
    env = make_env(env_name, seed, vectorized = True, num_envs = num_envs)
    obs = reset_env(env, seed = seed)

    observations = [obs_array(obs)]

    for _ in tqdm(range(num_steps), desc = "random obs"):
        action = env.action_space.sample()
        step_out = env.step(action)
        obs = step_out[0]
        observations.append(obs_array(obs))

    env.close()

    return torch.from_numpy(np.concatenate(observations, axis = 0))


def obs_to_latents_fn(tokenizer: ObservationTokenizer):
    @torch.no_grad()
    def inner(_world_model, obs, time_cache):
        state = obs["state"]
        if not torch.is_tensor(state):
            state = torch.tensor(state, device = tokenizer.device, dtype = torch.float32)
        else:
            state = state.to(tokenizer.device, dtype = torch.float32)

        if state.ndim == 1:
            state = rearrange(state, "d -> 1 d")

        latents = tokenizer(state, return_latents = True)
        latents = rearrange(latents, "b n d -> b 1 n d")
        return latents, time_cache

    return inner


# training helpers

def log_scalars(writer: SummaryWriter | None, scalars: dict[str, float], step: int):
    if not exists(writer):
        return

    for key, value in scalars.items():
        if value is None:
            continue
        writer.add_scalar(key, float(value), step)


def split_world_model_and_agent_params(world_model: DynamicsWorldModel):
    agent_params = set(world_model.policy_head_parameters()) | set(world_model.value_head_parameters())
    world_params = [param for param in world_model.parameters() if param not in agent_params]
    return world_params, list(agent_params)


def unique_parameters(params):
    seen = set()
    out = []

    for param in params:
        if param in seen:
            continue

        seen.add(param)
        out.append(param)

    return out


MUON_BYPASS_UPDATE_SENTINEL = "__dreamer4_default_muon_bypass_update__"


def default_muon_bypass_update_fn(ndim: int):
    return ndim < 2 or ndim > 3


def make_optimizer(
    params,
    *,
    lr: float,
    weight_decay: float,
    use_muon: bool,
    muon_params = (),
):
    params = unique_parameters(params)

    if not use_muon:
        return AdamW(params, lr = lr, weight_decay = weight_decay)

    optimizer_param_set = set(params)
    muon_params = [
        param for param in unique_parameters(muon_params)
        if param in optimizer_param_set
    ]

    if len(muon_params) == 0:
        return AdamW(params, lr = lr, weight_decay = weight_decay)

    return MuonAdamAtan2(
        muon_params = muon_params,
        params = params,
        lr = lr,
        weight_decay = weight_decay,
        muon_bypass_update_fn = default_muon_bypass_update_fn,
    )


def module_parameters(module: nn.Module | None):
    return [] if not exists(module) else list(module.parameters())


def optimizer_parameters(optimizer: Optimizer):
    return [param for group in optimizer.param_groups for param in group["params"]]


def disjoint_optimizer_param_groups(
    optimizer: AdamW,
    named_params: list[tuple[str, list[nn.Parameter]]],
    rest_name: str,
):
    optimizer_param_set = set(optimizer_parameters(optimizer))
    seen = set()
    groups = []

    for name, params in named_params:
        group_params = [
            param for param in params
            if param in optimizer_param_set and param not in seen
        ]

        if len(group_params) == 0:
            continue

        seen.update(group_params)
        groups.append((name, group_params))

    rest = [param for param in optimizer_parameters(optimizer) if param not in seen]

    if len(rest) > 0:
        groups.append((rest_name, rest))

    return groups


def clip_grad_norm_by_group(
    groups: list[tuple[str, list[nn.Parameter]]],
    max_grad_norm: float,
):
    metrics = {}
    total_norm_sq = 0.

    for name, params in groups:
        params_with_grad = [param for param in params if exists(param.grad)]

        if len(params_with_grad) == 0:
            continue

        norm = float(clip_grad_norm_(params_with_grad, max_grad_norm))
        metrics[f"{name}_grad_norm"] = norm
        total_norm_sq += norm ** 2

    return sqrt(total_norm_sq), metrics


def world_model_clip_groups(world_model: DynamicsWorldModel, optimizer: Optimizer):
    return disjoint_optimizer_param_groups(
        optimizer,
        [
            ("reward_head", module_parameters(getattr(world_model, "to_reward_pred", None))),
            ("action_head", module_parameters(getattr(world_model, "action_embedder", None))),
            ("terminal_head", module_parameters(getattr(world_model, "to_state_terminal_pred", None))),
            ("state_head", module_parameters(getattr(world_model, "to_state_pred", None))),
            ("agent_state_head", module_parameters(getattr(world_model, "to_agent_state_pred", None))),
            ("latent_ar_head", module_parameters(getattr(world_model, "latent_ar", None))),
        ],
        "trunk",
    )


def agent_clip_groups(world_model: DynamicsWorldModel, optimizer: Optimizer):
    return disjoint_optimizer_param_groups(
        optimizer,
        [
            ("policy_head", module_parameters(getattr(world_model, "policy_head", None))),
            ("action_unembed", list(world_model.action_embedder.unembed_parameters())),
            ("value_head", module_parameters(getattr(world_model, "value_head", None))),
            ("critic_state_embedder", module_parameters(getattr(world_model, "critic_state_embedder", None))),
        ],
        "agent_rest",
    )


def sample_experiences(replay: deque[Experience], batch_size: int):
    batch_size = min(batch_size, len(replay))
    return random.sample(list(replay), batch_size)


def slice_experience(exp: Experience, idx: Tensor):
    slice_maybe = lambda t: t[idx] if torch.is_tensor(t) else t

    actions = Actions(
        slice_maybe(exp.actions.discrete),
        slice_maybe(exp.actions.continuous),
    ) if exists(exp.actions) else None

    log_probs = Actions(
        slice_maybe(exp.log_probs.discrete),
        slice_maybe(exp.log_probs.continuous),
    ) if exists(exp.log_probs) else None

    old_action_unembeds = tuple(slice_maybe(t) for t in exp.old_action_unembeds) if exists(exp.old_action_unembeds) else None

    return Experience(
        latents = slice_maybe(exp.latents),
        video = slice_maybe(exp.video),
        proprio = slice_maybe(exp.proprio),
        critic_state = slice_maybe(exp.critic_state),
        agent_embed = slice_maybe(exp.agent_embed),
        rewards = slice_maybe(exp.rewards),
        terminals = slice_maybe(exp.terminals),
        actions = actions,
        log_probs = log_probs,
        old_action_unembeds = old_action_unembeds,
        values = slice_maybe(exp.values),
        step_size = slice_maybe(exp.step_size),
        lens = slice_maybe(exp.lens),
        is_truncated = slice_maybe(exp.is_truncated),
        agent_index = slice_maybe(exp.agent_index),
        is_from_world_model = slice_maybe(exp.is_from_world_model),
        episode_return = slice_maybe(exp.episode_return),
    )


def sample_experience_batch(replay: deque[Experience], batch_size: int):
    exp = combine_experiences(sample_experiences(replay, batch_size))
    total = exp.latents.shape[0]
    batch_size = min(batch_size, total)
    indices = torch.randperm(total)[:batch_size]
    return slice_experience(exp, indices)


def sample_imagination_prompts(
    replay: deque[Experience],
    batch_size: int,
    prompt_length: int,
    *,
    device: torch.device,
):
    if prompt_length <= 0 or len(replay) == 0:
        return None

    exp = combine_experiences(sample_experiences(replay, batch_size)).to(device)
    assert exists(exp.latents)
    assert exists(exp.actions)
    assert exists(exp.rewards)

    lens = exp.lens if exists(exp.lens) else torch.full((exp.latents.shape[0],), exp.latents.shape[1], device = device)
    valid_indices = torch.nonzero(lens >= prompt_length, as_tuple = False).flatten()

    if valid_indices.numel() == 0:
        return None

    selected = valid_indices[torch.randint(valid_indices.numel(), (batch_size,), device = device)]
    selected_lens = lens[selected].long()
    max_starts = (selected_lens - prompt_length).clamp(min = 0)
    starts = (torch.rand((batch_size,), device = device) * (max_starts + 1).float()).floor().long()

    def take_window(t: Tensor | None, length: int, offset: int = 0):
        if not exists(t):
            return None

        if length == 0:
            return t.new_empty((batch_size, 0, *t.shape[2:]))

        time = starts[:, None] + offset + torch.arange(length, device = device)
        return t[selected[:, None], time]

    return dict(
        prompt_latents = take_window(exp.latents, prompt_length),
        prompt_proprio = take_window(exp.proprio, prompt_length) if exists(exp.proprio) else None,
        prompt_discrete_actions = take_window(exp.actions.discrete, prompt_length) if exists(exp.actions.discrete) else None,
        prompt_continuous_actions = take_window(exp.actions.continuous, prompt_length) if exists(exp.actions.continuous) else None,
        prompt_rewards = take_window(exp.rewards, prompt_length - 1),
    )


def trim_prompt_from_dream(dream: Experience, prompt_length: int, horizon: int):
    if prompt_length <= 0:
        return dream

    prompted_tensors = (
        dream.latents,
        dream.video,
        dream.proprio,
        dream.critic_state,
        dream.rewards,
        dream.actions.discrete if exists(dream.actions) else None,
        dream.actions.continuous if exists(dream.actions) else None,
    )
    generated_tensors = (
        dream.agent_embed,
        dream.values,
        dream.log_probs.discrete if exists(dream.log_probs) else None,
        dream.log_probs.continuous if exists(dream.log_probs) else None,
        *(dream.old_action_unembeds or ()),
    )

    available_lengths = [
        t.shape[1] - prompt_length
        for t in prompted_tensors
        if exists(t)
    ]
    available_lengths.extend([
        t.shape[1]
        for t in generated_tensors
        if exists(t)
    ])

    actual_horizon = min(horizon, *available_lengths)

    def trim_prompted(t: Tensor | None):
        return t[:, prompt_length:prompt_length + actual_horizon] if exists(t) else None

    def trim_generated(t: Tensor | None):
        return t[:, :actual_horizon] if exists(t) else None

    actions = Actions(
        trim_prompted(dream.actions.discrete),
        trim_prompted(dream.actions.continuous),
    ) if exists(dream.actions) else None

    log_probs = Actions(
        trim_generated(dream.log_probs.discrete),
        trim_generated(dream.log_probs.continuous),
    ) if exists(dream.log_probs) else None

    old_action_unembeds = tuple(trim_generated(t) for t in dream.old_action_unembeds) if exists(dream.old_action_unembeds) else None

    lens = None
    if exists(dream.lens):
        lens = (dream.lens - prompt_length).clamp(min = 0, max = actual_horizon)

    rewards = trim_prompted(dream.rewards)
    episode_return = None
    if exists(rewards):
        if exists(lens):
            mask = torch.arange(actual_horizon, device = rewards.device)[None, :] < lens[:, None]
            episode_return = (rewards * mask.float()).sum(dim = 1)
        else:
            episode_return = rewards.sum(dim = 1)

    return Experience(
        latents = trim_prompted(dream.latents),
        video = trim_prompted(dream.video),
        proprio = trim_prompted(dream.proprio),
        critic_state = trim_prompted(dream.critic_state),
        agent_embed = trim_generated(dream.agent_embed),
        rewards = rewards,
        terminals = dream.terminals,
        actions = actions,
        log_probs = log_probs,
        old_action_unembeds = old_action_unembeds,
        values = trim_generated(dream.values),
        step_size = dream.step_size,
        lens = lens,
        is_truncated = dream.is_truncated,
        agent_index = dream.agent_index,
        is_from_world_model = dream.is_from_world_model,
        episode_return = episode_return,
    )


def cat_existing(tensors: tuple[Tensor | None, ...], dim: int = -1):
    tensors = tuple(t for t in tensors if exists(t))
    assert len(tensors) > 0
    return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim = dim)


class FrozenPolicyPrior(nn.Module):
    def __init__(self, world_model: DynamicsWorldModel):
        super().__init__()
        self.policy_head = deepcopy(world_model.policy_head)
        self.action_embedder = deepcopy(world_model.action_embedder)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def refresh_from(self, world_model: DynamicsWorldModel):
        self.policy_head.load_state_dict(world_model.policy_head.state_dict())
        self.action_embedder.load_state_dict(world_model.action_embedder.state_dict())
        self.to(world_model.device)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def action_unembeds(self, agent_embeds: Tensor):
        policy_embed = self.policy_head(agent_embeds.detach())
        return self.action_embedder.unembed(policy_embed, pred_head_index = 0)


@torch.no_grad()
def agent_approx_kl_metrics(world_model: DynamicsWorldModel, dream: Experience):
    assert exists(dream.agent_embed)
    assert exists(dream.actions)
    assert exists(dream.log_probs)

    agent_embeds = dream.agent_embed.detach()
    policy_embed = world_model.policy_head(agent_embeds)
    policy_time = policy_embed.shape[1]

    def align_time(t):
        if not exists(t):
            return None

        return t[:, :policy_time]

    discrete_actions = align_time(dream.actions.discrete)
    continuous_actions = align_time(dream.actions.continuous)
    old_log_probs = Actions(*(align_time(t) for t in dream.log_probs))

    log_probs = world_model.action_embedder.log_probs(
        policy_embed,
        pred_head_index = 0,
        discrete_targets = discrete_actions,
        continuous_targets = continuous_actions,
    )

    old_log_probs = cat_existing(old_log_probs, dim = -1)
    log_probs = cat_existing(log_probs, dim = -1)

    log_ratio = log_probs.sum(dim = -1) - old_log_probs.sum(dim = -1)
    approx_kl = log_ratio.exp() - 1. - log_ratio

    lens = dream.lens
    if exists(lens):
        is_truncated = dream.is_truncated if exists(dream.is_truncated) else torch.ones_like(lens, dtype = torch.bool)
        learnable_lens = (lens - is_truncated.long()).clamp(min = 0, max = policy_time)
        mask = torch.arange(policy_time, device = approx_kl.device)[None, :] < learnable_lens[:, None]
        approx_kl = approx_kl[mask]
    else:
        approx_kl = approx_kl.reshape(-1)

    if approx_kl.numel() == 0:
        approx_kl = log_ratio.new_zeros((1,))

    return {
        "approx_kl_mean": approx_kl.mean(),
        "approx_kl_min": approx_kl.min(),
        "approx_kl_max": approx_kl.max(),
    }


@torch.no_grad()
def agent_prior_kl_metrics(world_model: DynamicsWorldModel, dream: Experience, policy_prior: FrozenPolicyPrior):
    assert exists(dream.agent_embed)

    agent_embeds = dream.agent_embed.detach()
    policy_embed = world_model.policy_head(agent_embeds)
    current_unembeds = world_model.action_embedder.unembed(policy_embed, pred_head_index = 0)
    prior_unembeds = policy_prior.action_unembeds(agent_embeds)

    if world_model.pmpo_reverse_kl:
        current_unembeds, prior_unembeds = prior_unembeds, current_unembeds

    kl_values = None
    for kl_term in world_model.action_embedder.kl_div(current_unembeds, prior_unembeds):
        if not exists(kl_term):
            continue

        if kl_term.ndim == 3 and kl_term.shape[-1] == 1:
            kl_term = rearrange(kl_term, "b t 1 -> b t")

        kl_values = kl_term if not exists(kl_values) else kl_values + kl_term

    assert exists(kl_values)

    lens = dream.lens
    if exists(lens):
        policy_time = kl_values.shape[1]
        is_truncated = dream.is_truncated if exists(dream.is_truncated) else torch.ones_like(lens, dtype = torch.bool)
        learnable_lens = (lens - is_truncated.long()).clamp(min = 0, max = policy_time)
        mask = torch.arange(policy_time, device = kl_values.device)[None, :] < learnable_lens[:, None]
        kl_values = kl_values[mask]
    else:
        kl_values = kl_values.reshape(-1)

    if kl_values.numel() == 0:
        kl_values = agent_embeds.new_zeros((1,))

    return {
        "prior_kl_mean": kl_values.mean(),
        "prior_kl_min": kl_values.min(),
        "prior_kl_max": kl_values.max(),
    }


def attach_policy_prior_unembeds(dream: Experience, policy_prior: FrozenPolicyPrior):
    assert exists(dream.agent_embed)
    dream.old_action_unembeds = policy_prior.action_unembeds(dream.agent_embed)
    return dream


def train_tokenizer(
    tokenizer: ObservationTokenizer,
    observations: Tensor,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    max_grad_norm: float,
    device: torch.device,
    writer: SummaryWriter | None,
):
    tokenizer.to(device)
    observations = observations.to(device)
    tokenizer.set_normalization(observations)

    if steps <= 0:
        return None

    dataset = TensorDataset(observations)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = len(dataset) >= batch_size)
    optimizer = AdamW(tokenizer.parameters(), lr = learning_rate)
    iterator = iter(dataloader)

    last_loss = None
    pbar = tqdm(range(steps), desc = "tokenizer")

    for step in pbar:
        try:
            batch = next(iterator)[0]
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)[0]

        loss = tokenizer(batch)
        loss.backward()

        clip_grad_norm_(tokenizer.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        last_loss = loss.item()
        pbar.set_postfix(loss = f"{last_loss:.4f}")
        log_scalars(writer, {"tokenizer/recon_loss": last_loss}, step)

    return last_loss


@torch.no_grad()
def eval_tokenizer_on_experience(
    tokenizer: ObservationTokenizer,
    exp: Experience,
    *,
    max_samples: int,
):
    states = exp.critic_state

    if not exists(states):
        return None

    batch, time = states.shape[:2]
    device = states.device

    if exists(exp.lens):
        mask = torch.arange(time, device = device)[None, :] < exp.lens[:, None]
        states = states[mask]
    else:
        states = states.reshape(batch * time, states.shape[-1])

    if states.numel() == 0:
        return None

    if max_samples > 0 and states.shape[0] > max_samples:
        indices = torch.linspace(0, states.shape[0] - 1, max_samples, device = device).long()
        states = states[indices]

    sample_count = states.shape[0]

    was_training = tokenizer.training
    tokenizer.eval()
    loss = tokenizer(states)
    tokenizer.train(was_training)

    return loss.item(), sample_count


def train_world_model(
    world_model: DynamicsWorldModel,
    optimizer: Optimizer,
    replay: deque[Experience],
    *,
    steps: int,
    batch_size: int,
    max_grad_norm: float,
    global_step: int,
    writer: SummaryWriter | None,
):
    if len(replay) == 0 or steps <= 0:
        return global_step, {}

    last_metrics = {}
    pbar = tqdm(range(steps), desc = "world model", leave = False)
    grad_clip_groups = world_model_clip_groups(world_model, optimizer)

    for _ in pbar:
        exp = sample_experience_batch(replay, batch_size).to(world_model.device)

        loss, losses = world_model(
            latents = exp.latents,
            rewards = exp.rewards,
            terminals = exp.terminals,
            continuous_actions = exp.actions.continuous if exists(exp.actions) else None,
            lens = exp.lens,
            return_all_losses = True,
            update_loss_ema = True,
        )

        loss.backward()

        norm, grad_metrics = clip_grad_norm_by_group(grad_clip_groups, max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        last_metrics = {
            "world_model/loss": loss.item(),
            "world_model/flow_loss": losses.flow.item(),
            "world_model/shortcut_loss": losses.shortcut.item(),
            "world_model/reward_loss": losses.rewards.mean().item(),
            "world_model/terminal_loss": losses.terminals.item(),
            "world_model/agent_state_pred_loss": losses.agent_state_pred.item(),
            "world_model/grad_norm": float(norm),
        }
        last_metrics.update({
            f"world_model/{key}": value
            for key, value in grad_metrics.items()
        })

        log_scalars(writer, last_metrics, global_step)
        pbar.set_postfix(loss = f"{last_metrics['world_model/loss']:.3f}")
        global_step += 1

    return global_step, last_metrics


def train_agent_in_imagination(
    world_model: DynamicsWorldModel,
    optimizer: Optimizer,
    policy_prior: FrozenPolicyPrior | None,
    replay: deque[Experience],
    *,
    steps: int,
    batch_size: int,
    horizon: int,
    prompt_length: int,
    prompt_probability: float,
    generate_steps: int,
    max_grad_norm: float,
    objective: Literal["ppo", "pmpo", "spo"],
    use_delight_gating: bool,
    global_step: int,
    writer: SummaryWriter | None,
):
    if steps <= 0:
        return global_step, {}

    last_metrics = {}
    pbar = tqdm(range(steps), desc = "imagination", leave = False)
    grad_clip_groups = agent_clip_groups(world_model, optimizer)

    for _ in pbar:
        with torch.no_grad():
            use_prompt = random.random() < prompt_probability
            prompts = sample_imagination_prompts(
                replay,
                batch_size,
                prompt_length,
                device = world_model.device,
            ) if use_prompt else None

            generation_horizon = horizon + prompt_length if exists(prompts) else horizon

            dream = world_model.generate(
                generation_horizon,
                num_steps = generate_steps,
                batch_size = batch_size,
                return_decoded_video = False,
                return_agent_actions = True,
                return_log_probs_and_values = True,
                return_rewards_per_frame = True,
                return_terminals = False,
                store_agent_embed = True,
                store_old_action_unembeds = objective == "pmpo" and not exists(policy_prior),
                **(prompts or {}),
            )

            if exists(prompts):
                dream = trim_prompt_from_dream(dream, prompt_length, horizon)

            if objective == "pmpo" and exists(policy_prior):
                dream = attach_policy_prior_unembeds(dream, policy_prior)

        policy_loss, value_loss = world_model.learn_from_experience(
            dream,
            only_learn_policy_value_heads = True,
            objective = objective,
            use_delight_gating = use_delight_gating,
        )

        loss = policy_loss + value_loss
        loss.backward()

        norm, grad_metrics = clip_grad_norm_by_group(grad_clip_groups, max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        agent_metrics = agent_approx_kl_metrics(world_model, dream)
        if objective == "pmpo" and exists(policy_prior):
            agent_metrics.update(agent_prior_kl_metrics(world_model, dream, policy_prior))

        dream_return = dream.episode_return.mean().item() if exists(dream.episode_return) else 0.
        dream_len = dream.lens.float().mean().item() if exists(dream.lens) else horizon
        action_std = dream.actions.continuous.std().item() if exists(dream.actions) and exists(dream.actions.continuous) else 0.

        last_metrics = {
            "imagination/loss": loss.item(),
            "imagination/policy_loss": policy_loss.item(),
            "imagination/value_loss": value_loss.item(),
            "imagination/dream_return": dream_return,
            "imagination/dream_length": dream_len,
            "imagination/action_std": action_std,
            "imagination/prompt_length": prompt_length if exists(prompts) else 0,
            "imagination/grad_norm": float(norm),
            "imagination/approx_kl_mean": agent_metrics["approx_kl_mean"].item(),
            "imagination/approx_kl_min": agent_metrics["approx_kl_min"].item(),
            "imagination/approx_kl_max": agent_metrics["approx_kl_max"].item(),
            "imagination/prior_kl_mean": agent_metrics.get("prior_kl_mean", torch.tensor(0.)).item(),
            "imagination/prior_kl_min": agent_metrics.get("prior_kl_min", torch.tensor(0.)).item(),
            "imagination/prior_kl_max": agent_metrics.get("prior_kl_max", torch.tensor(0.)).item(),
        }
        last_metrics.update({
            f"imagination/{key}": value
            for key, value in grad_metrics.items()
        })

        log_scalars(writer, last_metrics, global_step)
        pbar.set_postfix(return_ = f"{dream_return:.1f}", loss = f"{loss.item():.3f}")
        global_step += 1

    return global_step, last_metrics


def save_checkpoint(
    path: Path,
    *,
    loop: int,
    tokenizer: ObservationTokenizer,
    world_model: DynamicsWorldModel,
    world_optimizer: Optimizer,
    agent_optimizer: Optimizer,
):
    path.parent.mkdir(parents = True, exist_ok = True)
    torch.save(
        dict(
            loop = loop,
            tokenizer = tokenizer.state_dict(),
            world_model = world_model.state_dict(),
            world_optimizer = checkpoint_optimizer_state_dict(world_optimizer),
            agent_optimizer = checkpoint_optimizer_state_dict(agent_optimizer),
        ),
        str(path),
    )


def checkpoint_optimizer_state_dict(optimizer: Optimizer):
    state_dict = optimizer.state_dict()

    for group in state_dict["param_groups"]:
        if callable(group.get("muon_bypass_update_fn")):
            group["muon_bypass_update_fn"] = MUON_BYPASS_UPDATE_SENTINEL

    return state_dict


def restore_optimizer_runtime_options(optimizer: Optimizer):
    for group in optimizer.param_groups:
        if group.get("muon_bypass_update_fn") == MUON_BYPASS_UPDATE_SENTINEL:
            group["muon_bypass_update_fn"] = default_muon_bypass_update_fn


def load_optimizer_state_if_compatible(optimizer: Optimizer, state_dict, name: str):
    try:
        optimizer.load_state_dict(state_dict)
        restore_optimizer_runtime_options(optimizer)
        return True
    except (KeyError, RuntimeError, ValueError) as exc:
        print(f"skipping {name} optimizer state: {exc}")
        return False


def resolve_log_dir(log_dir: str, checkpoint_path: str | None):
    log_root = Path(log_dir)

    if exists(checkpoint_path):
        return log_root / f"resume_{Path(checkpoint_path).stem}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_root / timestamp


def main(
    env_name = "HalfCheetah-v4",
    num_loops = 500,
    rollouts_per_loop = 1,
    num_envs = 32,
    max_timesteps = 200,
    replay_size = 256,
    seed = 42,
    cpu = False,
    obs_dim = 17,
    num_latent_tokens = 4,
    dim_latent = 32,
    model_dim = 128,
    depth = 3,
    time_block_every = 2,
    final_special_cross_attn = False,
    reward_encoder_type: Literal["symexp_two_hot", "hl_gauss"] = "hl_gauss",
    world_model_batch_size = 32,
    world_model_train_steps = 13,
    world_model_learning_rate = 3e-4,
    imagination_batch_size = 128,
    imagination_horizon = 32,
    imagination_prompt_length = 8,
    imagination_prompt_probability = 1.,
    imagination_train_steps = 3,
    imagination_generate_steps = 4,
    agent_learning_rate = 3e-4,
    use_muon_optimizer = True,
    optimizer_weight_decay = 0.01,
    objective: Literal["ppo", "pmpo", "spo"] = "pmpo",
    pmpo_pos_to_neg_weight = 0.5,
    pmpo_kl_div_loss_weight = 0.3,
    use_delight_gating = False,
    add_action_embed_to_agent_token = True,
    agent_predicts_state = True,
    agent_state_pred_loss_weight = 0.1,
    pretrain_tokenizer_steps = 1000,
    pretrain_tokenizer_observations = 8192,
    tokenizer_batch_size = 256,
    tokenizer_learning_rate = 3e-4,
    tokenizer_eval_every = 10,
    tokenizer_eval_batch_size = 2048,
    max_grad_norm = 0.5,
    use_tensorboard = True,
    log_dir = "runs/halfcheetah_imagination",
    checkpoint_folder = "checkpoints_halfcheetah_imagination",
    checkpoint_every = 25,
    checkpoint_path: str | None = None,
    clear_log_dir = False,
    unique_log_dir = True,
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")

    run_log_dir = resolve_log_dir(log_dir, checkpoint_path) if unique_log_dir else Path(log_dir)

    if clear_log_dir:
        shutil.rmtree(run_log_dir, ignore_errors = True)

    writer = SummaryWriter(str(run_log_dir)) if use_tensorboard else None

    tokenizer = ObservationTokenizer(
        obs_dim = obs_dim,
        num_latent_tokens = num_latent_tokens,
        dim_latent = dim_latent,
    ).to(device)

    tokenizer_loss = None

    if not exists(checkpoint_path):
        pretrain_obs = collect_random_observations(
            env_name,
            num_steps = max(1, pretrain_tokenizer_observations // num_envs),
            num_envs = num_envs,
            seed = seed,
        )[:pretrain_tokenizer_observations]

        assert pretrain_obs.shape[-1] == obs_dim, f"{env_name} produced obs dim {pretrain_obs.shape[-1]}, expected {obs_dim}"

        tokenizer_loss = train_tokenizer(
            tokenizer,
            pretrain_obs,
            steps = pretrain_tokenizer_steps,
            batch_size = tokenizer_batch_size,
            learning_rate = tokenizer_learning_rate,
            max_grad_norm = max_grad_norm,
            device = device,
            writer = writer,
        )

    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad_(False)

    env = make_env(env_name, seed, vectorized = True, num_envs = num_envs)

    single_action_space = getattr(env, "single_action_space", env.action_space)
    action_dim = int(np.prod(single_action_space.shape))
    action_lows = np.asarray(single_action_space.low)
    action_highs = np.asarray(single_action_space.high)

    assert np.allclose(action_lows, -1.) and np.allclose(action_highs, 1.), (
        "this script expects the native policy target range to match the env action range [-1, 1]"
    )

    reward_range = (-20., 20.)
    value_range = tuple(bound * imagination_horizon for bound in reward_range)

    world_model = DynamicsWorldModel(
        dim = model_dim,
        dim_latent = dim_latent,
        max_steps = 64,
        num_latent_tokens = num_latent_tokens,
        num_spatial_tokens = num_latent_tokens,
        num_register_tokens = 1,
        dim_critic_state = obs_dim,
        depth = depth,
        time_block_every = time_block_every,
        num_discrete_actions = 0,
        num_continuous_actions = action_dim,
        continuous_dist_type = "beta",
        continuous_target_action_range = (-1., 1.),
        reward_encoder_type = reward_encoder_type,
        reward_encoder_kwargs = dict(reward_range = reward_range),
        value_encoder_kwargs = dict(reward_range = value_range),
        predict_terminals = True,
        continuous_action_loss_weight = 0.,
        discrete_action_loss_weight = 0.,
        agent_predicts_state = agent_predicts_state,
        agent_state_pred_loss_weight = agent_state_pred_loss_weight,
        gae_discount_factor = 0.99,
        pmpo_pos_to_neg_weight = pmpo_pos_to_neg_weight,
        pmpo_kl_div_loss_weight = pmpo_kl_div_loss_weight,
        ppo_eps_clip = 0.2,
        normalize_advantages = True,
        policy_entropy_weight = 0.01,
        use_loss_normalization = False,
        add_action_embed_to_agent_token = add_action_embed_to_agent_token,
        attn_heads = 4,
        attn_dim_head = 16,
        final_special_cross_attn = final_special_cross_attn,
        policy_head_mlp_depth = 2,
        value_head_mlp_depth = 2,
    ).to(device)

    world_params, agent_params = split_world_model_and_agent_params(world_model)
    world_optimizer = make_optimizer(
        world_params,
        lr = world_model_learning_rate,
        weight_decay = optimizer_weight_decay,
        use_muon = use_muon_optimizer,
        muon_params = world_model.muon_parameters(),
    )
    agent_optimizer = make_optimizer(
        agent_params,
        lr = agent_learning_rate,
        weight_decay = optimizer_weight_decay,
        use_muon = use_muon_optimizer,
        muon_params = world_model.muon_parameters(),
    )

    start_loop = 0
    if exists(checkpoint_path):
        pkg = torch.load(checkpoint_path, map_location = device, weights_only = True)
        tokenizer.load_state_dict(pkg["tokenizer"])
        world_model.load_state_dict(pkg["world_model"])
        load_optimizer_state_if_compatible(world_optimizer, pkg["world_optimizer"], "world")
        load_optimizer_state_if_compatible(agent_optimizer, pkg["agent_optimizer"], "agent")
        start_loop = int(pkg.get("loop", 0)) + 1

    policy_prior = FrozenPolicyPrior(world_model).to(device) if objective == "pmpo" else None

    replay: deque[Experience] = deque(maxlen = replay_size)
    wm_step = 0
    imagination_step = 0

    print(f"training {env_name} from {obs_dim} raw observations on {device}")
    print(f"tensorboard log dir: {run_log_dir.absolute()}" if use_tensorboard else "tensorboard disabled")
    if exists(tokenizer_loss):
        print(f"tokenizer pretrain recon loss: {tokenizer_loss:.4f}")

    pbar = tqdm(range(start_loop, num_loops), desc = "loops")

    for loop in pbar:
        world_model.eval()
        tokenizer_eval_loss_sum = 0.
        tokenizer_eval_sample_count = 0
        rollout_returns = []

        for rollout_idx in range(rollouts_per_loop):
            exp = world_model.interact_with_env(
                env,
                seed = seed if loop == 0 and rollout_idx == 0 else None,
                max_timesteps = max_timesteps,
                env_is_vectorized = True,
                store_agent_embed = False,
                store_old_action_unembeds = False,
                obs_to_latents_fn = obs_to_latents_fn(tokenizer),
            )

            if tokenizer_eval_every > 0 and divisible_by(loop, tokenizer_eval_every):
                tokenizer_eval_loss = eval_tokenizer_on_experience(
                    tokenizer,
                    exp,
                    max_samples = tokenizer_eval_batch_size,
                )

                if exists(tokenizer_eval_loss):
                    loss, sample_count = tokenizer_eval_loss
                    tokenizer_eval_loss_sum += loss * sample_count
                    tokenizer_eval_sample_count += sample_count

            replay.append(exp.to("cpu"))
            rollout_returns.extend(exp.episode_return.detach().cpu().tolist())

        avg_return = float(np.mean(rollout_returns)) if len(rollout_returns) > 0 else 0.
        avg_length = float(np.mean([exp.lens.float().mean().item() for exp in replay])) if len(replay) > 0 else 0.
        tokenizer_policy_recon_loss = tokenizer_eval_loss_sum / tokenizer_eval_sample_count if tokenizer_eval_sample_count > 0 else None

        log_scalars(
            writer,
            {
                "rollout/average_return": avg_return,
                "rollout/replay_size": len(replay),
                "rollout/average_length": avg_length,
                "tokenizer/policy_recon_loss": tokenizer_policy_recon_loss,
            },
            loop,
        )

        world_model.train()
        wm_step, wm_metrics = train_world_model(
            world_model,
            world_optimizer,
            replay,
            steps = world_model_train_steps,
            batch_size = world_model_batch_size,
            max_grad_norm = max_grad_norm,
            global_step = wm_step,
            writer = writer,
        )

        world_model.train()
        if exists(policy_prior):
            policy_prior.refresh_from(world_model)

        imagination_step, imagination_metrics = train_agent_in_imagination(
            world_model,
            agent_optimizer,
            policy_prior,
            replay,
            steps = imagination_train_steps,
            batch_size = imagination_batch_size,
            horizon = imagination_horizon,
            prompt_length = imagination_prompt_length,
            prompt_probability = imagination_prompt_probability,
            generate_steps = imagination_generate_steps,
            max_grad_norm = max_grad_norm,
            objective = objective,
            use_delight_gating = use_delight_gating,
            global_step = imagination_step,
            writer = writer,
        )

        postfix = {"return": f"{avg_return:.1f}", "replay": len(replay)}
        if wm_metrics:
            postfix["wm"] = f"{wm_metrics['world_model/loss']:.2f}"
        if imagination_metrics:
            postfix["dream"] = f"{imagination_metrics['imagination/dream_return']:.1f}"
        pbar.set_postfix(postfix)

        if checkpoint_every > 0 and divisible_by(loop + 1, checkpoint_every):
            save_checkpoint(
                Path(checkpoint_folder) / f"loop_{loop + 1}.pt",
                loop = loop,
                tokenizer = tokenizer,
                world_model = world_model,
                world_optimizer = world_optimizer,
                agent_optimizer = agent_optimizer,
            )

    env.close()

    if exists(writer):
        writer.flush()
        writer.close()


if __name__ == "__main__":
    fire.Fire(main)
