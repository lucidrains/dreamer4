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
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
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


def module_parameters(module: nn.Module | None):
    return [] if not exists(module) else list(module.parameters())


def optimizer_parameters(optimizer: AdamW):
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


def world_model_clip_groups(world_model: DynamicsWorldModel, optimizer: AdamW):
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


def agent_clip_groups(world_model: DynamicsWorldModel, optimizer: AdamW):
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


def cat_existing(tensors: tuple[Tensor | None, ...], dim: int = -1):
    tensors = tuple(t for t in tensors if exists(t))
    assert len(tensors) > 0
    return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim = dim)


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


def train_world_model(
    world_model: DynamicsWorldModel,
    optimizer: AdamW,
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
    optimizer: AdamW,
    *,
    steps: int,
    batch_size: int,
    horizon: int,
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
            dream = world_model.generate(
                horizon,
                num_steps = generate_steps,
                batch_size = batch_size,
                return_decoded_video = False,
                return_for_policy_optimization = True,
                return_terminals = True,
                store_agent_embed = True,
                store_old_action_unembeds = True,
            )

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
            "imagination/grad_norm": float(norm),
            "imagination/agent_grad_norm": float(norm),
            "imagination/approx_kl_mean": agent_metrics["approx_kl_mean"].item(),
            "imagination/approx_kl_min": agent_metrics["approx_kl_min"].item(),
            "imagination/approx_kl_max": agent_metrics["approx_kl_max"].item(),
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
    world_optimizer: AdamW,
    agent_optimizer: AdamW,
):
    path.parent.mkdir(parents = True, exist_ok = True)
    torch.save(
        dict(
            loop = loop,
            tokenizer = tokenizer.state_dict(),
            world_model = world_model.state_dict(),
            world_optimizer = world_optimizer.state_dict(),
            agent_optimizer = agent_optimizer.state_dict(),
        ),
        str(path),
    )


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
    final_special_cross_attn = False,
    reward_encoder_type: Literal["symexp_two_hot", "hl_gauss"] = "hl_gauss",
    world_model_batch_size = 32,
    world_model_train_steps = 13,
    world_model_learning_rate = 3e-4,
    imagination_batch_size = 64,
    imagination_horizon = 32,
    imagination_train_steps = 5,
    imagination_generate_steps = 4,
    agent_learning_rate = 3e-4,
    objective: Literal["ppo", "pmpo", "spo"] = "ppo",
    use_delight_gating = True,
    agent_predicts_state = True,
    agent_state_pred_loss_weight = 0.1,
    pretrain_tokenizer_steps = 1000,
    pretrain_tokenizer_observations = 8192,
    tokenizer_batch_size = 256,
    tokenizer_learning_rate = 3e-4,
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
        time_block_every = max(depth, 1),
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
        ppo_eps_clip = 0.2,
        normalize_advantages = True,
        policy_entropy_weight = 0.01,
        use_loss_normalization = False,
        attn_heads = 4,
        attn_dim_head = 16,
        final_special_cross_attn = final_special_cross_attn,
        policy_head_mlp_depth = 2,
        value_head_mlp_depth = 2,
    ).to(device)

    world_params, agent_params = split_world_model_and_agent_params(world_model)
    world_optimizer = AdamW(world_params, lr = world_model_learning_rate)
    agent_optimizer = AdamW(agent_params, lr = agent_learning_rate)

    start_loop = 0
    if exists(checkpoint_path):
        pkg = torch.load(checkpoint_path, map_location = device, weights_only = True)
        tokenizer.load_state_dict(pkg["tokenizer"])
        world_model.load_state_dict(pkg["world_model"])
        world_optimizer.load_state_dict(pkg["world_optimizer"])
        agent_optimizer.load_state_dict(pkg["agent_optimizer"])
        start_loop = int(pkg.get("loop", 0)) + 1

    replay: deque[Experience] = deque(maxlen = replay_size)
    recent_returns = deque(maxlen = 20)
    wm_step = 0
    imagination_step = 0

    print(f"training {env_name} from {obs_dim} raw observations on {device}")
    print(f"tensorboard log dir: {run_log_dir.absolute()}" if use_tensorboard else "tensorboard disabled")
    if exists(tokenizer_loss):
        print(f"tokenizer pretrain recon loss: {tokenizer_loss:.4f}")

    pbar = tqdm(range(start_loop, num_loops), desc = "loops")

    for loop in pbar:
        world_model.eval()

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

            replay.append(exp.to("cpu"))
            recent_returns.extend(exp.episode_return.detach().cpu().tolist())

        avg_return = float(np.mean(recent_returns)) if len(recent_returns) > 0 else 0.
        avg_length = float(np.mean([exp.lens.float().mean().item() for exp in replay])) if len(replay) > 0 else 0.

        log_scalars(
            writer,
            {
                "rollout/average_return_20": avg_return,
                "rollout/replay_size": len(replay),
                "rollout/average_length": avg_length,
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
        imagination_step, imagination_metrics = train_agent_in_imagination(
            world_model,
            agent_optimizer,
            steps = imagination_train_steps,
            batch_size = imagination_batch_size,
            horizon = imagination_horizon,
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
