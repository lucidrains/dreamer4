from __future__ import annotations
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, Parameter, Sequential, RMSNorm, Embedding
from torch import Tensor, nn, cat, tensor, full, zeros, ones, randint, rand, randn, randn_like, empty

import math
from math import ceil, log2
from functools import partial, wraps
from contextlib import nullcontext

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

import einx
from einx import add
from torch.distributions import Normal, kl
from x_mlps_pytorch.ensemble import Ensemble
from x_mlps_pytorch.normed_mlp import create_mlp
from torch.optim import Optimizer

from collections import namedtuple

# Local imports
from .utils import (
    align_dims_left,
    calc_gae,
    create_multi_token_prediction_targets,
    default,
    divisible_by,
    exists,
    has_at_least_one,
    is_empty,
    is_power_two,
    lens_to_mask,
    masked_mean,
    pack_one,
    pad_at_dim,
    ramp_weight,
    safe_cat,
    sample_prob,
    xnor,
)
from .experience import Experience
from .losses import LossNormalizer
from .encodings import SymExpTwoHot
from .embeddings import ActionEmbedder
from .transformers import AxialSpaceTimeTransformer

# Constants and Type Aliases

LinearNoBias = partial(Linear, bias = False)
WorldModelLosses = namedtuple('WorldModelLosses', ('flow', 'rewards', 'discrete_actions', 'continuous_actions', 'state_pred'))
Predictions = namedtuple('Predictions', ('flow', 'proprioception', 'state'))
Embeds = namedtuple('Embeds', ['agent', 'state_pred'])
MaybeTensor = Tensor | None

class DynamicsWorldModel(Module):
    def __init__(
        self,
        dim,
        dim_latent,
        video_tokenizer: VideoTokenizer | None = None,
        max_steps = 64,                # K_max in paper
        num_register_tokens = 8,       # they claim register tokens led to better temporal consistency
        num_spatial_tokens = 2,        # latents projected to greater number of spatial tokens
        num_latent_tokens = None,
        num_agents = 1,
        num_tasks = 0,
        num_video_views = 1,
        dim_proprio = None,
        reward_encoder_kwargs: dict = dict(),
        depth = 4,
        pred_orig_latent = True,   # directly predicting the original x0 data yield better results, rather than velocity (x-space vs v-space)
        time_block_every = 4,      # every 4th block is time
        attn_kwargs: dict = dict(),
        transformer_kwargs: dict = dict(),
        attn_heads = 8,
        attn_dim_head = 64,
        attn_softclamp_value = 50.,
        ff_kwargs: dict = dict(),
        use_time_rnn = True,
        loss_weight_fn: Callable = ramp_weight,
        prob_no_shortcut_train = None,              # probability of no shortcut training, defaults to 1 / num_step_sizes
        add_reward_embed_to_agent_token = False,
        add_reward_embed_dropout = 0.1,
        add_state_pred_head = False,
        state_pred_loss_weight = 0.1,
        state_entropy_bonus_weight = 0.05,
        num_discrete_actions: int | tuple[int, ...] = 0,
        num_continuous_actions = 0,
        continuous_norm_stats = None,
        multi_token_pred_len = 8,                   # they do multi-token prediction of 8 steps forward
        value_head_mlp_depth = 3,
        policy_head_mlp_depth = 3,
        latent_flow_loss_weight = 1.,
        reward_loss_weight: float | list[float] = 1.,
        discrete_action_loss_weight: float | list[float] = 1.,
        continuous_action_loss_weight: float | list[float] = 1.,
        num_latent_genes = 0,                       # for carrying out evolution within the dreams https://web3.arxiv.org/abs/2503.19037
        num_residual_streams = 1,
        keep_reward_ema_stats = False,
        reward_ema_decay = 0.998,
        reward_quantile_filter = (0.05, 0.95),
        gae_discount_factor = 0.997,
        gae_lambda = 0.95,
        ppo_eps_clip = 0.2,
        pmpo_pos_to_neg_weight = 0.5, # pos and neg equal weight
        pmpo_reverse_kl = True,
        pmpo_kl_div_loss_weight = .3,
        normalize_advantages = None,
        value_clip = 0.4,
        policy_entropy_weight = .01,
        gae_use_accelerated = False
    ):
        super().__init__()

        # can accept raw video if tokenizer is passed in

        self.video_tokenizer = video_tokenizer

        if exists(video_tokenizer):
            num_latent_tokens = default(num_latent_tokens, video_tokenizer.num_latent_tokens)
            assert video_tokenizer.num_latent_tokens == num_latent_tokens, f'`num_latent_tokens` must be the same for the tokenizer and dynamics model'

        assert exists(num_latent_tokens), '`num_latent_tokens` must be set'

        # spatial

        self.num_latent_tokens = num_latent_tokens
        self.dim_latent = dim_latent
        self.latent_shape = (num_latent_tokens, dim_latent)

        if num_spatial_tokens >= num_latent_tokens:
            assert divisible_by(num_spatial_tokens, num_latent_tokens)

            expand_factor = num_spatial_tokens // num_latent_tokens

            self.latents_to_spatial_tokens = Sequential(
                Linear(dim_latent, dim * expand_factor),
                Rearrange('... (s d) -> ... s d', s = expand_factor)
            )

            self.to_latent_pred = Sequential(
                Reduce('b t v n s d -> b t v n d', 'mean'),
                RMSNorm(dim),
                LinearNoBias(dim, dim_latent)
            )

        else:
            assert divisible_by(num_latent_tokens, num_spatial_tokens)
            latent_tokens_to_space = num_latent_tokens // num_spatial_tokens

            self.latents_to_spatial_tokens = Sequential(
                Rearrange('... n d -> ... (n d)'),
                Linear(num_latent_tokens * dim_latent, dim * num_spatial_tokens),
                Rearrange('... (s d) -> ... s d', s = num_spatial_tokens)
            )

            self.to_latent_pred = Sequential(
                RMSNorm(dim),
                LinearNoBias(dim, dim_latent * latent_tokens_to_space),
                Rearrange('b t v s (n d) -> b t v (s n) d', n = latent_tokens_to_space)
            )

        # number of video views, for robotics, which could have third person + wrist camera at least

        assert num_video_views >= 1
        self.video_has_multi_view = num_video_views > 1

        self.num_video_views = num_video_views

        if self.video_has_multi_view:
            self.view_emb = nn.Parameter(torch.randn(num_video_views, dim) * 1e-2)

        # proprioception

        self.has_proprio = exists(dim_proprio)
        self.dim_proprio = dim_proprio

        if self.has_proprio:
            self.to_proprio_token = nn.Linear(dim_proprio, dim)

            self.to_proprio_pred = Sequential(
                RMSNorm(dim),
                nn.Linear(dim, dim_proprio)
            )

        # register tokens

        self.num_register_tokens = num_register_tokens
        self.register_tokens = Parameter(torch.randn(num_register_tokens, dim) * 1e-2)

        # signal and step sizes

        assert divisible_by(dim, 2)
        dim_half = dim // 2

        assert is_power_two(max_steps), '`max_steps` must be a power of 2'
        self.max_steps = max_steps
        self.num_step_sizes_log2 = int(log2(max_steps))

        self.signal_levels_embed = nn.Embedding(max_steps, dim_half)
        self.step_size_embed = nn.Embedding(self.num_step_sizes_log2, dim_half) # power of 2, so 1/1, 1/2, 1/4, 1/8 ... 1/Kmax

        self.prob_no_shortcut_train = default(prob_no_shortcut_train, self.num_step_sizes_log2 ** -1.)

        # loss related

        self.pred_orig_latent = pred_orig_latent # x-space or v-space
        self.loss_weight_fn = loss_weight_fn

        # state prediction, for state entropy bonus

        self.add_state_pred_head = add_state_pred_head
        self.state_pred_loss_weight = state_pred_loss_weight

        self.should_pred_state = add_state_pred_head and state_pred_loss_weight > 0.

        if self.should_pred_state:
            self.state_pred_token = nn.Parameter(torch.randn(dim) * 1e-2)

            self.to_state_pred = Sequential(
                RMSNorm(dim),
                nn.Linear(dim, num_latent_tokens * dim_latent * 2),
                Rearrange('... (n d two) -> ... n d two', n = num_latent_tokens, two = 2)
            )

        self.state_entropy_bonus_weight = state_entropy_bonus_weight
        self.add_state_entropy_bonus = self.should_pred_state and state_entropy_bonus_weight > 0.

        # reinforcement related

        # they sum all the actions into a single token

        self.num_agents = num_agents

        self.agent_learned_embed = Parameter(randn(self.num_agents, dim) * 1e-2)
        self.action_learned_embed = Parameter(randn(self.num_agents, dim) * 1e-2)

        self.reward_learned_embed = Parameter(randn(self.num_agents, dim) * 1e-2)

        self.num_tasks = num_tasks
        self.task_embed = nn.Embedding(num_tasks, dim)

        # learned set of latent genes

        self.agent_has_genes = num_latent_genes > 0
        self.num_latent_genes = num_latent_genes
        self.latent_genes = Parameter(randn(num_latent_genes, dim) * 1e-2)

        # policy head

        self.policy_head = create_mlp(
            dim_in = dim,
            dim = dim * 4,
            dim_out = dim * 4,
            depth = policy_head_mlp_depth
        )

        # action embedder

        self.action_embedder = ActionEmbedder(
            dim = dim,
            num_discrete_actions = num_discrete_actions,
            num_continuous_actions = num_continuous_actions,
            continuous_norm_stats = continuous_norm_stats,
            can_unembed = True,
            unembed_dim = dim * 4,
            num_unembed_preds = multi_token_pred_len,
            squeeze_unembed_preds = False
        )

        # multi token prediction length

        self.multi_token_pred_len = multi_token_pred_len

        # each agent token will have the reward embedding of the previous time step - but could eventually just give reward its own token

        self.add_reward_embed_to_agent_token = add_reward_embed_to_agent_token
        self.add_reward_embed_dropout = add_reward_embed_dropout

        self.reward_encoder = SymExpTwoHot(
            **reward_encoder_kwargs,
            dim_embed = dim,
            learned_embedding = add_reward_embed_to_agent_token
        )

        to_reward_pred = Sequential(
            RMSNorm(dim),
            LinearNoBias(dim, self.reward_encoder.num_bins)
        )

        self.to_reward_pred = Ensemble(
            to_reward_pred,
            multi_token_pred_len
        )

        # value head

        self.value_head = create_mlp(
            dim_in = dim,
            dim = dim * 4,
            dim_out = self.reward_encoder.num_bins,
            depth = value_head_mlp_depth,
        )

        # efficient axial space / time transformer

        self.transformer = AxialSpaceTimeTransformer(
            dim = dim,
            depth = depth,
            attn_heads = attn_heads,
            attn_dim_head = attn_dim_head,
            attn_softclamp_value = attn_softclamp_value,
            attn_kwargs = attn_kwargs,
            ff_kwargs = ff_kwargs,
            num_residual_streams = num_residual_streams,
            num_special_spatial_tokens = num_agents,
            time_block_every = time_block_every,
            final_norm = False,
            rnn_time = use_time_rnn,
            **transformer_kwargs
        )

        # ppo related

        self.gae_use_accelerated = gae_use_accelerated
        self.gae_discount_factor = gae_discount_factor
        self.gae_lambda = gae_lambda

        self.ppo_eps_clip = ppo_eps_clip
        self.value_clip = value_clip
        self.policy_entropy_weight = policy_entropy_weight

        # pmpo related

        self.pmpo_pos_to_neg_weight = pmpo_pos_to_neg_weight
        self.pmpo_kl_div_loss_weight = pmpo_kl_div_loss_weight
        self.pmpo_reverse_kl = pmpo_reverse_kl

        # rewards related

        self.keep_reward_ema_stats = keep_reward_ema_stats
        self.reward_ema_decay = reward_ema_decay

        self.register_buffer('reward_quantile_filter', tensor(reward_quantile_filter), persistent = False)

        self.register_buffer('ema_returns_mean', tensor(0.))
        self.register_buffer('ema_returns_var', tensor(1.))

        # loss related

        self.flow_loss_normalizer = LossNormalizer(1)
        self.reward_loss_normalizer = LossNormalizer(multi_token_pred_len)
        self.discrete_actions_loss_normalizer = LossNormalizer(multi_token_pred_len) if num_discrete_actions > 0 else None
        self.continuous_actions_loss_normalizer = LossNormalizer(multi_token_pred_len) if num_continuous_actions > 0 else None

        self.latent_flow_loss_weight = latent_flow_loss_weight

        self.register_buffer('reward_loss_weight', tensor(reward_loss_weight))
        self.register_buffer('discrete_action_loss_weight', tensor(discrete_action_loss_weight))
        self.register_buffer('continuous_action_loss_weight', tensor(continuous_action_loss_weight))

        assert self.reward_loss_weight.numel() in {1, multi_token_pred_len}
        assert self.discrete_action_loss_weight.numel() in {1, multi_token_pred_len}
        assert self.continuous_action_loss_weight.numel() in {1, multi_token_pred_len}

        self.register_buffer('zero', tensor(0.), persistent = False)

    @property
    def device(self):
        return self.zero.device

    # types of parameters

    def muon_parameters(self):
        return self.transformer.muon_parameters()

    def policy_head_parameters(self):
        return [
            *self.policy_head.parameters(),
            *self.action_embedder.unembed_parameters() # includes the unembed from the action-embedder
        ]

    def value_head_parameters(self):
        return self.value_head.parameters()

    def parameter(self):
        params = super().parameters()

        if not exists(self.video_tokenizer):
            return params

        return list(set(params) - set(self.video_tokenizer.parameters()))

    # helpers for shortcut flow matching

    def get_times_from_signal_level(
        self,
        signal_levels,
        align_dims_left_to = None
    ):
        times = signal_levels.float() / self.max_steps

        if not exists(align_dims_left_to):
            return times

        aligned_times, _ = align_dims_left((times,), align_dims_left_to)
        return aligned_times

    # evolutionary policy optimization - https://web3.arxiv.org/abs/2503.19037

    @torch.no_grad()
    def evolve_(
        self,
        fitness,
        select_frac = 0.5,
        tournament_frac = 0.5
    ):
        assert fitness.numel() == self.num_latent_genes

        pop = self.latent_genes

        pop_size = self.num_latent_genes
        num_selected = ceil(pop_size * select_frac)
        num_children = pop_size - num_selected

        dim_gene = pop.shape[-1]

        # natural selection just a sort and slice

        selected_fitness, selected_indices = fitness.topk(num_selected, dim = -1)
        selected = pop[selected_indices]

        # use tournament - one tournament per child

        tournament_size = max(2, ceil(num_selected * tournament_frac))

        tournaments = torch.randn((num_children, num_selected), device = self.device).argsort(dim = -1)[:, :tournament_size]

        parent_ids = selected_fitness[tournaments].topk(2, dim = -1).indices # get top 2 winners as parents

        parents = selected[parent_ids]

        # crossover by random interpolation from parent1 to parent2

        random_uniform_mix = torch.randn((num_children, dim_gene), device = self.device).sigmoid()

        parent1, parent2 = parents.unbind(dim = 1)
        children = parent1.lerp(parent2, random_uniform_mix)

        # store next population

        next_pop = cat((selected, children))

        self.latent_genes.copy_(next_pop)

    # interacting with env for experience

    @torch.no_grad()
    def interact_with_env(
        self,
        env,
        seed = None,
        agent_index = 0,
        num_steps = 4,
        max_timesteps = 16,
        env_is_vectorized = False,
        use_time_cache = True,
        store_agent_embed = True,
        store_old_action_unembeds = True,
    ):
        assert exists(self.video_tokenizer)

        init_obs = env.reset()

        assert 'image' in init_obs
        if self.has_proprio:
            assert 'proprio' in init_obs

        proprio = init_obs.get('proprio')

        # frame to video

        if env_is_vectorized:
            video = rearrange(init_obs['image'], 'b c vh vw -> b c 1 vh vw')
            accumulated_proprio = safe_rearrange(proprio, 'b d -> b 1 d')
        else:
            video = rearrange(init_obs['image'], 'c vh vw -> 1 c 1 vh vw')
            accumulated_proprio = safe_rearrange(proprio, 'd -> 1 1 d')

        batch, device = video.shape[0], video.device

        # accumulate

        rewards = None
        discrete_actions = None
        continuous_actions = None
        discrete_log_probs = None
        continuous_log_probs = None
        values = None
        latents = None

        acc_agent_embed = None
        acc_policy_embed = None

        # keep track of termination, for setting the `is_truncated` field on Experience and for early stopping interaction with env

        is_terminated = full((batch,), False, device = device)
        is_truncated = full((batch,), False, device = device)

        episode_lens = full((batch,), 0, device = device)

        # derive step size

        assert divisible_by(self.max_steps, num_steps)
        step_size = self.max_steps // num_steps

        # maybe time kv cache

        time_cache = None

        step_index = 0

        while not is_terminated.all():
            step_index += 1

            latents = self.video_tokenizer(video, return_latents = True)

            _, (embeds, next_time_cache) = self.forward(
                latents = latents,
                signal_levels = self.max_steps - 1,
                step_sizes = step_size,
                rewards = rewards,
                discrete_actions = discrete_actions,
                continuous_actions = continuous_actions,
                proprio = accumulated_proprio,
                time_cache = time_cache,
                latent_is_noised = True,
                return_pred_only = True,
                return_intermediates = True
            )

            # time kv cache

            if use_time_cache:
                time_cache = next_time_cache

            # get one agent

            agent_embed = embeds.agent

            one_agent_embed = agent_embed[..., -1:, agent_index, :]

            # values

            value_bins = self.value_head(one_agent_embed)
            value = self.reward_encoder.bins_to_scalar_value(value_bins)

            values = safe_cat((values, value), dim = 1)

            # policy embed

            policy_embed = self.policy_head(one_agent_embed)

            if store_old_action_unembeds:
                acc_policy_embed = safe_cat((acc_policy_embed, policy_embed), dim = 1)

            # sample actions

            sampled_discrete_actions, sampled_continuous_actions = self.action_embedder.sample(policy_embed, pred_head_index = 0, squeeze = True)

            discrete_actions = safe_cat((discrete_actions, sampled_discrete_actions), dim = 1)
            continuous_actions = safe_cat((continuous_actions, sampled_continuous_actions), dim = 1)

            # get the log prob and values for policy optimization

            one_discrete_log_probs, one_continuous_log_probs = self.action_embedder.log_probs(
                policy_embed,
                pred_head_index = 0,
                discrete_targets = sampled_discrete_actions,
                continuous_targets = sampled_continuous_actions,
            )

            discrete_log_probs = safe_cat((discrete_log_probs, one_discrete_log_probs), dim = 1)
            continuous_log_probs = safe_cat((continuous_log_probs, one_continuous_log_probs), dim = 1)

            # pass the sampled action to the environment and get back next state and reward

            env_step_out = env.step((sampled_discrete_actions, sampled_continuous_actions))

            assert len(env_step_out) in [2, 3, 4, 5]

            if len(env_step_out) == 2:
                obs, reward = env_step_out
                terminated = full((batch,), False)
                truncated = full((batch,), False)

            elif len(env_step_out) == 3:
                obs, reward, terminated = env_step_out
                truncated = full((batch,), False)

            elif len(env_step_out) == 4:
                obs, reward, terminated, truncated = env_step_out

            elif len(env_step_out) == 5:
                obs, reward, terminated, truncated, info = env_step_out

            assert 'image' in obs
            if self.has_proprio:
                assert 'proprio' in obs

            # maybe add state entropy bonus

            if self.add_state_entropy_bonus:
                state_pred_token = embeds.state_pred

                state_pred = self.to_state_pred(state_pred_token)

                state_pred_log_variance = state_pred[..., 1].sum()

                reward = reward + state_pred_log_variance * self.state_entropy_bonus_weight

            # update episode lens

            episode_lens = torch.where(is_terminated, episode_lens, episode_lens + 1)

            # update `is_terminated`

            # (1) - environment says it is terminated
            # (2) - previous step is truncated (this step is for bootstrap value)

            is_terminated |= (terminated | is_truncated)

            # update `is_truncated`

            if step_index <= max_timesteps:
                is_truncated |= truncated

            if step_index == max_timesteps:
                # if the step index is at the max time step allowed, set the truncated flag, if not already terminated

                is_truncated |= ~is_terminated

            # batch and time dimension

            proprio = obs.get('proprio')
            if env_is_vectorized:
                next_frame = rearrange(obs['image'], 'b c vh vw -> b c 1 vh vw')
                proprio = safe_rearrange(proprio, 'b d -> b 1 d')
                reward = rearrange(reward, 'b -> b 1')
            else:
                next_frame = rearrange(obs['image'], 'c vh vw -> 1 c 1 vh vw')
                proprio = safe_rearrange(proprio, 'd -> 1 1 d')
                reward = rearrange(reward, ' -> 1 1')

            # concat
            video = cat((video, next_frame), dim = 2)
            rewards = safe_cat((rewards, reward), dim = 1)

            accumulated_proprio = safe_cat((accumulated_proprio, proprio), dim = 1)

            acc_agent_embed = safe_cat((acc_agent_embed, one_agent_embed), dim = 1)

        # package up one experience for learning

        batch, device = latents.shape[0], latents.device

        one_experience = Experience(
            latents = latents,
            video = video[:, :, :-1],
            proprio = accumulated_proprio[:, :-1],
            rewards = rewards,
            actions = (discrete_actions, continuous_actions),
            log_probs = (discrete_log_probs, continuous_log_probs),
            values = values,
            old_action_unembeds = self.action_embedder.unembed(acc_policy_embed, pred_head_index = 0) if exists(acc_policy_embed) and store_old_action_unembeds else None,
            agent_embed = acc_agent_embed if store_agent_embed else None,
            step_size = step_size,
            agent_index = agent_index,
            is_truncated = is_truncated,
            lens = episode_lens,
            is_from_world_model = False
        )

        return one_experience

    # ppo

    def learn_from_experience(
        self,
        experience: Experience,
        policy_optim: Optimizer | None = None,
        value_optim: Optimizer | None = None,
        only_learn_policy_value_heads = True, # in the paper, they do not finetune the entire dynamics model, they just learn the heads
        use_pmpo = True,
        normalize_advantages = None,
        eps = 1e-6
    ):
        assert isinstance(experience, Experience)

        experience = experience.to(self.device)

        latents = experience.latents
        actions = experience.actions
        proprio = experience.proprio
        old_log_probs = experience.log_probs
        old_values = experience.values
        rewards = experience.rewards
        agent_embeds = experience.agent_embed
        old_action_unembeds = experience.old_action_unembeds

        step_size = experience.step_size
        agent_index = experience.agent_index

        assert all([*map(exists, (old_log_probs, actions, old_values, rewards, step_size))]), 'the generations need to contain the log probs, values, and rewards for policy optimization - world_model.generate(..., return_log_probs_and_values = True)'

        batch, time = latents.shape[0], latents.shape[1]

        # calculate returns

        # mask out anything after the `lens`, which may include a bootstrapped node at the very end if `is_truncated = True`

        if not exists(experience.is_truncated):
            experience.is_truncated = full((batch,), True, device = latents.device)

        if exists(experience.lens):
            mask_for_gae = lens_to_mask(experience.lens, time)

            rewards = rewards.masked_fill(~mask_for_gae, 0.)
            old_values = old_values.masked_fill(~mask_for_gae, 0.)

        # calculate returns

        returns = calc_gae(rewards, old_values, gamma = self.gae_discount_factor, lam = self.gae_lambda, use_accelerated = self.gae_use_accelerated)

        # handle variable lengths

        max_time = latents.shape[1]
        is_var_len = exists(experience.lens)

        mask = None

        if is_var_len:
            learnable_lens = experience.lens - experience.is_truncated.long() # if is truncated, remove the last one, as it is bootstrapped value
            mask = lens_to_mask(learnable_lens, max_time)

        # determine whether to finetune entire transformer or just learn the heads

        world_model_forward_context = torch.no_grad if only_learn_policy_value_heads else nullcontext

        # maybe keep track returns statistics and normalize returns and values before calculating advantage, as done in dreamer v3

        if self.keep_reward_ema_stats:
            ema_returns_mean, ema_returns_var = self.ema_returns_mean, self.ema_returns_var

            decay = 1. - self.reward_ema_decay

            # quantile filter

            lo, hi = torch.quantile(returns, self.reward_quantile_filter).tolist()
            returns_for_stats = returns.clamp(lo, hi)

            # mean, var - todo - handle distributed

            returns_mean, returns_var = returns_for_stats.mean(), returns_for_stats.var()

            # ema

            ema_returns_mean.lerp_(returns_mean, decay)
            ema_returns_var.lerp_(returns_var, decay)

            # normalize

            ema_returns_std = ema_returns_var.clamp(min = 1e-5).sqrt()

            normed_returns = (returns - ema_returns_mean) / ema_returns_std
            normed_old_values = (old_values - ema_returns_mean) / ema_returns_std

            advantage = normed_returns - normed_old_values
        else:
            advantage = returns - old_values

        # if using pmpo, do not normalize advantages, but can be overridden

        normalize_advantages = default(normalize_advantages, not use_pmpo)

        if normalize_advantages:
            advantage = F.layer_norm(advantage, advantage.shape, eps = eps)

        # https://arxiv.org/abs/2410.04166v1

        if use_pmpo:
            pos_advantage_mask = advantage >= 0.
            neg_advantage_mask = ~pos_advantage_mask

        # replay for the action logits and values
        # but only do so if fine tuning the entire world model for RL

        discrete_actions, continuous_actions = actions

        if (
            not only_learn_policy_value_heads or
            not exists(agent_embeds)
        ):

            with world_model_forward_context():
                _, (embeds, _) = self.forward(
                    latents = latents,
                    signal_levels = self.max_steps - 1,
                    step_sizes = step_size,
                    rewards = rewards,
                    discrete_actions = discrete_actions,
                    continuous_actions = continuous_actions,
                    proprio = proprio,
                    latent_is_noised = True,
                    return_pred_only = True,
                    return_intermediates = True
                )

            agent_embeds = embeds.agent[..., agent_index, :]

        # maybe detach agent embed

        if only_learn_policy_value_heads:
            agent_embeds = agent_embeds.detach()

        # ppo

        policy_embed = self.policy_head(agent_embeds)

        log_probs, entropies = self.action_embedder.log_probs(policy_embed, pred_head_index = 0, discrete_targets = discrete_actions, continuous_targets = continuous_actions, return_entropies = True)

        # concat discrete and continuous actions into one for optimizing

        old_log_probs = safe_cat(old_log_probs, dim = -1)
        log_probs = safe_cat(log_probs, dim = -1)
        entropies = safe_cat(entropies, dim = -1)

        advantage = rearrange(advantage, '... -> ... 1') # broadcast across all actions

        if use_pmpo:
            # pmpo - weighting the positive and negative advantages equally - ignoring magnitude of advantage and taking the sign
            # seems to be weighted across batch and time, iiuc
            # eq (10) in https://arxiv.org/html/2410.04166v1

            if exists(mask):
                pos_advantage_mask &= mask
                neg_advantage_mask &= mask

            α = self.pmpo_pos_to_neg_weight

            pos = masked_mean(log_probs, pos_advantage_mask)
            neg = -masked_mean(log_probs, neg_advantage_mask)

            policy_loss = -(α * pos + (1. - α) * neg)

            # take care of kl

            if self.pmpo_kl_div_loss_weight > 0.:

                new_unembedded_actions = self.action_embedder.unembed(policy_embed, pred_head_index = 0)

                kl_div_inputs, kl_div_targets = new_unembedded_actions, old_action_unembeds

                # mentioned that the "reverse direction for the prior KL" was used
                # make optional, as observed instability in toy task

                if self.pmpo_reverse_kl:
                    kl_div_inputs, kl_div_targets = kl_div_targets, kl_div_inputs

                discrete_kl_div, continuous_kl_div = self.action_embedder.kl_div(kl_div_inputs, kl_div_targets)

                # accumulate discrete and continuous kl div

                kl_div_loss = 0.

                if exists(discrete_kl_div):
                    kl_div_loss = kl_div_loss + masked_mean(discrete_kl_div, mask)

                if exists(continuous_kl_div):
                    kl_div_loss = kl_div_loss + masked_mean(continuous_kl_div, mask)

                policy_loss = policy_loss + kl_div_loss * self.pmpo_kl_div_loss_weight

        else:
            # ppo clipped surrogate loss

            ratio = (log_probs - old_log_probs).exp()
            clipped_ratio = ratio.clamp(1. - self.ppo_eps_clip, 1. + self.ppo_eps_clip)

            policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
            policy_loss = reduce(policy_loss, 'b t na -> b t', 'sum')

            policy_loss = masked_mean(policy_loss, mask)

        # handle entropy loss for naive exploration bonus

        entropy_loss = - reduce(entropies, 'b t na -> b t', 'sum')

        entropy_loss = masked_mean(entropy_loss, mask)

        # total policy loss

        total_policy_loss = (
            policy_loss +
            entropy_loss * self.policy_entropy_weight
        )

        # maybe take policy optimizer step

        if exists(policy_optim):
            total_policy_loss.backward()

            policy_optim.step()
            policy_optim.zero_grad()

        # value loss

        value_bins = self.value_head(agent_embeds)
        values = self.reward_encoder.bins_to_scalar_value(value_bins)

        clipped_values = old_values + (values - old_values).clamp(-self.value_clip, self.value_clip)
        clipped_value_bins = self.reward_encoder(clipped_values)

        return_bins = self.reward_encoder(returns)

        value_bins, return_bins, clipped_value_bins = tuple(rearrange(t, 'b t l -> b l t') for t in (value_bins, return_bins, clipped_value_bins))

        value_loss_1 = F.cross_entropy(value_bins, return_bins, reduction = 'none')
        value_loss_2 = F.cross_entropy(clipped_value_bins, return_bins, reduction = 'none')

        value_loss = torch.maximum(value_loss_1, value_loss_2)

        # maybe variable length

        if is_var_len:
            value_loss = value_loss[mask].mean()
        else:
            value_loss = value_loss.mean()

        # maybe take value optimizer step

        if exists(policy_optim):
            value_loss.backward()

            value_optim.step()
            value_optim.zero_grad()

        return total_policy_loss, value_loss

    @torch.no_grad()
    def generate(
        self,
        time_steps,
        num_steps = 4,
        batch_size = 1,
        agent_index = 0,
        tasks: int | Tensor | None = None,
        latent_gene_ids = None,
        image_height = None,
        image_width = None,
        return_decoded_video = None,
        context_signal_noise = 0.1,       # they do a noising of the past, this was from an old diffusion world modeling paper from EPFL iirc
        time_cache: Tensor | None = None,
        use_time_cache = True,
        return_rewards_per_frame = False,
        return_agent_actions = False,
        return_log_probs_and_values = False,
        return_for_policy_optimization = False,
        return_time_cache = False,
        store_agent_embed = True,
        store_old_action_unembeds = True

    ): # (b t n d) | (b c t h w)

        # handy flag for returning generations for rl

        if return_for_policy_optimization:
            return_agent_actions |= True
            return_log_probs_and_values |= True
            return_rewards_per_frame |= True

        # more variables

        has_proprio = self.has_proprio
        was_training = self.training
        self.eval()

        # validation

        assert log2(num_steps).is_integer(), f'number of steps {num_steps} must be a power of 2'
        assert 0 < num_steps <= self.max_steps, f'number of steps {num_steps} must be between 0 and {self.max_steps}'

        if isinstance(tasks, int):
            tasks = full((batch_size,), tasks, device = self.device)

        assert not exists(tasks) or tasks.shape[0] == batch_size

        # get state latent shape

        latent_shape = self.latent_shape

        # derive step size

        step_size = self.max_steps // num_steps

        # denoising
        # teacher forcing to start with

        latents = empty((batch_size, 0, self.num_video_views, *latent_shape), device = self.device)

        past_latents_context_noise = latents.clone()

        # maybe internal state

        if has_proprio:
            proprio = empty((batch_size, 0, self.dim_proprio), device = self.device)

            past_proprio_context_noise = proprio.clone()

        # maybe return actions

        return_agent_actions |= return_log_probs_and_values

        decoded_discrete_actions = None
        decoded_continuous_actions = None

        # policy optimization related

        decoded_discrete_log_probs = None
        decoded_continuous_log_probs = None
        decoded_values = None

        # maybe store agent embed

        acc_agent_embed = None

        # maybe store old actions for kl

        acc_policy_embed = None

        # maybe return rewards

        decoded_rewards = None
        if return_rewards_per_frame:
            decoded_rewards = empty((batch_size, 0), device = self.device, dtype = torch.float32)

        # while all the frames of the video (per latent) is not generated

        while latents.shape[1] < time_steps:

            curr_time_steps = latents.shape[1]

            # determine whether to take an extra step if
            # (1) using time kv cache
            # (2) decoding anything off agent embedding (rewards, actions, etc)

            take_extra_step = (
                use_time_cache or
                return_rewards_per_frame or
                store_agent_embed or
                return_agent_actions
            )

            # prepare noised latent / proprio inputs

            noised_latent = randn((batch_size, 1, self.num_video_views, *latent_shape), device = self.device)

            noised_proprio = None

            if has_proprio:
                noised_proprio = randn((batch_size, 1, self.dim_proprio), device = self.device)

            # denoising steps

            for step in range(num_steps + int(take_extra_step)):

                is_last_step = (step + 1) == num_steps

                signal_levels = full((batch_size, 1), step * step_size, dtype = torch.long, device = self.device)

                # noising past latent context

                noised_context = latents.lerp(past_latents_context_noise, context_signal_noise) # the paragraph after eq (8)

                noised_latent_with_context, pack_context_shape = pack((noised_context, noised_latent), 'b * v n d')

                # handle proprio

                noised_proprio_with_context = None

                if has_proprio:
                    noised_proprio_context = proprio.lerp(past_proprio_context_noise, context_signal_noise)
                    noised_proprio_with_context, _ = pack((noised_proprio_context, noised_proprio), 'b * d')

                # proper signal levels

                signal_levels_with_context = F.pad(signal_levels, (curr_time_steps, 0), value = self.max_steps - 1)

                pred, (embeds, next_time_cache) = self.forward(
                    latents = noised_latent_with_context,
                    signal_levels = signal_levels_with_context,
                    step_sizes = step_size,
                    rewards = decoded_rewards,
                    tasks = tasks,
                    latent_gene_ids = latent_gene_ids,
                    discrete_actions = decoded_discrete_actions,
                    continuous_actions = decoded_continuous_actions,
                    proprio = noised_proprio_with_context,
                    time_cache = time_cache,
                    latent_is_noised = True,
                    latent_has_view_dim = True,
                    return_pred_only = True,
                    return_intermediates = True,
                )

                if use_time_cache and is_last_step:
                    time_cache = next_time_cache

                # early break if taking an extra step for agent embedding off cleaned latents for decoding

                if take_extra_step and is_last_step:
                    break

                # maybe proprio

                # maybe proprio

                pred_proprio = pred.proprioception
                pred = pred.flow

                # unpack pred

                _, pred = unpack(pred, pack_context_shape, 'b * v n d')

                if has_proprio:
                    _, pred_proprio = unpack(pred_proprio, pack_context_shape, 'b * d')

                # derive flow, based on whether in x-space or not

                def denoise_step(pred, noised, signal_levels):
                    if self.pred_orig_latent:
                        times = self.get_times_from_signal_level(signal_levels)
                        aligned_times, _ = align_dims_left((times,), noised)

                        flow = (pred - noised) / (1. - aligned_times)
                    else:
                        flow = pred

                    return flow * (step_size / self.max_steps)

                # denoise

                noised_latent += denoise_step(pred, noised_latent, signal_levels)

                if has_proprio:
                    noised_proprio += denoise_step(pred_proprio, noised_proprio, signal_levels)

            denoised_latent = noised_latent # it is now denoised

            if has_proprio:
                denoised_proprio = noised_proprio

            # take care of the rewards by predicting on the agent token embedding on the last denoising step

            if return_rewards_per_frame:
                agent_embed = embeds.agent

                one_agent_embed = agent_embed[:, -1:, agent_index]

                reward_logits = self.to_reward_pred.forward_one(one_agent_embed, id = 0)
                pred_reward = self.reward_encoder.bins_to_scalar_value(reward_logits, normalize = True)

                decoded_rewards = cat((decoded_rewards, pred_reward), dim = 1)

            # maybe store agent embed

            if store_agent_embed:
                agent_embed = embeds.agent

                one_agent_embed = agent_embed[:, -1:, agent_index]
                acc_agent_embed = safe_cat((acc_agent_embed, one_agent_embed), dim = 1)

            # decode the agent actions if needed

            if return_agent_actions:
                assert self.action_embedder.has_actions

                one_agent_embed = agent_embed[:, -1:, agent_index]

                policy_embed = self.policy_head(one_agent_embed)

                # maybe store old actions

                if store_old_action_unembeds:
                    acc_policy_embed = safe_cat((acc_policy_embed, policy_embed), dim = 1)

                # sample actions

                sampled_discrete_actions, sampled_continuous_actions = self.action_embedder.sample(policy_embed, pred_head_index = 0, squeeze = True)

                decoded_discrete_actions = safe_cat((decoded_discrete_actions, sampled_discrete_actions), dim = 1)
                decoded_continuous_actions = safe_cat((decoded_continuous_actions, sampled_continuous_actions), dim = 1)

                if return_log_probs_and_values:
                    discrete_log_probs, continuous_log_probs = self.action_embedder.log_probs(
                        policy_embed,
                        pred_head_index = 0,
                        discrete_targets = sampled_discrete_actions,
                        continuous_targets = sampled_continuous_actions,
                    )

                    decoded_discrete_log_probs = safe_cat((decoded_discrete_log_probs, discrete_log_probs), dim = 1)
                    decoded_continuous_log_probs = safe_cat((decoded_continuous_log_probs, continuous_log_probs), dim = 1)

                    value_bins = self.value_head(one_agent_embed)
                    values = self.reward_encoder.bins_to_scalar_value(value_bins)

                    decoded_values = safe_cat((decoded_values, values), dim = 1)

            # concat the denoised latent

            latents = cat((latents, denoised_latent), dim = 1)

            # add new fixed context noise for the temporal consistency

            past_latents_context_noise = cat((past_latents_context_noise, randn_like(denoised_latent)), dim = 1)

            # handle proprio

            if has_proprio:
                proprio = cat((proprio, denoised_proprio), dim = 1)

                past_proprio_context_noise = cat((past_proprio_context_noise, randn_like(denoised_proprio)), dim = 1)

        # restore state

        self.train(was_training)

        # returning video

        has_tokenizer = exists(self.video_tokenizer)
        return_decoded_video = default(return_decoded_video, has_tokenizer)

        video = None

        if return_decoded_video:

            latents_for_video = rearrange(latents, 'b t v n d -> b v t n d')
            latents_for_video, unpack_view = pack_one(latents_for_video, '* t n d')

            video = self.video_tokenizer.decode(
                latents_for_video,
                height = image_height,
                width = image_width
            )

            video = unpack_view(video, '* t c vh vw')

        # remove the lone view dimension

        if not self.video_has_multi_view:
            latents = rearrange(latents, 'b t 1 ... -> b t ...')

            if exists(video):
                video = rearrange(video, 'b 1 ... -> b ...')

        # only return video or latent if not requesting anything else, for first stage training

        if not has_at_least_one(return_rewards_per_frame, return_agent_actions, has_proprio):
            out = video if return_decoded_video else latents

            if not return_time_cache:
                return out

            return out, time_cache

        # returning agent actions, rewards, and log probs + values for policy optimization

        batch, device = latents.shape[0], latents.device
        experience_lens = full((batch,), time_steps, device = device)

        gen = Experience(
            latents = latents,
            video = video,
            proprio = proprio if has_proprio else None,
            agent_embed = acc_agent_embed if store_agent_embed else None,
            old_action_unembeds = self.action_embedder.unembed(acc_policy_embed, pred_head_index = 0) if exists(acc_policy_embed) and store_old_action_unembeds else None,
            step_size = step_size,
            agent_index = agent_index,
            lens = experience_lens,
            is_from_world_model = True
        )

        if return_rewards_per_frame:
            gen.rewards = decoded_rewards

        if return_agent_actions:
            gen.actions = (decoded_discrete_actions, decoded_continuous_actions)

        if return_log_probs_and_values:
            gen.log_probs = (decoded_discrete_log_probs, decoded_continuous_log_probs)

            gen.values = decoded_values

        if not return_time_cache:
            return gen

        return gen, time_cache

    def forward(
        self,
        *,
        video = None,                    # (b v? c t vh vw)
        latents = None,                  # (b t v? n d) | (b t v? d)
        lens = None,                     # (b)
        signal_levels = None,            # () | (b) | (b t)
        step_sizes = None,               # () | (b)
        step_sizes_log2 = None,          # () | (b)
        latent_gene_ids = None,          # (b)
        tasks = None,                    # (b)
        rewards = None,                  # (b t)
        discrete_actions = None,         # (b t na) | (b t-1 na)
        continuous_actions = None,       # (b t na) | (b t-1 na)
        discrete_action_types = None,    # (na)
        continuous_action_types = None,  # (na)
        proprio = None,                  # (b t dp)
        time_cache = None,
        return_pred_only = False,
        latent_is_noised = False,
        return_all_losses = False,
        return_intermediates = False,
        add_autoregressive_action_loss = True,
        update_loss_ema = None,
        latent_has_view_dim = False
    ):
        # handle video or latents

        assert exists(video) ^ exists(latents)

        # standardize view dimension

        if not self.video_has_multi_view:
            if exists(video):
                video = rearrange(video, 'b ... -> b 1 ...')

            if exists(latents) and not latent_has_view_dim:
                latents = rearrange(latents, 'b t ... -> b t 1 ...')

        # if raw video passed in, tokenize

        if exists(video):
            assert video.ndim == 6

            video, unpack_views = pack_one(video, '* c t vh vw')
            assert exists(self.video_tokenizer), 'video_tokenizer must be passed in if training from raw video on dynamics model'

            latents = self.video_tokenizer.tokenize(video)
            latents = unpack_views(latents, '* t n d')
            latents = rearrange(latents, 'b v t n d -> b t v n d')

        if latents.ndim == 4:
            latents = rearrange(latents, 'b t v d -> b t v 1 d') # 1 latent edge case

        assert latents.shape[-2:] == self.latent_shape, f'latents must have shape {self.latent_shape}, got {latents.shape[-2:]}'
        assert latents.shape[2] == self.num_video_views, f'latents must have {self.num_video_views} views, got {latents.shape[2]}'

        # variables

        batch, time, device = *latents.shape[:2], latents.device

        # signal and step size related input conforming

        if exists(signal_levels):
            if isinstance(signal_levels, int):
                signal_levels = tensor(signal_levels, device = self.device)

            if signal_levels.ndim == 0:
                signal_levels = repeat(signal_levels, '-> b', b = batch)

            if signal_levels.ndim == 1:
                signal_levels = repeat(signal_levels, 'b -> b t', t = time)

        if exists(step_sizes):
            if isinstance(step_sizes, int):
                step_sizes = tensor(step_sizes, device = self.device)

            if step_sizes.ndim == 0:
                step_sizes = repeat(step_sizes, '-> b', b = batch)

        if exists(step_sizes_log2):
            if isinstance(step_sizes_log2, int):
                step_sizes_log2 = tensor(step_sizes_log2, device = self.device)

            if step_sizes_log2.ndim == 0:
                step_sizes_log2 = repeat(step_sizes_log2, '-> b', b = batch)

        # handle step sizes -> step size log2

        assert not (exists(step_sizes) and exists(step_sizes_log2))

        if exists(step_sizes):
            step_sizes_log2_maybe_float = torch.log2(step_sizes)
            step_sizes_log2 = step_sizes_log2_maybe_float.long()

            assert (step_sizes_log2 == step_sizes_log2_maybe_float).all(), f'`step_sizes` must be powers of 2'

        # flow related

        assert not (exists(signal_levels) ^ exists(step_sizes_log2))

        is_inference = exists(signal_levels)
        no_shortcut_train = not is_inference

        return_pred_only = return_pred_only or latent_is_noised

        # if neither signal levels or step sizes passed in, assume training
        # generate them randomly for training

        if not is_inference:

            no_shortcut_train = sample_prob(self.prob_no_shortcut_train)

            if no_shortcut_train:
                # if no shortcut training, step sizes are just 1 and noising is all steps, where each step is 1 / d_min
                # in original shortcut paper, they actually set d = 0 for some reason, look into that later, as there is no mention in the dreamer paper of doing this

                step_sizes_log2 = zeros((batch,), device = device).long() # zero because zero is equivalent to step size of 1
                signal_levels = randint(0, self.max_steps, (batch, time), device = device)
            else:

                # now we follow eq (4)

                step_sizes_log2 = randint(1, self.num_step_sizes_log2, (batch,), device = device)
                num_step_sizes = 2 ** step_sizes_log2

                signal_levels = randint(0, self.max_steps, (batch, time), device = device) // num_step_sizes[:, None] * num_step_sizes[:, None] # times are discretized to step sizes

        # times is from 0 to 1

        times = self.get_times_from_signal_level(signal_levels)

        if not latent_is_noised:
            # get the noise

            noise = randn_like(latents)
            aligned_times, _ = align_dims_left((times,), latents)

            # noise from 0 as noise to 1 as data

            noised_latents = noise.lerp(latents, aligned_times)

        else:
            noised_latents = latents

        # reinforcement learning related

        agent_tokens = repeat(self.agent_learned_embed, '... d -> b ... d', b = batch)

        if exists(tasks):
            assert self.num_tasks > 0

            task_embeds = self.task_embed(tasks)
            agent_tokens = add('b ... d, b d', agent_tokens, task_embeds)

        # maybe evolution

        if exists(latent_gene_ids):
            assert exists(self.latent_genes)
            latent_genes = self.latent_genes[latent_gene_ids]

            agent_tokens = add('b ... d,  b d', agent_tokens, latent_genes)

        # handle agent tokens w/ actions and task embeds

        agent_tokens = repeat(agent_tokens, 'b ... d -> b t ... d', t = time)

        # empty token

        empty_token = agent_tokens[:, :, 0:0]

        # maybe reward tokens

        reward_tokens = empty_token

        if exists(rewards):
            two_hot_encoding = self.reward_encoder(rewards)

            if (
                self.add_reward_embed_to_agent_token and
                (not self.training or not sample_prob(self.add_reward_embed_dropout)) # a bit of noise goes a long way
            ):
                assert self.num_agents == 1

                reward_tokens = self.reward_encoder.embed(two_hot_encoding)

                pop_last_reward = int(reward_tokens.shape[1] == agent_tokens.shape[1]) # the last reward is popped off during training, during inference, it is not known yet, so need to handle this edge case

                reward_tokens = pad_at_dim(reward_tokens, (1, -pop_last_reward), dim = -2, value = 0.)  # shift as each agent token predicts the next reward

                reward_tokens = add('1 d, b t d', self.reward_learned_embed, reward_tokens)

        # maybe proprioception

        assert xnor(self.has_proprio, exists(proprio)), 'proprio must be passed in if `dim_proprio` is set and vice versa'

        noised_proprio = None

        if self.has_proprio:

            if not latent_is_noised:
                # get the noise

                proprio_noise = randn_like(proprio)
                aligned_times, _ = align_dims_left((times,), proprio)

                # noise from 0 as noise to 1 as data

                noised_proprio = proprio_noise.lerp(proprio, aligned_times)

            else:
                noised_proprio = proprio

        # maybe state prediction token

        if self.should_pred_state:
            state_pred_token = repeat(self.state_pred_token, 'd -> b t 1 d', b = batch, t = time)
        else:
            state_pred_token = empty_token

        # maybe create the action tokens

        if exists(discrete_actions) or exists(continuous_actions):
            assert self.action_embedder.has_actions
            assert self.num_agents == 1, 'only one agent allowed for now'

            action_tokens = self.action_embedder(
                discrete_actions = discrete_actions,
                discrete_action_types = discrete_action_types,
                continuous_actions = continuous_actions,
                continuous_action_types = continuous_action_types
            )

            # handle first timestep not having an associated past action

            if action_tokens.shape[1] == (time - 1):
                action_tokens = pad_at_dim(action_tokens, (1, 0), value = 0. , dim = 1)

            action_tokens = add('1 d, b t d', self.action_learned_embed, action_tokens)

        elif self.action_embedder.has_actions:
            action_tokens = torch.zeros_like(agent_tokens[:, :, 0:1])

        else:
            action_tokens = empty_token # else empty off agent tokens

        # main function, needs to be defined as such for shortcut training - additional calls for consistency loss

        def get_prediction(noised_latents, noised_proprio, signal_levels, step_sizes_log2, state_pred_token, action_tokens, reward_tokens, agent_tokens, return_agent_tokens = False, return_time_cache = False):

            # latents to spatial tokens

            space_tokens = self.latents_to_spatial_tokens(noised_latents)

            # maybe add view embedding

            if self.video_has_multi_view:
                space_tokens = add('b t v ... d, v d', space_tokens, self.view_emb)

            # merge spatial tokens

            space_tokens, inverse_pack_space_per_latent = pack_one(space_tokens, 'b t * d')

            num_spatial_tokens = space_tokens.shape[-2]

            # action tokens

            num_action_tokens = 1 if not is_empty(action_tokens) else 0

            # reward tokens

            num_reward_tokens = 1 if not is_empty(reward_tokens) else 0

            # pack to tokens
            # [signal + step size embed] [latent space tokens] [register] [actions / agent]

            registers = repeat(self.register_tokens, 's d -> b t s d', b = batch, t = time)

            # maybe proprio

            if exists(noised_proprio):
                proprio_token = self.to_proprio_token(noised_proprio)
            else:
                proprio_token = registers[:, :, 0:0]

            # determine signal + step size embed for their diffusion forcing + shortcut

            signal_embed = self.signal_levels_embed(signal_levels)

            step_size_embed = self.step_size_embed(step_sizes_log2)
            step_size_embed = repeat(step_size_embed, 'b ... -> b t ...', t = time)

            flow_token = cat((signal_embed, step_size_embed), dim = -1)
            flow_token = rearrange(flow_token, 'b t d -> b t d')

            # pack to tokens for attending

            tokens, packed_tokens_shape = pack([flow_token, space_tokens, proprio_token, state_pred_token, registers, action_tokens, reward_tokens, agent_tokens], 'b t * d')

            # attention

            tokens, intermediates = self.transformer(tokens, cache = time_cache, return_intermediates = True)

            # unpack

            flow_token, space_tokens, proprio_token, state_pred_token, register_tokens, action_tokens, reward_tokens, agent_tokens = unpack(tokens, packed_tokens_shape, 'b t * d')

            # pooling

            space_tokens = inverse_pack_space_per_latent(space_tokens)

            pred = self.to_latent_pred(space_tokens)

            # maybe proprio

            if self.has_proprio:
                pred_proprio = self.to_proprio_pred(proprio_token)
            else:
                pred_proprio = None

            # maybe state pred

            if self.should_pred_state:
                pred_state = self.to_state_pred(state_pred_token)
            else:
                pred_state = None

            # returning

            predictions = Predictions(pred, pred_proprio, pred_state)

            embeds = Embeds(agent_tokens, state_pred_token)

            if not return_agent_tokens:
                return predictions

            if not return_time_cache:
                return predictions, embeds

            return predictions, (embeds, intermediates)

        # curry into get_prediction what does not change during first call as well as the shortcut ones

        _get_prediction = partial(get_prediction, state_pred_token = state_pred_token, action_tokens = action_tokens, reward_tokens = reward_tokens, agent_tokens = agent_tokens)

        # forward the network

        pred, (embeds, intermediates) = _get_prediction(noised_latents, noised_proprio, signal_levels, step_sizes_log2, return_agent_tokens = True, return_time_cache = True)

        if return_pred_only:
            if not return_intermediates:
                return pred

            return pred, (embeds, intermediates)

        # pack the predictions to calculate flow for different modalities all at once

        if self.has_proprio:
            packed_pred, for_flow_loss_packed_shape = pack((pred.flow, pred.proprioception), 'b t *')

            noised, _ = pack((noised_latents, noised_proprio), 'b t *')
            data, _ = pack((latents, proprio), 'b t *')
            noise, _ = pack((noise, proprio_noise), 'b t *')
        else:
            packed_pred = pred.flow
            noised = noised_latents
            data = latents

        # wrapper function for maybe unpacking and packing modalities for doing flow math in unison

        def maybe_pack_unpack(fn):
            @wraps(fn)
            @torch.no_grad()
            def inner(noised, *args, **kwargs):

                noised_proprio = None

                if self.has_proprio:
                    noised, noised_proprio = unpack(noised, for_flow_loss_packed_shape, 'b t *')

                pred = fn(noised, noised_proprio, *args, **kwargs)

                if self.has_proprio:
                    packed_flow, _ = pack((pred.flow, pred.proprioception), 'b t *')
                    return packed_flow

                return pred.flow
            return inner

        wrapped_get_prediction = maybe_pack_unpack(_get_prediction)

        # determine the target for the loss

        pred_target = None

        is_x_space = self.pred_orig_latent
        is_v_space_pred = not self.pred_orig_latent

        maybe_shortcut_loss_weight = 1.

        if no_shortcut_train:

            # allow for original velocity pred
            # x-space as in paper is in else clause

            if is_v_space_pred:
                pred_target = flow = data - noise
            else:
                pred_target = data
        else:
            # shortcut training - Frans et al. https://arxiv.org/abs/2410.12557

            # basically a consistency loss where you ensure quantity of two half steps equals one step
            # dreamer then makes it works for x-space with some math

            step_sizes_log2_minus_one = step_sizes_log2 - 1 # which equals d / 2
            half_step_size = 2 ** step_sizes_log2_minus_one

            first_step_pred = wrapped_get_prediction(noised, signal_levels, step_sizes_log2_minus_one)

            # first derive b'

            if is_v_space_pred:
                first_step_pred_flow = first_step_pred
            else:
                first_times = self.get_times_from_signal_level(signal_levels, noised)

                first_step_pred_flow = (first_step_pred - noised) / (1. - first_times)

            # take a half step

            half_step_size_align_left, _ = align_dims_left((half_step_size,), noised)

            denoised = noised + first_step_pred_flow * (half_step_size_align_left / self.max_steps)

            # get second prediction for b''

            signal_levels_plus_half_step = signal_levels + half_step_size[:, None]
            second_step_pred = wrapped_get_prediction(denoised, signal_levels_plus_half_step, step_sizes_log2_minus_one)

            if is_v_space_pred:
                second_step_pred_flow = second_step_pred
            else:
                second_times = self.get_times_from_signal_level(signal_levels_plus_half_step, denoised)
                second_step_pred_flow = (second_step_pred - denoised) / (1. - second_times)

            # pred target is sg(b' + b'') / 2

            pred_target = (first_step_pred_flow + second_step_pred_flow).detach() / 2

            # need to convert x-space to v-space

            if is_x_space:
                packed_pred = (packed_pred - noised) / (1. - first_times)
                maybe_shortcut_loss_weight = (1. - first_times) ** 2

        # mse loss

        flow_losses = F.mse_loss(packed_pred, pred_target, reduction = 'none')

        flow_losses = flow_losses * maybe_shortcut_loss_weight # handle the (1-t)^2 in eq(7)

        # loss weighting with their ramp function

        if exists(self.loss_weight_fn):
            loss_weight = self.loss_weight_fn(times)
            loss_weight, _ = align_dims_left((loss_weight,), flow_losses)

            flow_losses = flow_losses * loss_weight

        # handle variable lengths if needed

        is_var_len = exists(lens)

        if is_var_len:

            loss_mask = lens_to_mask(lens, time)
            loss_mask_without_last = loss_mask[:, :-1]

            flow_loss = flow_losses[loss_mask].mean()

        else:
            flow_loss = flow_losses.mean()

        # now take care of the agent token losses

        reward_loss = self.zero

        if exists(rewards):

            encoded_agent_tokens = embeds.agent

            if rewards.ndim == 2: # (b t)
                encoded_agent_tokens = reduce(encoded_agent_tokens, 'b t g d -> b t d', 'mean')

            reward_pred = self.to_reward_pred(encoded_agent_tokens[:, :-1])

            reward_pred = rearrange(reward_pred, 'mtp b t l -> b l t mtp')

            reward_targets, reward_loss_mask = create_multi_token_prediction_targets(two_hot_encoding[:, :-1], self.multi_token_pred_len)

            reward_targets = rearrange(reward_targets, 'b t mtp l -> b l t mtp')

            reward_losses = F.cross_entropy(reward_pred, reward_targets, reduction = 'none')

            reward_losses = reward_losses.masked_fill(~reward_loss_mask, 0.)

            if is_var_len:
                reward_loss = reward_losses[loss_mask_without_last].mean(dim = 0)
            else:
                reward_loss = reduce(reward_losses, '... mtp -> mtp', 'mean') # they sum across the prediction steps (mtp dimension) - eq(9)

        # maybe autoregressive state prediction loss

        state_pred_loss = self.zero

        if self.should_pred_state:
            pred_latent, latent_to_pred = pred.state[:, :-1], latents[:, 1:]

            pred_latent_mean, pred_latent_log_var = pred_latent.unbind(dim = -1)
            pred_latent_var = pred_latent_log_var.exp()

            state_pred_loss = F.gaussian_nll_loss(pred_latent_mean, latent_to_pred, var = pred_latent_var)

        # maybe autoregressive action loss

        discrete_action_loss = self.zero
        continuous_action_loss = self.zero

        if (
            self.num_agents == 1 and
            add_autoregressive_action_loss and
            time > 1,
            (exists(discrete_actions) or exists(continuous_actions))
        ):
            assert self.action_embedder.has_actions

            # handle actions having time vs time - 1 length
            # remove the first action if it is equal to time (as it would come from some agent token in the past)

            if exists(discrete_actions) and discrete_actions.shape[1] == time:
                discrete_actions = discrete_actions[:, 1:]

            if exists(continuous_actions) and continuous_actions.shape[1] == time:
                continuous_actions = continuous_actions[:, 1:]

            # only for 1 agent

            agent_tokens = rearrange(agent_tokens, 'b t 1 d -> b t d')
            policy_embed = self.policy_head(agent_tokens[:, :-1])

            # constitute multi token prediction targets

            discrete_action_targets = continuous_action_targets = None

            if exists(discrete_actions):
                discrete_action_targets, discrete_mask = create_multi_token_prediction_targets(discrete_actions, self.multi_token_pred_len)
                discrete_action_targets = rearrange(discrete_action_targets, 'b t mtp ... -> mtp b t ...')
                discrete_mask = rearrange(discrete_mask, 'b t mtp -> mtp b t')

            if exists(continuous_actions):
                continuous_action_targets, continuous_mask = create_multi_token_prediction_targets(continuous_actions, self.multi_token_pred_len)
                continuous_action_targets = rearrange(continuous_action_targets, 'b t mtp ... -> mtp b t ...')
                continuous_mask = rearrange(continuous_mask, 'b t mtp -> mtp b t')

            discrete_log_probs, continuous_log_probs = self.action_embedder.log_probs(
                policy_embed,
                discrete_targets = discrete_action_targets if exists(discrete_actions) else None,
                continuous_targets = continuous_action_targets if exists(continuous_actions) else None
            )

            if exists(discrete_log_probs):
                discrete_log_probs = discrete_log_probs.masked_fill(~discrete_mask[..., None], 0.)

                if is_var_len:
                    discrete_action_losses = rearrange(-discrete_log_probs, 'mtp b t na -> b t na mtp')
                    discrete_action_loss = reduce(discrete_action_losses[loss_mask_without_last], '... mtp -> mtp', 'mean')
                else:
                    discrete_action_loss = reduce(-discrete_log_probs, 'mtp b t na -> mtp', 'mean')

            if exists(continuous_log_probs):
                continuous_log_probs = continuous_log_probs.masked_fill(~continuous_mask[..., None], 0.)

                if is_var_len:
                    continuous_action_losses = rearrange(-continuous_log_probs, 'mtp b t na -> b t na mtp')
                    continuous_action_loss = reduce(continuous_action_losses[loss_mask_without_last], '... mtp -> mtp', 'mean')
                else:
                    continuous_action_loss = reduce(-continuous_log_probs, 'mtp b t na -> mtp', 'mean')

        # handle loss normalization

        losses = WorldModelLosses(flow_loss, reward_loss, discrete_action_loss, continuous_action_loss, state_pred_loss)

        if exists(self.flow_loss_normalizer):
            flow_loss = self.flow_loss_normalizer(flow_loss, update_ema = update_loss_ema)

        if exists(rewards) and exists(self.reward_loss_normalizer):
            reward_loss = self.reward_loss_normalizer(reward_loss, update_ema = update_loss_ema)

        if exists(discrete_actions) and exists(self.discrete_actions_loss_normalizer):
            discrete_action_loss = self.discrete_actions_loss_normalizer(discrete_action_loss, update_ema = update_loss_ema)

        if exists(continuous_actions) and exists(self.continuous_actions_loss_normalizer):
            continuous_action_loss = self.continuous_actions_loss_normalizer(continuous_action_loss, update_ema = update_loss_ema)

        # gather losses - they sum across the multi token prediction steps for rewards and actions - eq (9)

        total_loss = (
            flow_loss * self.latent_flow_loss_weight +
            (reward_loss * self.reward_loss_weight).sum() +
            (discrete_action_loss * self.discrete_action_loss_weight).sum() + 
            (continuous_action_loss * self.continuous_action_loss_weight).sum() +
            (state_pred_loss * self.state_pred_loss_weight)
        )

        if not return_all_losses:
            return total_loss

        return total_loss, losses

