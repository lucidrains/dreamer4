from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, Embedding
from torch import Tensor, cat, stack, arange, tensor, is_tensor, ones, rand, randn, randn_like

from einops import rearrange, reduce, pack, einsum

import einx
from einx import add, multiply
from torch.distributions import Normal, kl
from torch.nested import nested_tensor
from assoc_scan import AssocScan
from discrete_continuous_embed_readout import MultiCategorical

# Local imports
from .utils import (
    calc_gae,
    default,
    ensure_tuple,
    exists,
    gumbel_sample,
    log,
    mean_log_var_to_distr,
    pack_one,
    safe_squeeze_first,
)

class ActionEmbedder(Module):
    def __init__(
        self,
        dim,
        *,
        num_discrete_actions: int | tuple[int, ...] = 0,
        num_continuous_actions  = 0,
        continuous_norm_stats: tuple[tuple[float, float], ...] | None = None,
        parallel_discrete_calc = True,
        can_unembed = False,
        unembed_dim = None,
        num_unembed_preds = 1,
        squeeze_unembed_preds = True # will auto-squeeze if prediction is just 1
    ):
        super().__init__()

        # handle discrete actions

        num_discrete_actions = tensor(ensure_tuple(num_discrete_actions))
        total_discrete_actions = num_discrete_actions.sum().item()

        self.num_discrete_action_types = len(num_discrete_actions)
        self.discrete_action_embed = Embedding(total_discrete_actions, dim)

        self.register_buffer('num_discrete_actions', num_discrete_actions, persistent = False)

        # continuous actions

        self.num_continuous_action_types = num_continuous_actions
        self.continuous_action_embed = Embedding(num_continuous_actions, dim)

        self.continuous_need_norm = exists(continuous_norm_stats)

        if self.continuous_need_norm:
            self.register_buffer('continuous_norm_stats', tensor(continuous_norm_stats))

        # defaults

        self.register_buffer('default_discrete_action_types', arange(self.num_discrete_action_types), persistent = False)
        self.register_buffer('default_continuous_action_types', arange(self.num_continuous_action_types), persistent = False)

        # calculate offsets

        offsets = F.pad(num_discrete_actions.cumsum(dim = -1), (1, -1), value = 0)
        self.register_buffer('discrete_action_offsets', offsets, persistent = False)

        # unembedding

        self.can_unembed = can_unembed

        self.num_unembed_preds = num_unembed_preds
        self.squeeze_unembed_preds = squeeze_unembed_preds

        if not can_unembed:
            return

        unembed_dim = default(unembed_dim, dim)
        self.discrete_action_unembed = Parameter(torch.randn(total_discrete_actions, num_unembed_preds, unembed_dim) * 1e-2)

        discrete_action_index = arange(total_discrete_actions)

        padded_num_discrete_actions = F.pad(num_discrete_actions, (1, 0), value = 0)
        exclusive_cumsum = padded_num_discrete_actions.cumsum(dim = -1)

        discrete_action_mask = (
            einx.greater_equal('j, i -> i j', discrete_action_index, exclusive_cumsum[:-1]) &
            einx.less('j, i -> i j', discrete_action_index, exclusive_cumsum[1:])
        )

        self.register_buffer('discrete_action_mask', discrete_action_mask, persistent = False)

        self.continuous_action_unembed = Parameter(torch.randn(num_continuous_actions, num_unembed_preds, unembed_dim, 2) * 1e-2)

    def embed_parameters(self):
        return set([*self.discrete_action_embed.parameters(), *self.continuous_action_embed.parameters()])

    def unembed_parameters(self):
        return set([self.discrete_action_unembed, self.continuous_action_unembed])

    @property
    def device(self):
        return self.discrete_action_offsets.device

    @property
    def has_actions(self):
        return self.num_discrete_action_types > 0 or self.num_continuous_action_types > 0

    def cast_action_types(
        self,
        action_types = None
    ):
        if exists(action_types) and not is_tensor(action_types):
            if isinstance(action_types, int):
                action_types = (action_types,)

            action_types = tensor(action_types, device = self.device)

        return action_types

    def unembed(
        self,
        embeds,                          # (... d)
        discrete_action_types = None,    # (na)
        continuous_action_types = None,  # (na)
        return_split_discrete = False,
        pred_head_index: int | Tensor | None = None

    ):  # (... discrete_na), (... continuous_na 2)

        device = embeds.device

        assert self.can_unembed, 'can only unembed for predicted discrete and continuous actions if `can_unembed = True` is set on init'

        # handle only one prediction head during inference

        if exists(pred_head_index) and isinstance(pred_head_index, int):
            pred_head_index = tensor(pred_head_index, device = device)

        # if pred_head_index given as a solo int, just assume we want to squeeze out the prediction head dimension

        squeeze_one_pred_head = exists(pred_head_index) and pred_head_index.ndim == 0

        # get action types

        discrete_action_types, continuous_action_types = tuple(self.cast_action_types(t) for t in (discrete_action_types, continuous_action_types))

        # discrete actions

        discrete_action_logits = None

        if self.num_discrete_action_types > 0:

            discrete_action_unembed = self.discrete_action_unembed

            if exists(discrete_action_types):
                discrete_action_mask = self.discrete_action_mask[discrete_action_types].any(dim = 0)

                discrete_action_unembed = discrete_action_unembed[discrete_action_mask]

            if exists(pred_head_index):
                discrete_action_unembed = discrete_action_unembed.index_select(1, pred_head_index)

            discrete_action_logits = einsum(embeds, discrete_action_unembed, '... d, na mtp d -> mtp ... na')

            if self.squeeze_unembed_preds or squeeze_one_pred_head:
                discrete_action_logits = safe_squeeze_first(discrete_action_logits)

        # whether to split the discrete action logits by the number of actions per action type

        if exists(discrete_action_logits) and return_split_discrete:

            split_sizes = self.num_discrete_actions[discrete_action_types] if exists(discrete_action_types) else self.num_discrete_actions

            discrete_action_logits = discrete_action_logits.split(split_sizes.tolist(), dim = -1)

        # continuous actions

        continuous_action_mean_log_var = None

        if self.num_continuous_action_types > 0:

            continuous_action_unembed = self.continuous_action_unembed

            if exists(continuous_action_types):
                continuous_action_unembed = continuous_action_unembed[continuous_action_types]

            if exists(pred_head_index):
                continuous_action_unembed = continuous_action_unembed.index_select(1, pred_head_index)

            continuous_action_mean_log_var = einsum(embeds, continuous_action_unembed, '... d, na mtp d two -> mtp ... na two')

            if self.squeeze_unembed_preds or squeeze_one_pred_head:
                continuous_action_mean_log_var = safe_squeeze_first(continuous_action_mean_log_var)

        return discrete_action_logits, continuous_action_mean_log_var

    def sample(
        self,
        embed,
        discrete_temperature = 1.,
        continuous_temperature = 1.,
        inverse_norm_continuous = None,
        pred_head_index: int | Tensor | None = None,
        parallel_discrete_calc = True,
        squeeze = True,
        **kwargs
    ):
        inverse_norm_continuous = default(inverse_norm_continuous, self.continuous_need_norm)

        discrete_logits, continuous_mean_log_var = self.unembed(embed, return_split_discrete = True, pred_head_index = pred_head_index, **kwargs)

        sampled_discrete = sampled_continuous = None

        if exists(discrete_logits):
            dist = MultiCategorical(discrete_logits, use_parallel_multi_discrete = parallel_discrete_calc)
            sampled_discrete = dist.sample(temperature = discrete_temperature)

        if exists(continuous_mean_log_var):
            mean, log_var = continuous_mean_log_var.unbind(dim = -1)
            std = (0.5 * log_var).exp()

            sampled_continuous = mean + std * torch.randn_like(mean) * continuous_temperature

            # maybe inverse norm

            if inverse_norm_continuous:
                norm_mean, norm_std = self.continuous_norm_stats.unbind(dim = -1)
                sampled_continuous = (sampled_continuous * norm_std) + norm_mean

        return sampled_discrete, sampled_continuous

    def log_probs(
        self,
        embeds,                          # (... d)
        discrete_targets = None,         # (... na)
        continuous_targets = None,       # (... na)
        discrete_action_types = None,    # (na)
        continuous_action_types = None,  # (na)
        pred_head_index: int | Tensor | None = None,
        parallel_discrete_calc = None,
        return_entropies = False
    ):
        discrete_action_logits, continuous_action_mean_log_var = self.unembed(
            embeds,
            pred_head_index = pred_head_index,
            discrete_action_types = discrete_action_types,
            continuous_action_types = continuous_action_types,
            return_split_discrete = True
        )
        # discrete

        discrete_log_probs = None
        discrete_entropies = None

        if exists(discrete_targets):
            if not exists(pred_head_index) and self.num_unembed_preds > 1:
                # if multiple heads and no index, broadcast targets to mtp dim
                if discrete_targets.ndim == (discrete_action_logits[0].ndim - 1):
                    discrete_targets = rearrange(discrete_targets, '... -> 1 ...')
                    
            dist = MultiCategorical(discrete_action_logits, use_parallel_multi_discrete = parallel_discrete_calc)
            discrete_log_probs = dist.log_prob(discrete_targets)

            if return_entropies:
                    discrete_entropies = dist.entropy()
        # continuous

        continuous_log_probs = None
        continuous_entropies = None

        if exists(continuous_targets):
            if not exists(pred_head_index) and self.num_unembed_preds > 1:
                # if multiple heads and no index, broadcast targets to mtp dim
                if continuous_targets.ndim == (continuous_action_mean_log_var.ndim - 1):
                    continuous_targets = rearrange(continuous_targets, '... -> 1 ...')

            distr = mean_log_var_to_distr(continuous_action_mean_log_var)
            continuous_log_probs = distr.log_prob(continuous_targets)

            if return_entropies:
                continuous_entropies = distr.entropy()

        log_probs = (discrete_log_probs, continuous_log_probs)

        if not return_entropies:
            return log_probs

        entropies = (discrete_entropies, continuous_entropies)

        return log_probs, entropies

    def kl_div(
        self,
        src: tuple[MaybeTensor, MaybeTensor],
        tgt: tuple[MaybeTensor, MaybeTensor],
        reduce_across_num_actions = True
    ) -> tuple[MaybeTensor, MaybeTensor]:

        src_logits, src_params = src
        tgt_logits, tgt_params = tgt

        # discrete kl

        discrete_kl = None

        if exists(src_logits) and exists(tgt_logits):
            src_dist = MultiCategorical(src_logits, use_parallel_multi_discrete = True)
            tgt_dist = MultiCategorical(tgt_logits, use_parallel_multi_discrete = True)

            discrete_kl = src_dist.kl_div(tgt_dist)


            # MultiCategorical.kl_div already returns a reduced tensor across actions
            # so we should not sum it again if it is already reduced

        # continuous kl

        continuous_kl = None

        if exists(src_params) and exists(tgt_params):
            src_distr = mean_log_var_to_distr(src_params)
            tgt_distr = mean_log_var_to_distr(tgt_params)

            continuous_kl = kl.kl_divergence(src_distr, tgt_distr)

            if reduce_across_num_actions:
                continuous_kl = continuous_kl.sum(dim = -1)

        return discrete_kl, continuous_kl

    def forward(
        self,
        *,
        discrete_actions = None,         # (... na)
        continuous_actions = None,       # (... na)
        discrete_action_types = None,    # (na)
        continuous_action_types = None,  # (na)
        return_sum_pooled_embeds = True
    ):

        discrete_embeds = continuous_embeds = None

        if exists(discrete_actions):

            discrete_action_types = default(discrete_action_types, self.default_discrete_action_types)

            discrete_action_types = self.cast_action_types(discrete_action_types)

            offsets = self.discrete_action_offsets[discrete_action_types]

            assert offsets.shape[-1] == discrete_actions.shape[-1], 'mismatched number of discrete actions'

            # offset the discrete actions based on the action types passed in (by default all discrete actions) and the calculated offset

            discrete_actions_offsetted = add('... na, na', discrete_actions, offsets)
            discrete_embeds = self.discrete_action_embed(discrete_actions_offsetted)

        if exists(continuous_actions):
            continuous_action_types = default(continuous_action_types, self.default_continuous_action_types)

            continuous_action_types = self.cast_action_types(continuous_action_types)

            assert continuous_action_types.shape[-1] == continuous_actions.shape[-1], 'mismatched number of continuous actions'

            continuous_action_embed = self.continuous_action_embed(continuous_action_types)

            # maybe normalization

            if self.continuous_need_norm:
                norm_mean, norm_std = self.continuous_norm_stats.unbind(dim = -1)
                continuous_actions = (continuous_actions - norm_mean) / norm_std.clamp(min = 1e-6)

            # continuous embed is just the selected continuous action type with the scale

            continuous_embeds = multiply('na d, ... na -> ... na d', continuous_action_embed, continuous_actions)

        # return not pooled

        if not return_sum_pooled_embeds:
            return ActionEmbeds(discrete_embeds, continuous_embeds)

        # handle sum pooling, which is what they did in the paper for all the actions

        pooled = 0.

        if exists(discrete_embeds):
            pooled = pooled + reduce(discrete_embeds, '... na d -> ... d', 'sum')

        if exists(continuous_embeds):
            pooled = pooled + reduce(continuous_embeds, '... na d -> ... d', 'sum')

        return pooled

# generalized advantage estimate

@torch.no_grad()
def calc_gae(
    rewards,
    values,
    masks = None,
    gamma = 0.99,
    lam = 0.95,
    use_accelerated = None
):
    assert values.shape[-1] == rewards.shape[-1]
    use_accelerated = default(use_accelerated, rewards.is_cuda)

    if not exists(masks):
        masks = torch.ones_like(values)

    values = F.pad(values, (0, 1), value = 0.)
    values, values_next = values[..., :-1], values[..., 1:]

    delta = rewards + gamma * values_next * masks - values
    gates = gamma * lam * masks

    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)

    gae = scan(gates, delta)

    returns = gae + values

    return returns

# rotary embeddings for time

