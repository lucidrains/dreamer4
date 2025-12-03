from __future__ import annotations
from typing import Callable

import math
from math import ceil, log2
from random import random
from contextlib import nullcontext
from collections import namedtuple
from functools import partial, wraps
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from torch.nested import nested_tensor
from torch.distributions import Normal, kl
from torch.nn import Module, ModuleList, Embedding, Parameter, Sequential, Linear, RMSNorm, Identity
from torch import nn, cat, stack, arange, tensor, Tensor, is_tensor, full, zeros, ones, randint, rand, randn, randn_like, empty, full, linspace, arange
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten

import torchvision
from torchvision.models import VGG16_Weights

from torch.optim import Optimizer
from adam_atan2_pytorch import MuonAdamAtan2

from x_mlps_pytorch.ensemble import Ensemble
from x_mlps_pytorch.normed_mlp import create_mlp

from hyper_connections import get_init_and_expand_reduce_stream_functions

from vit_pytorch.vit_with_decorr import DecorrelationLoss

from assoc_scan import AssocScan

# ein related

# b - batch
# n - sequence
# h - attention heads
# d - feature dimension
# f - frequencies (rotary)
# l - logit / predicted bins
# y - layers of transformer
# p - positions (3 for spacetime in this work)
# t - time
# na - action dimension (number of discrete and continuous actions)
# g - groups of query heads to key heads (gqa)
# vc - video channels
# vh, vw - video height and width
# mtp - multi token prediction length
# v - video viewpoints

import einx
from einx import add, multiply
from einops import einsum, rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

# flex attention - but will make sure it works if it is not available
# may also end up crafting own custom flash attention kernel for this work

flex_attention = None

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass

# constants

LinearNoBias = partial(Linear, bias = False)

TokenizerLosses = namedtuple('TokenizerLosses', ('recon', 'lpips', 'time_decorr', 'space_decorr'))

WorldModelLosses = namedtuple('WorldModelLosses', ('flow', 'rewards', 'discrete_actions', 'continuous_actions', 'state_pred'))

AttentionIntermediates = namedtuple('AttentionIntermediates', ('next_kv_cache', 'normed_inputs'))

TransformerIntermediates = namedtuple('TransformerIntermediates', ('next_kv_cache', 'normed_time_inputs', 'normed_space_inputs', 'next_rnn_hiddens'))

Predictions = namedtuple('Predictions', ('flow', 'proprioception', 'state'))

Embeds = namedtuple('Embeds', ['agent', 'state_pred'])

MaybeTensor = Tensor | None

@dataclass
class Experience:
    latents: Tensor
    video: MaybeTensor = None
    proprio: MaybeTensor = None
    agent_embed: MaybeTensor = None
    rewards: Tensor | None = None
    actions: tuple[MaybeTensor, MaybeTensor] | None = None
    log_probs: tuple[MaybeTensor, MaybeTensor] | None = None
    old_action_unembeds: tuple[MaybeTensor, MaybeTensor] | None = None
    values: MaybeTensor = None
    step_size: int | None = None
    lens: MaybeTensor = None
    is_truncated: MaybeTensor = None
    agent_index: int = 0
    is_from_world_model: bool | Tensor = True

    def cpu(self):
        return self.to(torch.device('cpu'))

    def to(self, device):
        experience_dict = asdict(self)
        experience_dict = tree_map(lambda t: t.to(device) if is_tensor(t) else t, experience_dict)
        return Experience(**experience_dict)

def combine_experiences(
    exps: list[Experiences]
) -> Experience:

    assert len(exps) > 0

    # set lens if not there

    for exp in exps:
        latents = exp.latents
        batch, time, device = *latents.shape[:2], latents.device

        if not exists(exp.lens):
            exp.lens = full((batch,), time, device = device)

        if not exists(exp.is_truncated):
            exp.is_truncated = full((batch,), True, device = device)

        if isinstance(exp.is_from_world_model, bool):
            exp.is_from_world_model = tensor(exp.is_from_world_model)

    # convert to dictionary

    exps_dict = [asdict(exp) for exp in exps]

    values, tree_specs = zip(*[tree_flatten(exp_dict) for exp_dict in exps_dict])

    tree_spec = first(tree_specs)

    all_field_values = list(zip(*values))

    # an assert to make sure all fields are either all tensors, or a single matching value (for step size, agent index etc) - can change this later

    assert all([
        all([is_tensor(v) for v in field_values]) or len(set(field_values)) == 1
        for field_values in all_field_values
    ])

    concatted = []

    for field_values in all_field_values:

        first_value = first(field_values)

        if is_tensor(first_value):

            field_values = pad_tensors_at_dim_to_max_len(field_values, dims = (1, 2))

            cat_or_stack = cat if first_value.ndim > 0 else stack

            new_field_value = cat_or_stack(field_values)
        else:
            new_field_value = first(list(set(field_values)))

        concatted.append(new_field_value)

    # return experience

    concat_exp_dict = tree_unflatten(concatted, tree_spec)

    return Experience(**concat_exp_dict)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(arr):
    return arr[0]

def xnor(x, y):
    return not (x ^ y)

def has_at_least_one(*bools):
    return sum([*map(int, bools)]) > 0

def ensure_tuple(t):
    return (t,) if not isinstance(t, tuple) else t

def divisible_by(num, den):
    return (num % den) == 0

def sample_prob(prob):
    return random() < prob

def is_power_two(num):
    return log2(num).is_integer()

def maybe(fn):
    def inner(t, *args, **kwargs):
        if not exists(t) or not exists(fn):
            return None
        return fn(t)
    return inner

# tensor helpers

def is_empty(t):
    return t.numel() == 0

def lens_to_mask(t, max_len = None):
    if not exists(max_len):
        max_len = t.amax().item()

    device = t.device
    seq = torch.arange(max_len, device = device)

    return einx.less('j, i -> i j', seq, t)

def masked_mean(t, mask = None):
    if not exists(mask):
        return t.mean()

    if not mask.any():
        return t[mask].sum()

    return t[mask].mean()

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def mean_log_var_to_distr(
    mean_log_var: Tensor
) -> Normal:

    mean, log_var = mean_log_var.unbind(dim = -1)
    std = (0.5 * log_var).exp()
    return Normal(mean, std)

def safe_stack(tensors, dim = 0):
    tensors = [*filter(exists, tensors)]

    if len(tensors) == 0:
        return None

    return stack(tensors, dim = dim)

def safe_cat(tensors, dim):
    tensors = [*filter(exists, tensors)]

    if len(tensors) == 0:
        return None
    elif len(tensors) == 1:
        return tensors[0]

    return cat(tensors, dim = dim)

def safe_squeeze_first(t):
    if not exists(t):
        return None

    if t.shape[0] != 1:
        return t

    return rearrange(t, '1 ... -> ...')

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(
    t,
    temperature = 1.,
    dim = -1,
    keepdim = False,
    eps = 1e-10
):
    noised = (t / max(temperature, eps)) + gumbel_noise(t)
    return noised.argmax(dim = dim, keepdim = keepdim)

def pack_one(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return first(unpack(out, packed_shape, inv_pattern))

    return packed, inverse

def pad_at_dim(
    t,
    pad: tuple[int, int],
    dim = -1,
    value = 0.
):
    if pad == (0, 0):
        return t

    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pad_to_len(t, target_len, *, dim):
    curr_len = t.shape[dim]

    if curr_len >= target_len:
        return t

    return pad_at_dim(t, (0, target_len - curr_len), dim = dim)

def pad_tensors_at_dim_to_max_len(
    tensors: list[Tensor],
    dims: tuple[int, ...]
):
    for dim in dims:
        if dim >= first(tensors).ndim:
            continue

        max_time = max([t.shape[dim] for t in tensors])
        tensors = [pad_to_len(t, max_time, dim = dim) for t in tensors]

    return tensors

def align_dims_left(t, aligned_to):
    shape = t.shape
    num_right_dims = aligned_to.ndim - t.ndim

    if num_right_dims < 0:
        return

    return t.reshape(*shape, *((1,) * num_right_dims))

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def softclamp(t, value = 50.):
    return (t / value).tanh() * value

def create_multi_token_prediction_targets(
    t, # (b t ...)
    steps_future,

): # (b t-1 steps ...), (b t-1 steps) - targets and the mask, where mask is False for padding

    batch, seq_len, device = *t.shape[:2], t.device

    batch_arange = arange(batch, device = device)
    seq_arange = arange(seq_len, device = device)
    steps_arange = arange(steps_future, device = device)

    indices = add('t, steps -> t steps', seq_arange, steps_arange)
    mask = indices < seq_len

    batch_arange = rearrange(batch_arange, 'b -> b 1 1')

    indices[~mask] = 0
    mask = repeat(mask, 't steps -> b t steps', b = batch)

    out = t[batch_arange, indices]

    return out, mask

# loss related

class LossNormalizer(Module):

    # the authors mentioned the need for loss normalization in the dynamics transformer

    def __init__(
        self,
        num_losses: int,
        beta = 0.95,
        eps = 1e-6
    ):
        super().__init__()
        self.register_buffer('exp_avg_sq', torch.ones(num_losses))
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        losses: Tensor | list[Tensor] | dict[str, Tensor],
        update_ema = None
    ):
        exp_avg_sq, beta = self.exp_avg_sq, self.beta
        update_ema = default(update_ema, self.training)

        # get the rms value - as mentioned at the end of section 3 in the paper

        rms = exp_avg_sq.sqrt()

        if update_ema:
            decay = 1. - beta

            # update the ema

            exp_avg_sq.lerp_(losses.detach().square(), decay)

        # then normalize

        assert losses.numel() == rms.numel()

        normed_losses = losses / rms.clamp(min = self.eps)

        return normed_losses

class LPIPSLoss(Module):
    def __init__(
        self,
        vgg: Module | None = None,
        vgg_weights: VGG16_Weights = VGG16_Weights.DEFAULT,
        sampled_frames = 1
    ):
        super().__init__()

        if not exists(vgg):
            vgg = torchvision.models.vgg16(weights = vgg_weights)
            vgg.classifier = Sequential(*vgg.classifier[:-2])

        self.vgg = [vgg]
        self.sampled_frames = sampled_frames

    def forward(
        self,
        pred,
        data,
    ):
        batch, device, is_video = pred.shape[0], pred.device, pred.ndim == 5

        vgg, = self.vgg
        vgg = vgg.to(data.device)

        # take care of sampling random frames of the video

        if is_video:
            pred, data = tuple(rearrange(t, 'b c t ... -> b t c ...') for t in (pred, data))

            # batch randperm

            batch_randperm = randn(pred.shape[:2], device = pred.device).argsort(dim = -1)
            rand_frames = batch_randperm[..., :self.sampled_frames]

            batch_arange = arange(batch, device = device)
            batch_arange = rearrange(batch_arange, '... -> ... 1')

            pred, data = tuple(t[batch_arange, rand_frames] for t in (pred, data))

            # fold sampled frames into batch

            pred, data = tuple(rearrange(t, 'b t c ... -> (b t) c ...') for t in (pred, data))

        pred_embed, embed = tuple(vgg(t) for t in (pred, data))

        return F.mse_loss(embed, pred_embed)

def ramp_weight(times, slope = 0.9, intercept = 0.1):
    # equation (8) paper, their "ramp" loss weighting
    return slope * times + intercept

# reinforcement learning related

# rewards

class SymExpTwoHot(Module):
    def __init__(
        self,
        reward_range = (-20., 20.),
        num_bins = 255,
        learned_embedding = False,
        dim_embed = None,
    ):
        super().__init__()

        min_value, max_value = reward_range
        values = linspace(min_value, max_value, num_bins)
        values = values.sign() * (torch.exp(values.abs()) - 1.)

        self.reward_range = reward_range
        self.num_bins = num_bins
        self.register_buffer('bin_values', values)

        # take care of a reward embedding
        # for an improvisation where agent tokens can also see the past rewards - it makes sense that this information should not be thrown out, a la Decision Transformer

        self.learned_embedding = learned_embedding

        if learned_embedding:
            assert exists(dim_embed)
            self.bin_embeds = nn.Embedding(num_bins, dim_embed)

    @property
    def device(self):
        return self.bin_values.device

    def embed(
        self,
        two_hot_encoding,
    ):
        assert self.learned_embedding, f'can only embed if `learned_embedding` is True'

        weights, bin_indices = two_hot_encoding.topk(k = 2, dim = -1)

        two_embeds = self.bin_embeds(bin_indices)

        return einsum(two_embeds, weights, '... two d, ... two -> ... d')

    def bins_to_scalar_value(
        self,
        logits, # (... l)
        normalize = False
    ):
        two_hot_encoding = logits.softmax(dim = -1) if normalize else logits
        return einsum(two_hot_encoding, self.bin_values, '... l, l -> ...')

    def forward(
        self,
        values
    ):
        bin_values = self.bin_values
        min_bin_value, max_bin_value = self.bin_values[0], self.bin_values[-1]

        values, inverse_pack = pack_one(values, '*')
        num_values = values.shape[0]

        values = values.clamp(min = min_bin_value, max = max_bin_value)

        indices = torch.searchsorted(self.bin_values, values)

        # fetch the closest two indices (two-hot encoding)

        left_indices = (indices - 1).clamp(min = 0)
        right_indices = left_indices + 1

        left_indices, right_indices = tuple(rearrange(t, '... -> ... 1') for t in (left_indices, right_indices))

        # fetch the left and right values for the consecutive indices

        left_values = self.bin_values[left_indices]
        right_values = self.bin_values[right_indices]

        # calculate the left and right values by the distance to the left and right

        values = rearrange(values, '... -> ... 1')
        total_distance = right_values - left_values

        left_logit_value = (right_values - values) / total_distance
        right_logit_value = 1. - left_logit_value

        # set the left and right values (two-hot)

        encoded = torch.zeros((num_values, self.num_bins), device = self.device)

        encoded.scatter_(-1, left_indices, left_logit_value)
        encoded.scatter_(-1, right_indices, right_logit_value)

        return inverse_pack(encoded, '* l')

# action related

ActionEmbeds = namedtuple('ActionEmbed', ('discrete', 'continuous'))

class ActionEmbedder(Module):
    def __init__(
        self,
        dim,
        *,
        num_discrete_actions: int | tuple[int, ...] = 0,
        num_continuous_actions  = 0,
        continuous_norm_stats: tuple[tuple[float, float], ...] | None = None,
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
        squeeze = True,
        **kwargs
    ):
        inverse_norm_continuous = default(inverse_norm_continuous, self.continuous_need_norm)

        discrete_logits, continuous_mean_log_var = self.unembed(embed, return_split_discrete = True, pred_head_index = pred_head_index, **kwargs)

        sampled_discrete = sampled_continuous = None

        if exists(discrete_logits):
            sampled_discrete = []

            for one_discrete_logits in discrete_logits:
                sampled_discrete.append(gumbel_sample(one_discrete_logits, temperature = discrete_temperature, keepdim = True))

            sampled_discrete = cat(sampled_discrete, dim = -1)

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
        parallel_discrete_calc = default(parallel_discrete_calc, exists(discrete_targets) and discrete_targets.shape[-1] > 1)

        discrete_action_logits, continuous_action_mean_log_var = self.unembed(embeds, pred_head_index = pred_head_index, discrete_action_types = discrete_action_types, continuous_action_types = continuous_action_types, return_split_discrete = True)

        # discrete

        discrete_log_probs = None
        discrete_entropies = None

        if exists(discrete_targets):

            if parallel_discrete_calc:
                # use nested tensors

                jagged_dims = tuple(t.shape[-1] for t in discrete_action_logits)

                discrete_action_logits = cat(discrete_action_logits, dim = -1)

                discrete_action_logits, inverse_pack_lead_dims = pack_one(discrete_action_logits, '* l')
                batch = discrete_action_logits.shape[0]

                discrete_action_logits = rearrange(discrete_action_logits, 'b l -> (b l)')

                # to nested tensor

                nested_logits = nested_tensor(discrete_action_logits.split(jagged_dims * batch), layout = torch.jagged, device = self.device, requires_grad = True)

                prob = nested_logits.softmax(dim = -1)

                log_probs = log(prob)

                # maybe entropy

                if return_entropies:
                    discrete_entropies = (-prob * log_probs).sum(dim = -1, keepdim = True)
                    discrete_entropies = cat(discrete_entropies.unbind())
                    discrete_entropies = rearrange(discrete_entropies, '(b na) -> b na', b = batch)

                    discrete_entropies = inverse_pack_lead_dims(discrete_entropies, '* na')

                # back to regular tensor

                log_probs = cat(log_probs.unbind())
                log_probs = rearrange(log_probs, '(b l) -> b l', b = batch)

                log_probs = inverse_pack_lead_dims(log_probs)

                # get indices to gather

                discrete_action_types = default(discrete_action_types, self.default_discrete_action_types)

                num_discrete_actions = self.num_discrete_actions[discrete_action_types]

                offset = F.pad(num_discrete_actions.cumsum(dim = -1), (1, -1), value = 0)
                log_prob_indices = discrete_targets + offset

                # gather

                discrete_log_probs = log_probs.gather(-1, log_prob_indices)

            else:
                discrete_log_probs = []
                discrete_entropies = []

                for one_discrete_action_logit, one_discrete_target in zip(discrete_action_logits, discrete_targets.unbind(dim = -1)):

                    one_discrete_probs = one_discrete_action_logit.softmax(dim = -1)
                    one_discrete_log_probs = log(one_discrete_probs)
                    one_discrete_target = rearrange(one_discrete_target, '... -> ... 1')

                    log_prob = one_discrete_log_probs.gather(-1, one_discrete_target)
                    discrete_log_probs.append(log_prob)

                    if return_entropies:
                        entropy = (-one_discrete_probs * one_discrete_log_probs).sum(dim = -1)
                        discrete_entropies.append(entropy)

                discrete_log_probs = cat(discrete_log_probs, dim = -1)

                if return_entropies:
                    discrete_entropies = stack(discrete_entropies, dim = -1)

        # continuous

        continuous_log_probs = None
        continuous_entropies = None

        if exists(continuous_targets):
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

        src_discrete, src_continuous = src
        tgt_discrete, tgt_continuous = tgt

        discrete_kl_div = None

        # split discrete if it is not already (multiple discrete actions)

        if exists(src_discrete):

            discrete_split = self.num_discrete_actions.tolist()

            if is_tensor(src_discrete):
                src_discrete = src_discrete.split(discrete_split, dim = -1)

            if is_tensor(tgt_discrete):
                tgt_discrete = tgt_discrete.split(discrete_split, dim = -1)

            discrete_kl_divs = []

            for src_logit, tgt_logit in zip(src_discrete, tgt_discrete):

                src_log_probs = src_logit.log_softmax(dim = -1)
                tgt_prob = tgt_logit.softmax(dim = -1)

                one_discrete_kl_div = F.kl_div(src_log_probs, tgt_prob, reduction = 'none')

                discrete_kl_divs.append(one_discrete_kl_div.sum(dim = -1))

            discrete_kl_div = stack(discrete_kl_divs, dim = -1)

        # calculate kl divergence for continuous

        continuous_kl_div = None

        if exists(src_continuous):
            src_normal = mean_log_var_to_distr(src_continuous)
            tgt_normal = mean_log_var_to_distr(tgt_continuous)

            continuous_kl_div = kl.kl_divergence(src_normal, tgt_normal)

        # maybe reduce

        if reduce_across_num_actions:
            if exists(discrete_kl_div):
                discrete_kl_div = discrete_kl_div.sum(dim = -1)

            if exists(continuous_kl_div):
                continuous_kl_div = continuous_kl_div.sum(dim = -1)

        return discrete_kl_div, continuous_kl_div

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

class Rotary1D(Module):
    def __init__(
        self,
        dim_head,
        theta = 10000.
    ):
        super().__init__()
        inv_freq = 1.0 / (theta ** (arange(0, dim_head, 2).float() / dim_head))
        self.register_buffer('inv_freq', inv_freq)

    def forward(
        self,
        seq_len,
        offset = 0
    ):
        device, dtype = self.inv_freq.device, self.inv_freq.dtype

        t = torch.arange(seq_len, device = device).type(dtype) + offset
        freqs = einsum(t, self.inv_freq, 'i, j -> i j')

        return cat((freqs, freqs), dim = -1)


def apply_rotations(
    rotations, # (h n d) | (n d)
    t          # (b h n d)
):

    heads, seq_len, dtype = *t.shape[1:3], t.dtype

    rotations_seq_len = rotations.shape[-2]

    # handle kv caching with rotations

    if rotations_seq_len > seq_len:
        rotations = rotations[-seq_len:]

    # precision

    t = t.float()

    # handle gqa for rotary

    if rotations.ndim == 3 and rotations.shape[0] < heads:
        rotary_heads = rotations.shape[0]

        assert divisible_by(heads, rotary_heads)
        groups = heads // rotary_heads
        rotations = repeat(rotations, 'h ... -> (h g) ...', g = groups)

    x1, x2 = t.chunk(2, dim = -1)
    rotated_half_t = cat((-x2, x1), dim = -1)

    # rotate in the positions

    rotated = t * rotations.cos() + rotated_half_t * rotations.sin()
    return rotated.type(dtype)

# multi-head rmsnorm

class MultiHeadRMSNorm(Module):
    def __init__(
        self,
        dim_head,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** 0.5
        self.gamma = Parameter(torch.zeros(heads, dim_head)) # weight decay friendly

    def forward(
        self,
        x # (b h n d)
    ):
        normed = l2norm(x)
        scale = (self.gamma + 1.) * self.scale
        return multiply('... h n d, h d', normed, scale)

# naive attend

def naive_attend(
    q, k, v,
    softclamp_value = None,
    scale = None,
    causal = False,
    causal_block_size = 1,
    mask = None
):

    if not exists(scale):
        scale = q.shape[-1] ** -0.5

    # grouped query attention

    groups = q.shape[1] // k.shape[1]

    q = rearrange(q, 'b (h g) ... -> b h g ...', g = groups)

    # similarity

    sim = einsum(q, k, 'b h g i d, b h j d -> b h g i j')

    # scale and attention

    sim = sim * scale

    # softclamping a la gemma 3

    if exists(softclamp_value):
        sim = softclamp(sim, softclamp_value)

    # masking

    mask_value = -torch.finfo(sim.dtype).max

    if exists(mask):
        sim = sim.masked_fill(~mask, mask_value)

    if causal:
        is_blocked_causal = causal_block_size > 1
        i, j = sim.shape[-2:]

        if is_blocked_causal:
          i = ceil(i / causal_block_size)
          j = ceil(j / causal_block_size)

        causal_mask = torch.ones((i, j), dtype = torch.bool, device = sim.device).triu(j - i + 1)

        if causal_block_size > 1:
            causal_mask = repeat(causal_mask, 'i j -> (i b1) (j b2)', b1 = causal_block_size, b2 = causal_block_size)
            causal_mask = causal_mask[:sim.shape[-2], :sim.shape[-1]]

        sim = sim.masked_fill(causal_mask, mask_value)

    # attend

    attn = sim.softmax(dim = -1)

    # aggregate

    out = einsum(attn, v, 'b h g i j, b h j d -> b h g i d')

    # merge the groups

    return rearrange(out, 'b h g i d -> b (h g) i d')

# flex attention related and factory function for attend depending on whether on cuda + flex attention available

def block_mask_causal(block_size):

    def inner(b, h, q, k):
        bq = q // block_size
        bk = k // block_size
        return bq >= bk

    return inner

def special_token_mask(q, k, seq_len, num_tokens, special_attend_only_itself = False):
    bq = q % seq_len
    bk = k % seq_len

    is_special_start_index = seq_len - num_tokens

    q_is_special = q >= is_special_start_index
    k_is_special = k >= is_special_start_index

    if special_attend_only_itself:
        out = ~(q_is_special & ~k_is_special) # modality attends to everything, but latent can only attend to itself (proposed attention pattern for encoder of video tokenizer)
    else:
        out = ~(~q_is_special & k_is_special) # modality cannot attend to agent tokens

    return out

def block_mask_special_tokens_right(
    seq_len,
    num_tokens,
    special_attend_only_itself = False
):
    def inner(b, h, q, k):
        return special_token_mask(q, k, seq_len, num_tokens, special_attend_only_itself)
    return inner

def compose_mask(mask1, mask2):
    def inner(b, h, q, k):
        return mask1(b, h, q, k) & mask2(b, h, q, k)

    return inner

def block_mask_noop(b, h, q, k):
    return b >= 0

def score_mod_softclamp(value):
    def inner(sim, b, h, q, k):
        if not exists(value):
           return sim

        sim = sim / value
        sim = torch.tanh(sim)
        sim = sim * value
        return sim

    return inner

# factory for attend function

def get_attend_fn(
    use_flex,
    seq_len,
    k_seq_len,
    causal = False,
    causal_block_size = 1,
    softclamp_value = 50.,
    num_special_tokens = 0,             # special tokens are latents / agents
    block_size_per_special = None,      # defaults to k_seq_len
    special_attend_only_itself = False, # by default, modality only attends to itself while special sees everything, but if turned True, will be the inverse - special can only attend to itself but modality can attend everything
    device = None
):
    block_size_per_special = default(block_size_per_special, k_seq_len)

    if use_flex:
        # flex pathway

        block_mask_fn = block_mask_causal(causal_block_size) if causal else block_mask_noop

        if num_special_tokens > 0:
            special_block_mask = block_mask_special_tokens_right(block_size_per_special, num_special_tokens, special_attend_only_itself)
            block_mask_fn = compose_mask(block_mask_fn, special_block_mask)

        block_mask = create_block_mask(block_mask_fn, B = None, H = None, Q_LEN = seq_len, KV_LEN = k_seq_len)

        score_mod = score_mod_softclamp(softclamp_value)
        attend_fn = partial(flex_attention, block_mask = block_mask, score_mod = score_mod, enable_gqa = True)
    else:
        # naive pathway

        mask = None
        if num_special_tokens > 0:
            q_seq = torch.arange(seq_len, device = device)[:, None]
            k_seq = torch.arange(k_seq_len, device = device)[None, :]

            mask = special_token_mask(q_seq, k_seq, block_size_per_special, num_special_tokens, special_attend_only_itself)

        attend_fn = partial(naive_attend, causal = causal, causal_block_size = causal_block_size, mask = mask, softclamp_value = softclamp_value)

    return attend_fn

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        query_heads = None,
        heads = 8,
        pre_rmsnorm = True,
        gate_values = True,
        rmsnorm_query = False, # a paper claims that it is better to just norm only the keys https://openreview.net/forum?id=HkztQWZfl2
        rmsnorm_key = True,
        value_residual = True
    ):
        super().__init__()
        self.norm = RMSNorm(dim) if pre_rmsnorm else Identity()

        # setup grouped query attention

        query_heads = default(query_heads, heads)
        assert query_heads >= heads and divisible_by(query_heads, heads)

        # scaling, splitting and merging of heads

        self.split_heads = Rearrange('b n (h d) -> b h n d', d = dim_head)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        dim_q_inner = dim_head * query_heads
        dim_kv_inner = dim_head * heads

        self.to_q = LinearNoBias(dim, dim_q_inner)
        self.to_k = LinearNoBias(dim, dim_kv_inner)
        self.to_v = LinearNoBias(dim, dim_kv_inner)
        self.to_out = LinearNoBias(dim_q_inner, dim)

        # alphafold gating per head, for attending to nothing

        self.to_gates = None

        if gate_values:
            self.to_gates = Sequential(
                LinearNoBias(dim, query_heads),
                Rearrange('b n h -> b h n 1'),
                nn.Sigmoid()
            )

        # stability related

        self.q_heads_rmsnorm = MultiHeadRMSNorm(dim_head, heads = query_heads) if rmsnorm_query else nn.Identity()
        self.k_heads_rmsnorm = MultiHeadRMSNorm(dim_head, heads = heads) if rmsnorm_key else nn.Identity()

        # value residual

        self.to_learned_value_residual_mix = nn.Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if value_residual else None

    def muon_parameters(self):
        # omit the queries and keys for now given what we learned from kimi 2 paper

        return [
            *self.to_v.parameters(),
            *self.to_out.parameters(),
        ]

    def forward(
        self,
        tokens, # (b n d)
        kv_cache = None,
        return_intermediates = False,
        rotary_pos_emb = None,
        residual_values = None,  # (b n h d)
        attend_fn: Callable | None = None
    ):
        tokens, inverse_packed_batch = pack_one(tokens, '* n d')

        tokens = self.norm(tokens)

        q, k, v = (self.to_q(tokens), self.to_k(tokens), self.to_v(tokens))

        # split heads

        q, k, v = map(self.split_heads, (q, k, v))

        # handle maybe value residual

        if exists(residual_values):
            residual_values = rearrange(residual_values, '... n h d -> (...) h n d')

            assert exists(self.to_learned_value_residual_mix)

            learned_mix = self.to_learned_value_residual_mix(tokens)

            v = v.lerp(residual_values, learned_mix)

        # qk rmsnorm

        q = self.q_heads_rmsnorm(q)
        k = self.k_heads_rmsnorm(k)

        # rotary

        if exists(rotary_pos_emb):
            q = apply_rotations(rotary_pos_emb, q)
            k = apply_rotations(rotary_pos_emb, k)

        # caching

        if exists(kv_cache):
            ck, cv = kv_cache
            k = cat((ck, k), dim = -2)
            v = cat((cv, v), dim = -2)

        # attention

        attend_fn = default(attend_fn, naive_attend)

        out = attend_fn(q, k, v)

        # gate values

        if exists(self.to_gates):
            gates = self.to_gates(tokens)
            out = out * gates

        # merge heads

        out = self.merge_heads(out)

        # combine heads

        out = self.to_out(out)

        out = inverse_packed_batch(out)

        if not return_intermediates:
            return out

        return out, AttentionIntermediates(stack((k, v)), tokens)

# feedforward

class SwiGLUFeedforward(Module):
    def __init__(
        self,
        dim,
        expansion_factor = 4,
        pre_rmsnorm = True
    ):
        super().__init__()
        self.norm = RMSNorm(dim) if pre_rmsnorm else Identity()

        dim_inner = int(dim * expansion_factor * 2 / 3)

        self.proj_in = Linear(dim, dim_inner * 2)
        self.proj_out = Linear(dim_inner, dim)

    def muon_parameters(self):
        return [
            self.proj_in.weight,
            self.proj_out.weight,
        ]

    def forward(self, x):
        x = self.norm(x)

        x, gates = self.proj_in(x).chunk(2, dim = -1)
        x = x * F.gelu(gates)

        return self.proj_out(x)

# rnn

class GRULayer(Module):
    def __init__(
        self,
        dim,
        dim_out
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.gru = nn.GRU(dim, dim_out, batch_first = True)

    def forward(
        self,
        x,
        prev_hiddens = None
    ):
        x = self.norm(x)

        x, hiddens = self.gru(x, prev_hiddens)

        return x, hiddens

# axial space time transformer

class AxialSpaceTimeTransformer(Module):
    def __init__(
        self,
        dim,
        depth,
        attn_heads = 8,
        attn_dim_head = 64,
        attn_softclamp_value = 50.,
        time_block_every = 4,
        attn_kwargs: dict = dict(),
        ff_kwargs: dict = dict(),
        num_residual_streams = 1,
        num_special_spatial_tokens = 1,
        special_attend_only_itself = False,  # this is set to True for the video tokenizer decoder (latents can only attend to itself while spatial modalities attend to the latents and everything)
        final_norm = True,
        value_residual = True,               # https://arxiv.org/abs/2410.17897 - but with learned mixing from OSS
        rnn_time = True
    ):
        super().__init__()
        assert depth >= time_block_every, f'depth must be at least {time_block_every}'

        # hyper connections

        hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, dim = dim)

        # attention

        self.attn_softclamp_value = attn_softclamp_value

        # attention masking

        self.special_attend_only_itself = special_attend_only_itself

        # time rotary embedding

        self.time_rotary = Rotary1D(attn_dim_head)

        # project initial for value residuals

        self.value_residual = value_residual

        if value_residual:
            dim_inner = attn_dim_head * attn_heads

            self.to_value_residual = nn.Sequential(
                nn.RMSNorm(dim),
                nn.Linear(dim, dim_inner, bias = False),
                Rearrange('... (h d) -> ... h d', h = attn_heads)
            )

        # a gru layer across time

        self.rnn_time = rnn_time
        rnn_layers = []

        # transformer

        layers = []
        is_time = []

        for i in range(depth):
            layer_index = i + 1

            is_time_block = divisible_by(layer_index, time_block_every)
            is_time.append(is_time_block)

            rearrange_to_attend = Rearrange('b t s ... -> b s t ...') if is_time_block else Identity()
            rearrange_from_attend = Rearrange('b s t ... -> b t s ...') if is_time_block else Identity()

            layers.append(ModuleList([
                rearrange_to_attend,
                rearrange_from_attend,
                hyper_conn(branch = Attention(dim = dim, heads = attn_heads, dim_head = attn_dim_head, value_residual = value_residual, **attn_kwargs)),
                hyper_conn(branch = SwiGLUFeedforward(dim = dim, **ff_kwargs))
            ]))

            rnn_layers.append(hyper_conn(branch = GRULayer(dim, dim)) if is_time_block and rnn_time else None)

        self.layers = ModuleList(layers)
        self.rnn_layers = ModuleList(rnn_layers)

        self.is_time = is_time

        # final norm

        self.final_norm = nn.RMSNorm(dim) if final_norm else nn.Identity()

        # special tokens

        self.num_special_spatial_tokens = num_special_spatial_tokens

    def muon_parameters(self):
        muon_params = []

        for m in self.modules():
            if isinstance(m, (Attention, SwiGLUFeedforward)):
                muon_params.extend(m.muon_parameters())

        return muon_params

    def forward(
        self,
        tokens, # (b t s d)
        cache: TransformerIntermediates | None = None,
        return_intermediates = False

    ): # (b t s d) | (y 2 b h t d)

        batch, time, space_seq_len, _, device = *tokens.shape, tokens.device

        assert tokens.ndim == 4

        # destruct intermediates to cache for attention and rnn respectively

        kv_cache = rnn_prev_hiddens = None

        if exists(cache):
            kv_cache = cache.next_kv_cache
            rnn_prev_hiddens = cache.next_rnn_hiddens

        # attend functions for space and time

        has_kv_cache = exists(kv_cache) 
        use_flex = exists(flex_attention) and tokens.is_cuda and not has_kv_cache # KV cache shape breaks flex attention TODO: Fix

        attend_kwargs = dict(use_flex = use_flex, softclamp_value = self.attn_softclamp_value, special_attend_only_itself = self.special_attend_only_itself, device = device)

        space_attend = get_attend_fn(causal = False, seq_len = space_seq_len, k_seq_len = space_seq_len, num_special_tokens = self.num_special_spatial_tokens, **attend_kwargs) # space has an agent token on the right-hand side for reinforcement learning - cannot be attended to by modality

        time_attend = get_attend_fn(causal = True, seq_len = time, k_seq_len = time, **attend_kwargs)

        # prepare cache

        time_attn_kv_caches = []
        rnn_hiddens = []

        if has_kv_cache:
            past_tokens, tokens = tokens[:, :-1], tokens[:, -1:]

            rotary_seq_len = 1
            rotary_pos_offset = past_tokens.shape[1]
        else:
            rotary_seq_len = time
            rotary_pos_offset = 0

        kv_cache = default(kv_cache, (None,))

        iter_kv_cache = iter(kv_cache)

        rnn_prev_hiddens = default(rnn_prev_hiddens, (None,))

        iter_rnn_prev_hiddens = iter(rnn_prev_hiddens)

        # rotary

        rotary_pos_emb = self.time_rotary(rotary_seq_len, offset = rotary_pos_offset)

        # value residual

        residual_values = None

        if self.value_residual:
            residual_values = self.to_value_residual(tokens)

        # normed attention inputs

        normed_time_attn_inputs = []
        normed_space_attn_inputs = []

        # attention

        tokens = self.expand_streams(tokens)

        for (pre_attn_rearrange, post_attn_rearrange, attn, ff), maybe_rnn, layer_is_time in zip(self.layers, self.rnn_layers, self.is_time):

            tokens = pre_attn_rearrange(tokens)

            # maybe rnn for time

            if layer_is_time and exists(maybe_rnn):

                tokens, inverse_pack_batch = pack_one(tokens, '* t d')

                tokens, layer_rnn_hiddens = maybe_rnn(tokens, next(iter_rnn_prev_hiddens, None)) # todo, handle rnn cache

                tokens = inverse_pack_batch(tokens)

                rnn_hiddens.append(layer_rnn_hiddens)

            # when is a axial time attention block, should be causal

            attend_fn = time_attend if layer_is_time else space_attend

            layer_rotary_pos_emb = rotary_pos_emb if layer_is_time else None

            # maybe past kv cache

            maybe_kv_cache = next(iter_kv_cache, None) if layer_is_time else None

            # residual values

            layer_residual_values = maybe(pre_attn_rearrange)(residual_values)

            # attention layer

            tokens, attn_intermediates = attn(
                tokens,
                rotary_pos_emb = layer_rotary_pos_emb,
                attend_fn = attend_fn,
                kv_cache = maybe_kv_cache,
                residual_values = layer_residual_values,
                return_intermediates = True
            )

            tokens = post_attn_rearrange(tokens)

            # feedforward layer

            tokens = ff(tokens)

            # save kv cache if is time layer

            if layer_is_time:
                time_attn_kv_caches.append(attn_intermediates.next_kv_cache)

            # save time attention inputs for decorr

            space_or_time_inputs = normed_time_attn_inputs if layer_is_time else normed_space_attn_inputs

            space_or_time_inputs.append(attn_intermediates.normed_inputs)

        tokens = self.reduce_streams(tokens)

        out = self.final_norm(tokens)

        if has_kv_cache:
            # just concat the past tokens back on for now, todo - clean up the logic
            out = cat((past_tokens, out), dim = 1)

        if not return_intermediates:
            return out

        intermediates = TransformerIntermediates(
            stack(time_attn_kv_caches),
            safe_stack(normed_time_attn_inputs),
            safe_stack(normed_space_attn_inputs),
            safe_stack(rnn_hiddens)
        )

        return out, intermediates

# video tokenizer

class VideoTokenizer(Module):
    def __init__(
        self,
        dim,
        dim_latent,
        patch_size,
        image_height = None,
        image_width = None,
        num_latent_tokens = 4,
        encoder_depth = 4,
        decoder_depth = 4,
        time_block_every = 4,
        attn_kwargs: dict = dict(),
        attn_dim_head = 64,
        attn_heads = 8,
        attn_softclamp_value = 50.,
        ff_kwargs: dict = dict(),
        decoder_pos_mlp_depth = 2,
        channels = 3,
        per_image_patch_mask_prob = (0., 0.9), # probability of patch masking appears to be per image probabilities drawn uniformly between 0. and 0.9 - if you are a phd student and think i'm mistakened, please open an issue
        lpips_loss_network: Module | None = None,
        lpips_loss_weight = 0.2,
        encoder_add_decor_aux_loss = False,
        decor_auxx_loss_weight = 0.1,
        decorr_sample_frac = 0.25,
        num_residual_streams = 1,
    ):
        super().__init__()

        self.patch_size = patch_size

        # special tokens

        assert num_latent_tokens >= 1
        self.num_latent_tokens = num_latent_tokens
        self.latent_tokens = Parameter(randn(num_latent_tokens, dim) * 1e-2)

        # hyper connections

        hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, dim = dim)

        # mae masking - Kaiming He paper from long ago

        self.per_image_patch_mask_prob = per_image_patch_mask_prob
        self.mask_token = Parameter(randn(dim) * 1e-2)

        # patch and unpatch

        dim_patch = channels * patch_size ** 2

        self.patch_to_tokens = Sequential(
            Rearrange('b c t (h p1) (w p2) -> b t h w (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            Linear(dim_patch, dim)
        )

        self.tokens_to_patch = Sequential(
            Linear(dim, dim_patch),
            Rearrange('b t h w (p1 p2 c) -> b c t (h p1) (w p2)', p1 = patch_size, p2 = patch_size),
        )

        # encoder space / time transformer

        self.encoder_transformer = AxialSpaceTimeTransformer(
            dim = dim,
            depth = encoder_depth,
            attn_dim_head = attn_dim_head,
            attn_softclamp_value = attn_softclamp_value,
            time_block_every = time_block_every,
            num_special_spatial_tokens = num_latent_tokens,
            num_residual_streams = num_residual_streams,
            final_norm = True
        )

        # latents

        self.encoded_to_latents = Sequential(
            LinearNoBias(dim, dim_latent),
            nn.Tanh(),
        )

        self.latents_to_decoder = LinearNoBias(dim_latent, dim)

        # decoder

        self.image_height = image_height
        self.image_width = image_width

        # parameterize the decoder positional embeddings for MAE style training so it can be resolution agnostic

        self.to_decoder_pos_emb = create_mlp(
            dim_in = 2,
            dim = dim * 2,
            dim_out = dim,
            depth = decoder_pos_mlp_depth,
        )

        # decoder transformer

        self.decoder_transformer = AxialSpaceTimeTransformer(
            dim = dim,
            depth = decoder_depth,
            attn_dim_head = attn_dim_head,
            attn_softclamp_value = attn_softclamp_value,
            time_block_every = time_block_every,
            num_special_spatial_tokens = num_latent_tokens,
            num_residual_streams = num_residual_streams,
            special_attend_only_itself = True,
            final_norm = True
        )

        # loss related

        self.register_buffer('zero', tensor(0.), persistent = False)

        self.has_lpips_loss = lpips_loss_weight > 0.
        self.lpips_loss_weight = lpips_loss_weight

        if self.has_lpips_loss:
            self.lpips = LPIPSLoss(lpips_loss_network)

        # decorr aux loss
        # https://arxiv.org/abs/2510.14657

        self.encoder_add_decor_aux_loss = encoder_add_decor_aux_loss
        self.decorr_aux_loss_weight = decor_auxx_loss_weight

        self.decorr_loss = DecorrelationLoss(decorr_sample_frac, soft_validate_num_sampled = True) if encoder_add_decor_aux_loss else None

    @property
    def device(self):
        return self.zero.device

    def muon_parameters(self):
        return [
            *self.encoder_transformer.muon_parameters(),
            *self.decoder_transformer.muon_parameters()
        ]

    @torch.no_grad()
    def tokenize(
        self,
        video
    ):
        self.eval()
        return self.forward(video, return_latents = True)

    def decode(
        self,
        latents, # (b t n d)
        height = None,
        width = None,
    ): # (b c t h w)

        height = default(height, self.image_height)
        width = default(width, self.image_width)

        assert exists(height) and exists(width), f'image height and width need to be passed in when decoding latents'

        batch, time, device = *latents.shape[:2], latents.device

        use_flex = latents.is_cuda and exists(flex_attention)

        num_patch_height = height // self.patch_size
        num_patch_width = width // self.patch_size

        # latents to tokens

        latent_tokens = self.latents_to_decoder(latents)

        # generate decoder positional embedding and concat the latent token

        spatial_pos_height = torch.linspace(-1., 1., num_patch_height, device = device)
        spatial_pos_width = torch.linspace(-1., 1., num_patch_width, device = device)

        space_height_width_coor = stack(torch.meshgrid(spatial_pos_height, spatial_pos_width, indexing = 'ij'), dim = -1)

        decoder_pos_emb = self.to_decoder_pos_emb(space_height_width_coor)
        decoder_pos_emb = repeat(decoder_pos_emb, '... -> b t ...', b = batch, t = time)

        tokens, packed_latent_shape = pack((decoder_pos_emb, latent_tokens), 'b t * d')

        # decoder attention

        tokens = self.decoder_transformer(tokens)

        # unpack latents

        tokens, latent_tokens = unpack(tokens, packed_latent_shape, 'b t * d')

        # project back to patches

        recon_video = self.tokens_to_patch(tokens)

        return recon_video

    def forward(
        self,
        video, # (b c t h w) 
        return_latents = False,
        mask_patches = None,
        return_all_losses = False
    ):
        batch, _, time, height, width = video.shape
        patch_size, device = self.patch_size, video.device

        assert divisible_by(height, patch_size) and divisible_by(width, patch_size)

        # to tokens

        tokens = self.patch_to_tokens(video)

        # get some dimensions

        num_patch_height, num_patch_width, _ = tokens.shape[-3:]

        # masking

        mask_patches = default(mask_patches, self.training)

        if mask_patches:
            min_mask_prob, max_mask_prob = self.per_image_patch_mask_prob

            mask_prob = torch.empty(tokens.shape[:2], device = tokens.device).uniform_(min_mask_prob, max_mask_prob) # (b t)

            mask_prob = repeat(mask_prob, 'b t -> b t vh vw', vh = tokens.shape[2], vw = tokens.shape[3])
            mask_patch = torch.bernoulli(mask_prob) == 1.

            tokens = einx.where('..., d, ... d', mask_patch, self.mask_token, tokens)

        # pack space

        tokens, inverse_pack_space = pack_one(tokens, 'b t * d')

        # add the latent

        latents = repeat(self.latent_tokens, 'n d -> b t n d', b = tokens.shape[0], t = tokens.shape[1])

        tokens, packed_latent_shape = pack((tokens, latents), 'b t * d')

        # encoder attention

        tokens, (_, time_attn_normed_inputs, space_attn_normed_inputs, _) = self.encoder_transformer(tokens, return_intermediates = True)

        # latent bottleneck

        tokens, latents = unpack(tokens, packed_latent_shape, 'b t * d')

        latents = self.encoded_to_latents(latents)

        if return_latents:
            return latents

        recon_video = self.decode(latents, height = height, width = width)

        # losses

        recon_loss = F.mse_loss(video, recon_video)

        lpips_loss = self.zero

        if self.has_lpips_loss:
            lpips_loss = self.lpips(video, recon_video)

        time_decorr_loss = space_decorr_loss = self.zero

        if self.encoder_add_decor_aux_loss:
            if exists(time_attn_normed_inputs):
                time_decorr_loss = self.decorr_loss(time_attn_normed_inputs)

            if exists(space_attn_normed_inputs):
                space_decorr_loss = self.decorr_loss(space_attn_normed_inputs)

        # losses

        total_loss = (
            recon_loss +
            lpips_loss * self.lpips_loss_weight +
            time_decorr_loss * self.decorr_aux_loss_weight +
            space_decorr_loss * self.decorr_aux_loss_weight
        )

        if not return_all_losses:
            return total_loss

        losses = (recon_loss, lpips_loss, decorr_loss)

        return total_loss, TokenizerLosses(*losses)

# dynamics model, axial space-time transformer

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

        return align_dims_left(times, align_dims_left_to)

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

        init_frame = env.reset()

        # frame to video

        if env_is_vectorized:
            video = rearrange(init_frame, 'b c vh vw -> b c 1 vh vw')
        else:
            video = rearrange(init_frame, 'c vh vw -> 1 c 1 vh vw')

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

            if len(env_step_out) == 2:
                next_frame, reward = env_step_out
                terminated = full((batch,), False)
                truncated = full((batch,), False)

            elif len(env_step_out) == 3:
                next_frame, reward, terminated = env_step_out
                truncated = full((batch,), False)

            elif len(env_step_out) == 4:
                next_frame, reward, terminated, truncated = env_step_out

            elif len(env_step_out) == 5:
                next_frame, reward, terminated, truncated, info = env_step_out

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

            if env_is_vectorized:
                next_frame = rearrange(next_frame, 'b c vh vw -> b c 1 vh vw')
                reward = rearrange(reward, 'b -> b 1')
            else:
                next_frame = rearrange(next_frame, 'c vh vw -> 1 c 1 vh vw')
                reward = rearrange(reward, ' -> 1 1')

            # concat

            video = cat((video, next_frame), dim = 2)
            rewards = safe_cat((rewards, reward), dim = 1)

            acc_agent_embed = safe_cat((acc_agent_embed, one_agent_embed), dim = 1)

        # package up one experience for learning

        batch, device = latents.shape[0], latents.device

        one_experience = Experience(
            latents = latents,
            video = video[:, :, :-1],
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
                    kl_div_loss = kl_div_loss + discrete_kl_div[mask].mean()

                if exists(continuous_kl_div):
                    kl_div_loss = kl_div_loss + continuous_kl_div[mask].mean()

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
                        aligned_times = align_dims_left(times, noised)

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

                signal_levels = randint(0, self.max_steps, (batch, time)) // num_step_sizes[:, None] * num_step_sizes[:, None] # times are discretized to step sizes

        # times is from 0 to 1

        times = self.get_times_from_signal_level(signal_levels)

        if not latent_is_noised:
            # get the noise

            noise = randn_like(latents)
            aligned_times = align_dims_left(times, latents)

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
                aligned_times = align_dims_left(times, proprio)

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

            half_step_size_align_left = align_dims_left(half_step_size, noised)

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
            loss_weight = align_dims_left(loss_weight, flow_losses)

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
