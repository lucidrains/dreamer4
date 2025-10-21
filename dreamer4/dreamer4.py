from __future__ import annotations

import math
from math import ceil, log2
from random import random
from collections import namedtuple
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nested import nested_tensor
from torch.distributions import Normal
from torch.nn import Module, ModuleList, Embedding, Parameter, Sequential, Linear, RMSNorm, Identity
from torch import nn, cat, stack, arange, tensor, Tensor, is_tensor, zeros, ones, randint, rand, randn, randn_like, empty, full, linspace, arange

import torchvision
from torchvision.models import VGG16_Weights

from torch.optim import Optimizer
from adam_atan2_pytorch import MuonAdamAtan2

from x_mlps_pytorch.ensemble import Ensemble
from x_mlps_pytorch.normed_mlp import create_mlp

from hyper_connections import get_init_and_expand_reduce_stream_functions

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

TokenizerLosses = namedtuple('TokenizerLosses', ('recon', 'lpips'))

WorldModelLosses = namedtuple('WorldModelLosses', ('flow', 'rewards', 'discrete_actions', 'continuous_actions'))

@dataclass
class Experience:
    latents: Tensor
    video: Tensor | None = None
    rewards: Tensor | None = None
    actions: tuple[Tensor, Tensor] | None = None
    log_probs: tuple[Tensor, Tensor] | None = None
    values: Tensor | None = None
    step_size: int | None = None
    agent_index: int = 0
    is_from_world_model: bool = True
    is_batched: bool = True

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(arr):
    return arr[0]

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

# tensor helpers

def is_empty(t):
    return t.numel() == 0

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

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
    seq_arange = arange(seq_len, device = device)[1:]
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
        return set([*self.discrete_action_unembed.parameters(), *self.continuous_action_unembed.parameters()])

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
            mean, log_var = continuous_action_mean_log_var.unbind(dim = -1)
            std = (0.5 * log_var).exp()

            distr = Normal(mean, std)
            continuous_log_probs = distr.log_prob(continuous_targets)

            if return_entropies:
                continuous_entropies = distr.entropy()

        log_probs = (discrete_log_probs, continuous_log_probs)

        if not return_entropies:
            return log_probs

        entropies = (discrete_entropies, continuous_entropies)

        return log_probs, entropies

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
    num_tokens
):
    def inner(b, h, q, k):
        return special_token_mask(q, k, seq_len, num_tokens)
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
        rmsnorm_key = True
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
        return_kv_cache = False,
        rotary_pos_emb = None,
        attend_fn: Callable | None = None
    ):
        tokens, inverse_packed_batch = pack_one(tokens, '* n d')

        tokens = self.norm(tokens)

        q, k, v = (self.to_q(tokens), self.to_k(tokens), self.to_v(tokens))

        # split heads

        q, k, v = map(self.split_heads, (q, k, v))

        # qk rmsnorm

        q = self.q_heads_rmsnorm(q)
        k = self.k_heads_rmsnorm(k)

        # caching

        if exists(kv_cache):
            ck, cv = kv_cache
            k = cat((ck, k), dim = -2)
            v = cat((cv, v), dim = -2)

        # rotary

        if exists(rotary_pos_emb):
            q = apply_rotations(rotary_pos_emb, q)
            k = apply_rotations(rotary_pos_emb, k)

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

        if not return_kv_cache:
            return out

        return out, stack((k, v))

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

# axial space time transformer

class AxialSpaceTimeTransformer(Module):
    def __init__(
        self,
        dim,
        depth,
        attn_dim_head = 64,
        attn_softclamp_value = 50.,
        time_block_every = 4,
        attn_kwargs: dict = dict(),
        ff_kwargs: dict = dict(),
        num_residual_streams = 1,
        num_special_spatial_tokens = 1,
        special_attend_only_itself = False,  # this is set to True for the video tokenizer decoder (latents can only attend to itself while spatial modalities attend to the latents and everything)
        final_norm = True
    ):
        super().__init__()

        # hyper connections

        hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, dim = dim)

        # attention

        self.attn_softclamp_value = attn_softclamp_value

        # attention masking

        self.special_attend_only_itself = special_attend_only_itself

        # time rotary embedding

        self.time_rotary = Rotary1D(attn_dim_head)

        # transformer

        layers = []
        is_time = []

        for i in range(depth):
            layer_index = i + 1

            is_time_block = divisible_by(layer_index, time_block_every)
            is_time.append(is_time_block)

            rearrange_to_attend = Rearrange('b t s d -> b s t d') if is_time_block else Identity()
            rearrange_from_attend = Rearrange('b s t d -> b t s d') if is_time_block else Identity()

            layers.append(ModuleList([
                rearrange_to_attend,
                rearrange_from_attend,
                hyper_conn(branch = Attention(dim = dim, dim_head = attn_dim_head, **attn_kwargs)),
                hyper_conn(branch = SwiGLUFeedforward(dim = dim, **ff_kwargs))
            ]))

        self.layers = ModuleList(layers)
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
        tokens,                          # (b t s d)
        kv_cache: Tensor | None = None,  # (y 2 b h t d)
        return_kv_cache = False

    ): # (b t s d) | (y 2 b h t d)

        batch, time, space_seq_len, _, device = *tokens.shape, tokens.device

        assert tokens.ndim == 4

        # attend functions for space and time

        use_flex = exists(flex_attention) and tokens.is_cuda

        attend_kwargs = dict(use_flex = use_flex, softclamp_value = self.attn_softclamp_value, special_attend_only_itself = self.special_attend_only_itself, device = device)

        space_attend = get_attend_fn(causal = False, seq_len = space_seq_len, k_seq_len = space_seq_len, num_special_tokens = self.num_special_spatial_tokens, **attend_kwargs) # space has an agent token on the right-hand side for reinforcement learning - cannot be attended to by modality

        time_attend = get_attend_fn(causal = True, seq_len = time, k_seq_len = time, **attend_kwargs)

        # prepare cache

        time_attn_kv_caches = []

        has_kv_cache = exists(kv_cache)


        if has_kv_cache:
            past_tokens, tokens = tokens[:, :-1], tokens[:, -1:]

            rotary_seq_len = 1
            rotary_pos_offset = past_tokens.shape[-2]
        else:
            rotary_seq_len = time
            rotary_pos_offset = 0

        kv_cache = default(kv_cache, (None,))

        iter_kv_cache = iter(kv_cache)

        # rotary

        rotary_pos_emb = self.time_rotary(rotary_seq_len, offset = rotary_pos_offset)

        # attention

        tokens = self.expand_streams(tokens)

        for (pre_attn_rearrange, post_attn_rearrange, attn, ff), layer_is_time in zip(self.layers, self.is_time):

            tokens = pre_attn_rearrange(tokens)

            # when is a axial time attention block, should be causal

            attend_fn = time_attend if layer_is_time else space_attend

            layer_rotary_pos_emb = rotary_pos_emb if layer_is_time else None

            # maybe past kv cache

            maybe_kv_cache = next(iter_kv_cache, None) if layer_is_time else None

            # attention layer

            tokens, next_kv_cache = attn(
                tokens,
                rotary_pos_emb = layer_rotary_pos_emb,
                attend_fn = attend_fn,
                kv_cache = maybe_kv_cache,
                return_kv_cache = True
            )

            tokens = post_attn_rearrange(tokens)

            # feedforward layer

            tokens = ff(tokens)

            # save kv cache if is time layer

            if layer_is_time:
                time_attn_kv_caches.append(next_kv_cache)

        tokens = self.reduce_streams(tokens)

        out = self.final_norm(tokens)

        if has_kv_cache:
            # just concat the past tokens back on for now, todo - clean up the logic
            out = cat((past_tokens, out), dim = 1)

        if not return_kv_cache:
            return out

        return out, stack(time_attn_kv_caches)

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
        nd_rotary_kwargs: dict = dict(
            rope_min_freq = 1.,
            rope_max_freq = 10000.,
            rope_p_zero_freqs = 0.
        ),
        num_residual_streams = 1
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
            final_norm = True
        )

        # loss related

        self.register_buffer('zero', tensor(0.), persistent = False)

        self.has_lpips_loss = lpips_loss_weight > 0.
        self.lpips_loss_weight = lpips_loss_weight

        if self.has_lpips_loss:
            self.lpips = LPIPSLoss(lpips_loss_network)

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

        tokens = self.encoder_transformer(tokens)

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

        # losses

        total_loss = (
            recon_loss +
            lpips_loss * self.lpips_loss_weight
        )

        if not return_all_losses:
            return total_loss

        losses = (recon_loss, lpips_loss)

        return total_loss, TokenizerLosses(losses)

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
        reward_encoder_kwargs: dict = dict(),
        depth = 4,
        pred_orig_latent = True,   # directly predicting the original x0 data yield better results, rather than velocity (x-space vs v-space)
        time_block_every = 4,      # every 4th block is time
        attn_kwargs: dict = dict(
            heads = 8,
        ),
        attn_dim_head = 64,
        attn_softclamp_value = 50.,
        ff_kwargs: dict = dict(),
        loss_weight_fn: Callable = ramp_weight,
        num_future_predictions = 8,                 # they do multi-token prediction of 8 steps forward
        prob_no_shortcut_train = None,              # probability of no shortcut training, defaults to 1 / num_step_sizes
        add_reward_embed_to_agent_token = False,
        add_reward_embed_dropout = 0.1,
        num_discrete_actions: int | tuple[int, ...] = 0,
        num_continuous_actions = 0,
        continuous_norm_stats = None,
        multi_token_pred_len = 8,
        value_head_mlp_depth = 3,
        policy_head_mlp_depth = 3,
        latent_flow_loss_weight = 1.,
        reward_loss_weight: float | list[float] = 1.,
        discrete_action_loss_weight: float | list[float] = 1.,
        continuous_action_loss_weight: float | list[float] = 1.,
        num_latent_genes = 0,                       # for carrying out evolution within the dreams https://web3.arxiv.org/abs/2503.19037
        num_residual_streams = 1,
        gae_discount_factor = 0.997,
        gae_lambda = 0.95,
        ppo_eps_clip = 0.2,
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
                Reduce('b t n s d -> b t n d', 'mean'),
                RMSNorm(dim),
                LinearNoBias(dim, dim_latent)
            )

        else:
            assert divisible_by(num_latent_tokens, num_spatial_tokens)
            latent_tokens_to_space = num_latent_tokens // num_spatial_tokens

            self.latents_to_spatial_tokens = Sequential(
                Rearrange('b t n d -> b t (n d)'),
                Linear(num_latent_tokens * dim_latent, dim * num_spatial_tokens),
                Rearrange('b t (s d) -> b t s d', s = num_spatial_tokens)
            )

            self.to_latent_pred = Sequential(
                RMSNorm(dim),
                LinearNoBias(dim, dim_latent * latent_tokens_to_space),
                Rearrange('b t s (n d) -> b t (s n) d', n = latent_tokens_to_space)
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
            attn_dim_head = attn_dim_head,
            attn_softclamp_value = attn_softclamp_value,
            attn_kwargs = attn_kwargs,
            ff_kwargs = ff_kwargs,
            num_residual_streams = num_residual_streams,
            num_special_spatial_tokens = num_agents,
            final_norm = False
        )

        # ppo related

        self.gae_use_accelerated = gae_use_accelerated
        self.gae_discount_factor = gae_discount_factor
        self.gae_lambda = gae_lambda

        self.ppo_eps_clip = ppo_eps_clip
        self.value_clip = value_clip
        self.policy_entropy_weight = value_clip

        # loss related

        self.flow_loss_normalizer = LossNormalizer(1)
        self.reward_loss_normalizer = LossNormalizer(multi_token_pred_len)
        self.discrete_actions_loss_normalizer = LossNormalizer(multi_token_pred_len) if num_discrete_actions > 0 else None
        self.continuous_actions_loss_normalizer = LossNormalizer(multi_token_pred_len) if num_discrete_actions > 0 else None

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

    def muon_parameters(self):
        return self.transformer.muon_parameters()

    def get_times_from_signal_level(
        self,
        signal_levels,
        align_dims_left_to = None
    ):
        times = signal_levels.float() / self.max_steps

        if not exists(align_dims_left_to):
            return times

        return align_dims_left(times, align_dims_left_to)

    def parameter(self):
        params = super().parameters()

        if not exists(self.video_tokenizer):
            return params

        return list(set(params) - set(self.video_tokenizer.parameters()))

    def learn_from_experience(
        self,
        experience: Experience,
        policy_optim: Optimizer | None = None,
        value_optim: Optimizer | None = None
    ):
        assert experience.is_batched

        latents = experience.latents
        actions = experience.actions
        old_log_probs = experience.log_probs
        old_values = experience.values
        rewards = experience.rewards

        step_size = experience.step_size
        agent_index = experience.agent_index

        assert all([*map(exists, (old_log_probs, actions, old_values, rewards, step_size))]), 'the generations need to contain the log probs, values, and rewards for policy optimization'

        returns = calc_gae(rewards, old_values, gamma = self.gae_discount_factor, lam = self.gae_lambda, use_accelerated = self.gae_use_accelerated)

        # apparently they just use the sign of the advantage
        # https://arxiv.org/abs/2410.04166v1

        advantage = (returns - old_values).sign()

        # replay for the action logits and values

        discrete_actions, continuous_actions = actions

        _, (agent_embed, _) = self.forward(
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

        agent_embed = agent_embed[..., agent_index, :]

        # ppo

        policy_embed = self.policy_head(agent_embed)

        log_probs, entropies = self.action_embedder.log_probs(policy_embed, pred_head_index = 0, discrete_targets = discrete_actions, continuous_targets = continuous_actions, return_entropies = True)

        # concat discrete and continuous actions into one for optimizing

        old_log_probs = safe_cat(old_log_probs, dim = -1)
        log_probs = safe_cat(log_probs, dim = -1)
        entropies = safe_cat(entropies, dim = -1)

        ratio = (log_probs - old_log_probs).exp()

        clipped_ratio = ratio.clamp(1. - self.ppo_eps_clip, 1. + self.ppo_eps_clip)

        advantage = rearrange(advantage, '... -> ... 1') # broadcast across all actions

        # clipped surrogate loss

        policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
        policy_loss = reduce(policy_loss, 'b t na -> b t', 'sum')

        policy_loss = policy_loss.mean()

        # handle entropy loss for naive exploration bonus

        entropy_loss = - reduce(entropies, 'b t na -> b t', 'sum').mean()

        total_policy_loss = (
            policy_loss +
            entropy_loss * self.policy_entropy_weight
        )

        # maye take policy optimizer step

        if exists(policy_optim):
            total_policy_loss.backward()

            policy_optim.step()
            policy_optim.zero_grad()

        # value loss

        value_bins = self.value_head(agent_embed)
        values = self.reward_encoder.bins_to_scalar_value(value_bins)

        clipped_values = old_values + (values - old_values).clamp(-self.value_clip, self.value_clip)
        clipped_value_bins = self.reward_encoder(clipped_values)

        return_bins = self.reward_encoder(returns)

        value_loss_1 = F.cross_entropy(value_bins, return_bins, reduction = 'none')
        value_loss_2 = F.cross_entropy(clipped_value_bins, return_bins, reduction = 'none')

        value_loss = torch.maximum(value_loss_1, value_loss_2).mean()

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
        image_height = None,
        image_width = None,
        return_decoded_video = None,
        context_signal_noise = 0.1,       # they do a noising of the past, this was from an old diffusion world modeling paper from EPFL iirc
        use_time_kv_cache = True,
        return_rewards_per_frame = False,
        return_agent_actions = False,
        return_log_probs_and_values = False,
        return_time_kv_cache = False

    ): # (b t n d) | (b c t h w)

        was_training = self.training
        self.eval()

        # validation

        assert log2(num_steps).is_integer(), f'number of steps {num_steps} must be a power of 2'
        assert 0 < num_steps <= self.max_steps, f'number of steps {num_steps} must be between 0 and {self.max_steps}'

        if isinstance(tasks, int):
            tasks = torch.full((batch_size,), tasks, device = self.device)

        assert not exists(tasks) or tasks.shape[0] == batch_size

        # get state latent shape

        latent_shape = self.latent_shape

        # derive step size

        step_size = self.max_steps // num_steps

        # denoising
        # teacher forcing to start with

        latents = empty((batch_size, 0, *latent_shape), device = self.device)

        past_context_noise = latents.clone()

        # maybe return actions

        return_agent_actions |= return_log_probs_and_values

        decoded_discrete_actions = None
        decoded_continuous_actions = None

        # policy optimization related

        decoded_discrete_log_probs = None
        decoded_continuous_log_probs = None
        decoded_values = None

        # maybe return rewards

        if return_rewards_per_frame:
            decoded_rewards = empty((batch_size, 0), device = self.device, dtype = torch.float32)

        # handle maybe time kv cache

        time_kv_cache = None

        # while all the frames of the video (per latent) is not generated

        while latents.shape[1] < time_steps:

            curr_time_steps = latents.shape[1]
            noised_latent = randn((batch_size, 1, *latent_shape), device = self.device)

            for step in range(num_steps):
                is_last_step = (step + 1) == num_steps

                signal_levels = full((batch_size, 1), step * step_size, dtype = torch.long, device = self.device)

                noised_context = latents.lerp(past_context_noise, context_signal_noise) # the paragraph after eq (8)

                noised_latent_with_context, pack_context_shape = pack((noised_context, noised_latent), 'b * n d')

                signal_levels_with_context = F.pad(signal_levels, (curr_time_steps, 0), value = self.max_steps - 1)

                pred, (agent_embed, next_time_kv_cache) = self.forward(
                    latents = noised_latent_with_context,
                    signal_levels = signal_levels_with_context,
                    step_sizes = step_size,
                    rewards = decoded_rewards,
                    tasks = tasks,
                    discrete_actions = decoded_discrete_actions,
                    continuous_actions = decoded_continuous_actions,
                    time_kv_cache = time_kv_cache,
                    latent_is_noised = True,
                    return_pred_only = True,
                    return_intermediates = True,
                )

                if use_time_kv_cache and is_last_step:
                    time_kv_cache = next_time_kv_cache

                # unpack pred

                _, pred = unpack(pred, pack_context_shape, 'b * n d')

                # derive flow, based on whether in x-space or not

                if self.pred_orig_latent:
                    times = self.get_times_from_signal_level(signal_levels, noised_latent)
                    flow = (pred - noised_latent) / (1. - times)
                else:
                    flow = pred

                # denoise

                noised_latent += flow * (step_size / self.max_steps)

            denoised_latent = noised_latent # it is now denoised

            # take care of the rewards by predicting on the agent token embedding on the last denoising step

            if return_rewards_per_frame:
                one_agent_embed = agent_embed[:, -1:, agent_index]

                reward_logits = self.to_reward_pred.forward_one(one_agent_embed, id = 0)
                pred_reward = self.reward_encoder.bins_to_scalar_value(reward_logits, normalize = True)

                decoded_rewards = cat((decoded_rewards, pred_reward), dim = 1)

            # decode the agent actions if needed

            if return_agent_actions:
                assert self.action_embedder.has_actions

                one_agent_embed = agent_embed[:, -1:, agent_index]

                policy_embed = self.policy_head(one_agent_embed)

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

            past_context_noise = cat((past_context_noise, randn_like(denoised_latent)), dim = 1)

        # restore state

        self.train(was_training)

        # returning video

        has_tokenizer = exists(self.video_tokenizer)
        return_decoded_video = default(return_decoded_video, has_tokenizer)

        video = None

        if return_decoded_video:
            video = self.video_tokenizer.decode(
                latents,
                height = image_height,
                width = image_width
            )

        # only return video or latent if not requesting anything else, for first stage training

        if not has_at_least_one(return_rewards_per_frame, return_agent_actions):
            out = video if return_decoded_video else latents

            if not return_time_kv_cache:
                return out

            return out, time_kv_cache

        # returning agent actions, rewards, and log probs + values for policy optimization

        gen = Experience(
            latents = latents,
            video = video,
            step_size = step_size,
            agent_index = agent_index,
            is_from_world_model = True
        )

        if return_rewards_per_frame:
            gen.rewards = decoded_rewards

        if return_agent_actions:
            gen.actions = (decoded_discrete_actions, decoded_continuous_actions)

        if return_log_probs_and_values:
            gen.log_probs = (decoded_discrete_log_probs, decoded_continuous_log_probs)

            gen.values = decoded_values

        if not return_time_kv_cache:
            return gen

        return gen, time_kv_cache

    def forward(
        self,
        *,
        video = None,                    # (b c t vh vw)
        latents = None,                  # (b t n d) | (b t d)
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
        time_kv_cache = None,
        return_pred_only = False,
        latent_is_noised = False,
        return_all_losses = False,
        return_intermediates = False,
        add_autoregressive_action_loss = False,
        update_loss_ema = None
    ):
        # handle video or latents

        assert exists(video) ^ exists(latents)

        if exists(video):
            assert exists(self.video_tokenizer), 'video_tokenizer must be passed in if training from raw video on dynamics model'

            latents = self.video_tokenizer.tokenize(video)

        if latents.ndim == 3:
            latents = rearrange(latents, 'b t d -> b t 1 d') # 1 latent edge case

        assert latents.shape[-2:] == self.latent_shape

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

        times = self.get_times_from_signal_level(signal_levels, latents)

        if not latent_is_noised:
            # get the noise

            noise = randn_like(latents)

            # noise from 0 as noise to 1 as data

            noised_latents = noise.lerp(latents, times)

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

        # maybe reward tokens

        reward_tokens = agent_tokens[:, :, 0:0]

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
            action_tokens = agent_tokens[:, :, 0:0] # else empty off agent tokens

        # main function, needs to be defined as such for shortcut training - additional calls for consistency loss

        def get_prediction(noised_latents, signal_levels, step_sizes_log2, action_tokens, reward_tokens, agent_tokens, return_agent_tokens = False, return_time_kv_cache = False):
            # latents to spatial tokens

            space_tokens = self.latents_to_spatial_tokens(noised_latents)

            space_tokens, inverse_pack_space_per_latent = pack_one(space_tokens, 'b t * d')

            num_spatial_tokens = space_tokens.shape[-2]

            # action tokens

            num_action_tokens = 1 if not is_empty(action_tokens) else 0

            # reward tokens

            num_reward_tokens = 1 if not is_empty(reward_tokens) else 0

            # pack to tokens
            # [signal + step size embed] [latent space tokens] [register] [actions / agent]

            registers = repeat(self.register_tokens, 's d -> b t s d', b = batch, t = time)

            # determine signal + step size embed for their diffusion forcing + shortcut

            signal_embed = self.signal_levels_embed(signal_levels)

            step_size_embed = self.step_size_embed(step_sizes_log2)
            step_size_embed = repeat(step_size_embed, 'b ... -> b t ...', t = time)

            flow_token = cat((signal_embed, step_size_embed), dim = -1)
            flow_token = rearrange(flow_token, 'b t d -> b t d')

            # pack to tokens for attending

            tokens, packed_tokens_shape = pack([flow_token, space_tokens, registers, action_tokens, reward_tokens, agent_tokens], 'b t * d')

            # attention

            tokens, next_time_kv_cache = self.transformer(tokens, kv_cache = time_kv_cache, return_kv_cache = True)

            # unpack

            flow_token, space_tokens, register_tokens, action_tokens, reward_tokens, agent_tokens = unpack(tokens, packed_tokens_shape, 'b t * d')

            # pooling

            space_tokens = inverse_pack_space_per_latent(space_tokens)

            pred = self.to_latent_pred(space_tokens)

            if not return_agent_tokens:
                return pred

            if not return_time_kv_cache:
                return pred, agent_tokens

            return pred, (agent_tokens, next_time_kv_cache)

        # curry into get_prediction what does not change during first call as well as the shortcut ones

        _get_prediction = partial(get_prediction, action_tokens = action_tokens, reward_tokens = reward_tokens, agent_tokens = agent_tokens)

        # forward the network

        pred, (encoded_agent_tokens, next_time_kv_cache) = _get_prediction(noised_latents, signal_levels, step_sizes_log2, return_agent_tokens = True, return_time_kv_cache = True)

        if return_pred_only:
            if not return_intermediates:
                return pred

            return pred, (encoded_agent_tokens, next_time_kv_cache)

        # determine the target for the loss

        pred_target = None

        is_x_space = self.pred_orig_latent
        is_v_space_pred = not self.pred_orig_latent

        maybe_shortcut_loss_weight = 1.

        if no_shortcut_train:

            # allow for original velocity pred
            # x-space as in paper is in else clause

            if is_v_space_pred:
                pred_target = flow = latents - noise
            else:
                pred_target = latents
        else:
            # shortcut training - Frans et al. https://arxiv.org/abs/2410.12557

            # basically a consistency loss where you ensure quantity of two half steps equals one step
            # dreamer then makes it works for x-space with some math

            get_prediction_no_grad = torch.no_grad()(_get_prediction)

            step_sizes_log2_minus_one = step_sizes_log2 - 1 # which equals d / 2
            half_step_size = 2 ** step_sizes_log2_minus_one

            first_step_pred = get_prediction_no_grad(noised_latents, signal_levels, step_sizes_log2_minus_one)

            # first derive b'

            if is_v_space_pred:
                first_step_pred_flow = first_step_pred
            else:
                first_times = self.get_times_from_signal_level(signal_levels, noised_latents)
                first_step_pred_flow = (first_step_pred - noised_latents) / (1. - first_times)

            # take a half step

            half_step_size_align_left = align_dims_left(half_step_size, noised_latents)

            denoised_latent = noised_latents + first_step_pred_flow * (half_step_size_align_left / self.max_steps)

            # get second prediction for b''

            signal_levels_plus_half_step = signal_levels + half_step_size[:, None]
            second_step_pred = get_prediction_no_grad(denoised_latent, signal_levels_plus_half_step, step_sizes_log2_minus_one)

            if is_v_space_pred:
                second_step_pred_flow = second_step_pred
            else:
                second_times = self.get_times_from_signal_level(signal_levels_plus_half_step, denoised_latent)
                second_step_pred_flow = (second_step_pred - denoised_latent) / (1. - second_times)

            # pred target is sg(b' + b'') / 2

            pred_target = (first_step_pred_flow + second_step_pred_flow).detach() / 2

            # need to convert x-space to v-space

            if is_x_space:
                pred = (pred - noised_latents) / (1. - first_times)
                maybe_shortcut_loss_weight = (1. - first_times) ** 2

        # mse loss

        flow_losses = F.mse_loss(pred, pred_target, reduction = 'none')

        flow_losses = flow_losses * maybe_shortcut_loss_weight # handle the (1-t)^2 in eq(7)

        # loss weighting with their ramp function

        if exists(self.loss_weight_fn):
            loss_weight = self.loss_weight_fn(times)
            flow_losses = flow_losses * loss_weight

        flow_loss = flow_losses.mean()

        # now take care of the agent token losses

        reward_loss = self.zero

        if exists(rewards):

            if rewards.ndim == 2: # (b t)
                encoded_agent_tokens = reduce(encoded_agent_tokens, 'b t g d -> b t d', 'mean')

            reward_pred = self.to_reward_pred(encoded_agent_tokens[:, :-1])

            reward_pred = rearrange(reward_pred, 'mtp b t l -> b l t mtp')

            reward_targets, reward_loss_mask = create_multi_token_prediction_targets(two_hot_encoding, self.multi_token_pred_len)

            reward_targets = rearrange(reward_targets, 'b t mtp l -> b l t mtp')

            reward_losses = F.cross_entropy(reward_pred, reward_targets, reduction = 'none')

            reward_losses = reward_losses.masked_fill(reward_loss_mask, 0.)

            reward_loss = reduce(reward_losses, '... mtp -> mtp', 'mean') # they sum across the prediction steps (mtp dimension) - eq(9)

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
                continuous_action_targets, continuous_mask = create_multi_token_prediction_targets(discrete_actions, self.multi_token_pred_len)
                continuous_action_targets = rearrange(continuous_action_targets, 'b t mtp ... -> mtp b t ...')
                continuous_mask = rearrange(continuous_mask, 'b t mtp -> mtp b t')

            discrete_log_probs, continuous_log_probs = self.action_embedder.log_probs(
                policy_embed,
                discrete_targets = discrete_action_targets if exists(discrete_actions) else None,
                continuous_targets = continuous_action_targets if exists(continuous_actions) else None
            )

            if exists(discrete_log_probs):
                discrete_log_probs = discrete_log_probs.masked_fill(~discrete_mask[..., None], 0.)

                discrete_action_loss = reduce(-discrete_log_probs, 'mtp b t na -> mtp', 'mean')

            if exists(continuous_log_probs):
                continuous_log_probs = continuous_log_probs.masked_fill(~continuous_mask[..., None], 0.)

                continuous_action_loss = reduce(-continuous_log_probs, 'mtp b t na -> mtp', 'mean')

        # handle loss normalization

        losses = WorldModelLosses(flow_loss, reward_loss, discrete_action_loss, continuous_action_loss)

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
            (continuous_action_loss * self.continuous_action_loss_weight).sum()
        )

        if not return_all_losses:
            return total_loss

        return total_loss, losses

# dreamer

class Dreamer(Module):
    def __init__(
        self,
        state_tokenizer: VideoTokenizer,
        world_model: DynamicsWorldModel,
    ):
        super().__init__()
        self.state_toke = state_tokenizer
        self.world_model = world_model

    def interact_with_sim(self, env) -> Experience:
        raise NotImplementedError
