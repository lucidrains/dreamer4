from __future__ import annotations

from functools import partial
import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, Sequential, RMSNorm, Identity
from torch import nn, cat, stack

from einops import rearrange, reduce, pack
from einops.layers.torch import Rearrange
from hyper_connections import mc_get_init_and_expand_reduce_stream_functions

from collections import namedtuple

# flex attention - optional
flex_attention = None
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    # Note: Not using torch.compile due to recompilation issues with dynamic shapes
    # Using native flex_attention instead (will use unfused implementation)
    # Original: flex_attention = torch.compile(flex_attention, dynamic=True)
except ImportError:
    pass

# Local imports
from .utils import (
    default,
    divisible_by,
    exists,
    get_attend_fn,
    maybe,
    pack_one,
    safe_stack,
)
from .attention import Rotary1D, Attention
from .layers import SwiGLUFeedforward, GRULayer

# Constants and Type Aliases

LinearNoBias = partial(Linear, bias = False)
TransformerIntermediates = namedtuple('TransformerIntermediates', ('next_kv_cache', 'normed_time_inputs', 'normed_space_inputs', 'next_rnn_hiddens'))

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

        hyper_conn, self.expand_streams, self.reduce_streams = mc_get_init_and_expand_reduce_stream_functions(num_residual_streams, dim = dim)

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

