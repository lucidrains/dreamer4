from __future__ import annotations

from functools import partial
import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, Parameter, Sequential
from torch import nn, cat, stack, tensor, rand, randn, empty, linspace

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

import einx
from x_mlps_pytorch.normed_mlp import create_mlp
from hyper_connections import mc_get_init_and_expand_reduce_stream_functions
from vit_pytorch.vit_with_decorr import DecorrelationLoss

from collections import namedtuple

from .attention import flex_attention

from .utils import (
    default,
    divisible_by,
    exists,
    pack_one,
)
from .transformers import AxialSpaceTimeTransformer
from .losses import LPIPSLoss, LossNormalizer

# Constants and Type Aliases

LinearNoBias = partial(Linear, bias = False)
VideoTokenizerIntermediates = namedtuple('VideoTokenizerIntermediates', ('losses', 'recon'))
TokenizerLosses = namedtuple('TokenizerLosses', ('recon', 'lpips', 'time_decorr', 'space_decorr'))

class VideoTokenizer(Module):
    def __init__(
        self,
        dim,
        dim_latent,
        patch_size,
        image_height = None,
        image_width = None,
        num_latent_tokens = 64,
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
        use_loss_normalization = True,  # PAPER: RMS loss normalization from Dreamer v4 section 3
    ):
        super().__init__()

        self.patch_size = patch_size

        # special tokens

        assert num_latent_tokens >= 1
        self.num_latent_tokens = num_latent_tokens
        self.latent_tokens = Parameter(randn(num_latent_tokens, dim) * 1e-2)

        # hyper connections

        hyper_conn, self.expand_streams, self.reduce_streams = mc_get_init_and_expand_reduce_stream_functions(num_residual_streams, dim = dim)

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

        self.loss_normalizer = LossNormalizer(num_losses=2)

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
        video_or_image, # (b c t h w) | (b c h w)
        return_latents = False,
        mask_patches = None,
        return_intermediates = False,
    ):

        # handle image pretraining

        is_image = video_or_image.ndim == 4

        if is_image:
            video = rearrange(video_or_image, 'b c h w -> b c 1 h w')
        else:
            video = video_or_image

        # shapes

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


        # Apply loss normalization before multiplying loss weights

        losses_to_normalize = torch.stack([recon_loss, lpips_loss])
        recon_loss_norm, lpips_loss_norm = self.loss_normalizer(losses_to_normalize)

        total_loss = (
            recon_loss_norm +
            lpips_loss_norm * self.lpips_loss_weight +
            time_decorr_loss * self.decorr_aux_loss_weight +
            space_decorr_loss * self.decorr_aux_loss_weight
        )

        if not return_intermediates:
            return total_loss

        losses = TokenizerLosses(recon_loss, lpips_loss, time_decorr_loss, space_decorr_loss)

        # handle returning of reconstructed, and image pretraining

        if is_image:
            recon_video = rearrange(recon_video, 'b c 1 h w -> b c h w')

        out = (losses, recon_video)

        return total_loss, out

