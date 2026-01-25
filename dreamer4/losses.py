from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential
from torch import Tensor, arange, ones, rand, randn

from einops import rearrange

import torchvision
from torchvision.models import VGG16_Weights

# Local imports
from .utils import (
    default,
    exists,
    ramp_weight,
)


class LossNormalizer(Module):

    # the authors mentioned the need for loss normalization in the dynamics transformer and video tokenizer

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

