from __future__ import annotations

from functools import partial
import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, RMSNorm, Identity
from torch import nn


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
