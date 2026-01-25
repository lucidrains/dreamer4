from __future__ import annotations

import torch
from torch import Tensor, cat, stack, tensor, is_tensor, full

from dataclasses import dataclass, asdict

from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten

# Local imports
from .utils import (
    exists,
    first,
    pad_tensors_at_dim_to_max_len,
)

# Type Aliases
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
    exps: list[Experience]
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
