# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fire",
#     "accelerate",
#     "dreamer4"
# ]
# ///

import fire
from accelerate import Accelerator

import torch
from torch import zeros, tensor
from torch.optim import Adam
from einops import rearrange

from dreamer4.dreamer4 import DynamicsWorldModel

# helpers

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

# main

def main(
    is_continuous = False,
    cpu = False
):
    accelerator = Accelerator(cpu = cpu)
    device = accelerator.device
    print = accelerator.print

    print(f"Testing {'Continuous' if is_continuous else 'Discrete'} Actions Autoregression")

    num_discrete = () if is_continuous else (4,)
    num_continuous_actions = 1 if is_continuous else 0

    model = DynamicsWorldModel(
        dim = 16,
        dim_latent = 16,
        num_latent_tokens = 64,
        num_spatial_tokens = 8,
        depth = 4,
        time_block_every = 2,
        value_head_mlp_depth = 1,
        policy_head_mlp_depth = 1,
        attn_heads = 4,
        num_continuous_actions = num_continuous_actions,
        num_discrete_actions = num_discrete,
        multi_token_pred_len = 1,
        use_loss_normalization = False
    ).to(device)

    optimizer = Adam(model.parameters(), lr = 3e-4)

    # mock actions

    if is_continuous:
        actions_seq = [0.1, 0.5, -0.2, 0.8] * 2
    else:
        actions_seq = [1, 2, 3, 0] * 2

    actions_tensor = tensor(actions_seq, device = device)
    actions_tensor = rearrange(actions_tensor, 't -> 1 t 1')

    # mock state consisting of all zeros

    latents = zeros(1, 8, 64, 16, device = device)

    model.train()

    for i in range(501):
        optimizer.zero_grad()

        loss = model(
            latents = latents,
            discrete_actions = None if is_continuous else actions_tensor,
            continuous_actions = actions_tensor if is_continuous else None,
            add_autoregressive_action_loss = True
        )

        loss.backward()
        optimizer.step()

        if divisible_by(i, 100):
            print(f"Step {i} Loss: {loss.item():.4f}")

    print(f"\nTraining complete. Verifying...")

    model.eval()

    with torch.no_grad():
        prompt_latents = latents[:, 0:1]
        prompt_actions = actions_tensor[:, 0:1]

        # evaluate the prompt and generate 7 more steps

        gen = model.generate(
            time_steps = 8,
            num_steps = 4,
            prompt_latents = prompt_latents,
            prompt_discrete_actions = None if is_continuous else prompt_actions,
            prompt_continuous_actions = prompt_actions if is_continuous else None,
            context_signal_noise = 0.,
            return_agent_actions = True,
            discrete_temperature = 0., # 0. temperature is greedy / argmax
            continuous_temperature = 0.
        )

        generated = gen.actions.continuous if is_continuous else gen.actions.discrete
        generated_actions = rearrange(generated, '1 t 1 -> t').tolist() if exists(generated) else []

        if is_continuous:
            generated_actions = [round(v, 3) for v in generated_actions]

        print(f"Target: {actions_seq}")
        print(f"Preds:  {generated_actions}")

if __name__ == '__main__':
    fire.Fire(main)
