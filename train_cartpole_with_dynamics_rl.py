# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "gymnasium[classic_control]",
#     "tqdm",
#     "ninja",
#     "dreamer4",
#     "fire",
#     "wandb"
# ]
# [tool.uv.sources]
# dreamer4 = { path = "." }
# ///

import fire
from collections import deque
from functools import partial

import wandb

import gymnasium as gym

from pathlib import Path
import shutil
from torchvision.utils import save_image

from tqdm import tqdm
import numpy as np

import torch
from torch import nn, stack, tensor, is_tensor, zeros
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
pad_sequence = partial(pad_sequence, batch_first = True)

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from accelerate import Accelerator

from dreamer4.dreamer4 import (
    DynamicsWorldModel,
    Experience,
    Actions,
    divisible_by,
    combine_experiences,
    exists,
    default,
    cast_to_tensor,
    VideoTokenizer
)

# env

class ImageObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        img_tensor = render_frame(self.env)
        img_tensor = rearrange(img_tensor, '1 c h w -> c h w')
        return dict(state = obs, image = img_tensor)

def make_env(seed, use_image_input = False, vectorized = False, num_envs = 8):
    env_kwargs = dict(render_mode = 'rgb_array') if use_image_input else dict()

    if vectorized:
        env = gym.make_vec('CartPole-v1', num_envs = num_envs, **env_kwargs)
        env = gym.wrappers.vector.RecordEpisodeStatistics(env)
    else:
        env = gym.make('CartPole-v1', **env_kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)

    if exists(seed):
        env.action_space.seed(seed)

    if use_image_input:
        if vectorized:
            raise NotImplementedError('Image observation wrapper not yet implemented for vectorized environments')
        env = ImageObservationWrapper(env)

    return env

# helpers

def render_frame(env):
    img = env.render()
    img = tensor(img, dtype = torch.float32, device = 'cpu')
    img = rearrange(img, 'h w c -> 1 c h w')
    img = F.interpolate(img, size = (64, 64), mode = 'bilinear', align_corners = False)
    img = img / 255.0
    return img

# agent

class TransformerPPOAgent(nn.Module):
    def __init__(
        self,
        use_asym_critic = False,
        agent_value_gradient_frac = 1.0,
        agent_policy_gradient_frac = 1.0,
        use_image_input = False,
        use_time_rnn = False
    ):
        super().__init__()
        self.use_image_input = use_image_input

        tokenizer = None
        if use_image_input:
            tokenizer = VideoTokenizer(
                channels = 3,
                patch_size = 8,
                dim = 64,
                dim_latent = 32,
                num_latent_tokens = 2,
                image_height = 64,
                image_width = 64,
                encoder_depth = 2,
                decoder_depth = 2,
                time_block_every = 2,
                use_causal_conv3d = True,
                use_time_rnn = use_time_rnn,
                use_loss_normalization = True
            )

        self.dynamics = DynamicsWorldModel(
            video_tokenizer = tokenizer,
            dim = 128,
            dim_latent = 32,
            dim_state = None if use_image_input else 4,
            num_latent_tokens = 2,
            num_spatial_tokens = 4,
            num_register_tokens = 1,
            num_discrete_actions = 2,
            use_time_rnn = use_time_rnn,
            transformer_kwargs = dict(
                use_attn_pool = False
            ),
            depth = 3,
            time_block_every = 3,
            policy_head_mlp_depth = 2,
            value_head_mlp_depth = 2,
            gae_discount_factor = 0.99,
            ppo_eps_clip = 0.2,
            agent_value_gradient_frac = agent_value_gradient_frac,
            agent_policy_gradient_frac = agent_policy_gradient_frac,
            normalize_advantages = True,
            use_loss_normalization = False,
            attn_heads = 4,
            attn_dim_head = 16,
            reward_encoder_kwargs = dict(reward_range = (-10., 10.)),
            dim_critic_state = 4 if use_asym_critic else None
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def num_discrete_actions(self):
        return self.dynamics.action_embedder.num_discrete_actions.sum().item()

# experience processing

def slice_experience(exp, idx):
    slice_maybe = lambda t: t[idx] if is_tensor(t) else t

    old_action_unembeds = tuple(slice_maybe(t) for t in exp.old_action_unembeds) if exists(exp.old_action_unembeds) else None

    actions = Actions(slice_maybe(exp.actions.discrete), slice_maybe(exp.actions.continuous)) if exists(exp.actions) else None
    log_probs = Actions(slice_maybe(exp.log_probs.discrete), slice_maybe(exp.log_probs.continuous)) if exists(exp.log_probs) else None

    return Experience(
        latents = slice_maybe(exp.latents),
        video = slice_maybe(exp.video),
        critic_state = slice_maybe(exp.critic_state),
        rewards = slice_maybe(exp.rewards),
        terminals = slice_maybe(exp.terminals),
        actions = actions,
        log_probs = log_probs,
        old_action_unembeds = old_action_unembeds,
        values = slice_maybe(exp.values),
        step_size = slice_maybe(exp.step_size),
        is_truncated = slice_maybe(exp.is_truncated),
        lens = slice_maybe(exp.lens),
        agent_index = slice_maybe(exp.agent_index),
        is_from_world_model = slice_maybe(exp.is_from_world_model),
        episode_return = slice_maybe(exp.episode_return)
    )

# training loop

def main(
    batch_size = 8,
    grad_accum_every = 1,
    update_episodes = 16,
    update_epochs = 4,
    num_episodes = 5000,
    max_timesteps = 500,
    learning_rate = 3e-4,
    max_grad_norm = 0.5,
    target_return = 70.0,
    use_asym_critic = True,
    max_policy_updates = 250,
    agent_value_gradient_frac = 0.1,
    agent_policy_gradient_frac = 0.1,
    seed = 42,
    use_wandb = False,
    use_pmpo = False,
    use_image_input = False,
    vectorized = False,
    num_envs = 8,
    cpu = False,
    use_time_rnn = True,
    ssl_every_rl_updates = 2,
    ssl_max_epochs = 25,
    ssl_target_recon_loss = None,
    ssl_batch_size = 8,
    ssl_max_memories = 64,
    tokenizer_learning_rate = 1e-4
):
    torch.manual_seed(seed)

    if vectorized:
        update_episodes = max(1, update_episodes // num_envs)

    assert divisible_by(update_episodes, batch_size) if not vectorized else True

    if use_image_input:
        use_asym_critic = True

    if use_wandb:
        wandb.init(project = 'dreamer4-cartpole')

    accelerator = Accelerator(
        gradient_accumulation_steps = grad_accum_every,
        cpu = cpu
    )

    device = accelerator.device

    def log(msg):
        accelerator.print(msg)

    results_folder = Path('./results')
    shutil.rmtree(results_folder, ignore_errors=True)
    results_folder.mkdir(exist_ok = True, parents = True)
    
    log(f'\nReconstruction grids will be saved directly to {results_folder.absolute()}\n')

    env = make_env(seed, use_image_input, vectorized=vectorized, num_envs=num_envs)

    agent = TransformerPPOAgent(
        use_asym_critic = use_asym_critic,
        agent_value_gradient_frac = agent_value_gradient_frac,
        agent_policy_gradient_frac = agent_policy_gradient_frac,
        use_image_input = use_image_input,
        use_time_rnn = use_time_rnn
    ).to(device)

    # optimizers

    has_tokenizer = use_image_input and exists(agent.dynamics.video_tokenizer)
    should_ssl = has_tokenizer and ssl_every_rl_updates > 0

    if should_ssl:
        tokenizer = agent.dynamics.video_tokenizer
        tokenizer_params = set(tokenizer.parameters())
        agent_params = [p for p in agent.parameters() if p not in tokenizer_params]

        tokenizer_optim = AdamW(tokenizer.parameters(), lr = tokenizer_learning_rate)
        tokenizer_optim = accelerator.prepare(tokenizer_optim)
    else:
        agent_params = agent.parameters()

    optimizer = AdamW(agent_params, lr = learning_rate)
    agent, optimizer = accelerator.prepare(agent, optimizer)

    # rollout state

    recent_returns = deque(maxlen = 20)
    memories = deque(maxlen = update_episodes)
    ssl_memories = deque(maxlen = ssl_max_memories) if should_ssl else None

    pbar = tqdm(range(num_episodes), desc = 'episodes')
    agent.eval()
    num_policy_updates = 0

    for episode in pbar:

        # collect experience

        experience = agent.dynamics.interact_with_env(
            env = env,
            seed = seed if episode == 0 else None,
            max_timesteps = max_timesteps,
            env_is_vectorized = vectorized,
            store_agent_embed = False,
            store_old_action_unembeds = True
        )

        # track returns

        episode_return = experience.episode_return.mean().item()
        recent_returns.append(episode_return)

        # store to replay buffers

        cpu_experience = experience.to('cpu')

        memories.append(cpu_experience)

        if exists(ssl_memories):
            ssl_memories.append(cpu_experience)

        avg_ret = np.mean(recent_returns) if len(recent_returns) > 0 else 0.0

        if avg_ret >= target_return:
            log(f'\n✅ Target average return of {target_return} reached! (avg={avg_ret:.1f}) Stopping training.')
            break

        loss_metrics = dict(average_return = f'{avg_ret:.1f}')

        # determine update schedule

        is_rl_update = divisible_by(episode + 1, update_episodes)

        if is_rl_update:
            num_rl_updates = (episode + 1) // update_episodes
            should_do_ssl_now = should_ssl and divisible_by(num_rl_updates - 1, ssl_every_rl_updates)
        else:
            should_do_ssl_now = False

        # prepare ssl video from accumulated buffer

        ssl_video = None
        ssl_lens = None

        if should_do_ssl_now:
            ssl_exp_batch = combine_experiences(list(ssl_memories))
            if exists(ssl_exp_batch.video):
                ssl_video = ssl_exp_batch.video.clone().to(device)
                ssl_lens = ssl_exp_batch.lens.clone().to(device) if exists(ssl_exp_batch.lens) else None

        # rl update from most recent experiences

        if is_rl_update:
            agent.train()

            exp_batch = combine_experiences(list(memories)).to(device)

            total_envs = exp_batch.latents.shape[0] if exists(exp_batch.latents) else exp_batch.critic_state.shape[0]

            epoch_policy_loss = 0.
            epoch_value_loss = 0.

            for _ in range(update_epochs):
                batches = torch.randperm(total_envs, device = device).split(batch_size)

                for i, batch_idx in enumerate(batches):
                    micro = slice_experience(exp_batch, batch_idx)

                    policy_loss, value_loss = agent.dynamics.learn_from_experience(
                        experience = micro,
                        only_learn_policy_value_heads = False,
                        use_pmpo = use_pmpo
                    )

                    total_loss = (policy_loss + value_loss) / grad_accum_every
                    accelerator.backward(total_loss)

                    is_last_batch = (i + 1) == len(batches)

                    if divisible_by(i + 1, grad_accum_every) or is_last_batch:
                        accelerator.clip_grad_norm_(agent.parameters(), max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    epoch_policy_loss += policy_loss.item() / len(batches)
                    epoch_value_loss += value_loss.item() / len(batches)

            avg_policy_loss = epoch_policy_loss / update_epochs
            avg_value_loss = epoch_value_loss / update_epochs

            loss_metrics.update(policy_loss = f'{avg_policy_loss:.3f}', value_loss = f'{avg_value_loss:.3f}')

            if use_wandb:
                wandb.log(dict(
                    average_return = avg_ret,
                    policy_loss = avg_policy_loss,
                    value_loss = avg_value_loss
                ))

            num_policy_updates += 1
            memories.clear()

            if num_policy_updates >= max_policy_updates:
                log(f'\nReached {num_policy_updates} PPO updates! Stopping training.')
                break

            agent.eval()

        # autoencoder ssl training

        if should_do_ssl_now and exists(ssl_video):
            agent.train()

            epoch_recon_loss = 0.

            ssl_pbar = tqdm(range(ssl_max_epochs), desc = 'ssl epochs', leave = False)
            for epoch in ssl_pbar:
                batches = torch.randperm(ssl_video.shape[0], device = device).split(ssl_batch_size)
                
                epoch_recon_loss = 0.

                for i, batch_indices in enumerate(batches):
                    is_last_batch = (i + 1) == len(batches)
                    batch_video = ssl_video[batch_indices]
                    batch_lens = ssl_lens[batch_indices] if exists(ssl_lens) else None

                    loss, (losses, recon_video) = tokenizer(
                        batch_video,
                        time_lens = batch_lens,
                        update_loss_ema = True,
                        return_intermediates = True
                    )

                    loss = loss / grad_accum_every
                    accelerator.backward(loss)

                    if divisible_by(i + 1, grad_accum_every) or is_last_batch:
                        accelerator.clip_grad_norm_(tokenizer.parameters(), max_grad_norm)
                        tokenizer_optim.step()
                        tokenizer_optim.zero_grad()

                    epoch_recon_loss += losses.recon.item() / len(batches)
                    
                ssl_pbar.set_postfix(recon_loss = f'{epoch_recon_loss:.4f}')
                    
                if exists(ssl_target_recon_loss) and ssl_target_recon_loss > 0:
                    if epoch_recon_loss <= ssl_target_recon_loss:
                        log(f'\nSSL recon loss reached target {epoch_recon_loss:.4f} <= {ssl_target_recon_loss:.4f} at epoch {epoch + 1}! Stopping SSL early.')
                        break

            avg_recon_loss = epoch_recon_loss
            loss_metrics.update(recon = f'{avg_recon_loss:.3f}')

            if use_wandb:
                wandb.log(dict(recon_loss = avg_recon_loss))

            with torch.no_grad():
                agent.eval()
                sample_video = ssl_video[:1]
                sample_lens = ssl_lens[:1] if exists(ssl_lens) else None

                _, (_, sample_recon) = tokenizer(
                    sample_video,
                    time_lens = sample_lens,
                    mask_patches = False,
                    return_intermediates = True
                )

                orig = rearrange(sample_video[0], 'c t h w -> t c h w')[:8]
                recon = rearrange(sample_recon[0].clamp(0., 1.), 'c t h w -> t c h w')[:8]
                grid = torch.cat((orig, recon), dim = 0)
                save_image(grid, str(results_folder / f'recon_ep{episode}.png'), nrow = orig.shape[0])

            agent.eval()

        pbar.set_postfix(loss_metrics)

    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    fire.Fire(main)
