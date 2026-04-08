# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "gymnasium[classic_control]",
#     "memmap-replay-buffer",
#     "tqdm",
#     "x-mlps-pytorch",
#     "ninja",
#     "dreamer4",
#     "opencv-python-headless",
#     "pygame",
#     "fire",
#     "wandb"
# ]
# [tool.uv.sources]
# dreamer4 = { path = "." }
# ///

import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from collections import deque, namedtuple
from functools import partial

import wandb
import numpy as np

import torch
from torch import nn, stack, tensor, is_tensor
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
pad_sequence = partial(pad_sequence, batch_first = True)

from einops import rearrange, repeat
from tqdm import tqdm
import gymnasium as gym
import fire

from accelerate import Accelerator

from dreamer4.dreamer4 import (
    DynamicsWorldModel, 
    VideoTokenizer, 
    Experience, 
    Actions, 
    divisible_by
)

# constants

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# memory tuple

Memory = namedtuple('Memory', [
    'eps',
    'obs',
    'critic_state',
    'action',
    'log_prob',
    'reward',
    'is_boundary',
    'value',
    'done'
])

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# wrappers

class VideoFrameProcess(gym.ObservationWrapper):
    def __init__(self, env, render_live = False):
        super().__init__(env)
        self.render_live = render_live
        
        self.observation_space = gym.spaces.Box(
            low = 0, high = 255, shape = (3, 64, 64), dtype = np.uint8
        )
        
    def _render_frame(self):
        import cv2
        frame = self.env.render()
        frame = cv2.resize(frame, (64, 64))
        return rearrange(frame, 'h w c -> c h w')
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['state'] = obs
        return self._render_frame(), info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['state'] = obs
        return self._render_frame(), reward, terminated, truncated, info

class NormalizeImage(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low = -1.0, high = 1.0, shape = (3, 64, 64), dtype = np.float32
        )

    def observation(self, obs):
        return obs.astype(np.float32) / 255. * 2. - 1.

def make_env(seed, is_first=True):
    env = gym.make('CartPole-v1', render_mode = 'rgb_array')
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = VideoFrameProcess(env, render_live = is_first)
    env = NormalizeImage(env)
    env.action_space.seed(seed)
    return env

# agent

class TransformerPPOAgent(nn.Module):
    def __init__(self, use_asym_critic = False, agent_value_gradient_frac = 1.0, agent_policy_gradient_frac = 1.0):
        super().__init__()
        
        self.tokenizer = VideoTokenizer(
            dim = 96,
            dim_latent = 64,
            patch_size = 8,
            image_height = 64,
            image_width = 64,
            num_latent_tokens = 16,
            encoder_depth = 3,
            decoder_depth = 3,
            time_block_every = 3,
            attn_heads = 4,
            attn_dim_head = 32,
            use_causal_conv3d = True
        )
        
        self.dynamics = DynamicsWorldModel(
            dim = 128,
            dim_latent = self.tokenizer.dim_latent,
            num_latent_tokens = self.tokenizer.num_latent_tokens,
            num_discrete_actions = 2,
            video_tokenizer = self.tokenizer,
            depth = 1,
            time_block_every = 1,
            policy_head_mlp_depth = 2,
            value_head_mlp_depth = 2,
            gae_discount_factor = 0.99,
            ppo_eps_clip = 0.2,
            agent_value_gradient_frac = agent_value_gradient_frac,
            agent_policy_gradient_frac = agent_policy_gradient_frac,
            normalize_advantages = True,
            use_loss_normalization = False,
            attn_heads = 4,
            attn_dim_head = 32,
            reward_encoder_kwargs = dict(reward_range = (-10., 10.)),
            dim_critic_state = 4 if use_asym_critic else None
        )
        
    def forward(self, obs, critic_state = None, past_action = None, time_cache = None):
        tokenizer_cache, dynamics_cache = default(time_cache, (None, None))
        
        with torch.no_grad():
            latents, tokenizer_cache = self.tokenizer(
                obs,
                return_latents = True,
                time_cache = tokenizer_cache,
                return_time_cache = True
            )
            
            b = obs.shape[0]
            step_size = torch.ones((b,), device = obs.device)
            
            _, (embeds, next_dynamics_cache) = self.dynamics(
                latents = latents,
                signal_levels = self.dynamics.max_steps - 1,
                step_sizes = step_size,
                discrete_actions = past_action,
                time_cache = dynamics_cache,
                latent_is_noised = True,
                return_pred_only = True,
                return_intermediates = True
            )
            
            agent_embed = embeds.agent[:, -1]
            policy_embed = self.dynamics.policy_head(agent_embed)
            
            value_agent_embed = agent_embed

            if exists(self.dynamics.critic_state_embedder) and exists(critic_state):
                critic_embeds = self.dynamics.critic_state_embedder(critic_state)

                if critic_embeds.ndim == 2 and value_agent_embed.ndim == 3:
                    critic_embeds = rearrange(critic_embeds, 'b d -> b 1 d')

                value_agent_embed = value_agent_embed + critic_embeds

            value_bins = self.dynamics.value_head(value_agent_embed)
            
            value = self.dynamics.reward_encoder.bins_to_scalar_value(value_bins)
            
            discrete_action, _ = self.dynamics.action_embedder.sample(policy_embed, pred_head_index = 0, squeeze = True)
            log_prob, _        = self.dynamics.action_embedder.log_probs(policy_embed, pred_head_index = 0, discrete_targets = discrete_action)
                
            return discrete_action, log_prob, value, (tokenizer_cache, next_dynamics_cache)

# experience processing

def construct_padded_experience(memories, episode_lens, is_episode_truncated, device):
    def stack_and_to_device(t):
        return stack(t).to(device)

    def stack_memories(episode_memories):
        return tuple(map(stack_and_to_device, zip(*episode_memories)))

    memories = map(stack_memories, memories)
    
    (
        episodes,
        obs_seq,
        critic_state_seq,
        actions_seq,
        log_probs_seq,
        rewards_seq,
        is_boundaries_seq,
        values_seq,
        dones_seq
    ) = tuple(map(pad_sequence, zip(*memories)))
    
    num_envs = obs_seq.shape[0]
    
    exp_obs = rearrange(obs_seq, 'e t c h w -> e c t h w')
    exp_critic_states = critic_state_seq
    exp_actions = rearrange(actions_seq, 'e t -> e t 1')
    exp_log_probs = rearrange(log_probs_seq, 'e t -> e t 1')
    exp_rewards = rewards_seq
    exp_values = values_seq
    exp_terminals = dones_seq

    step_sizes = repeat(tensor([1.], device = device), '1 -> e', e = num_envs)

    lens_tensor = stack(tuple(episode_lens)).to(device)
    is_truncated_tensor = stack(tuple(is_episode_truncated)).to(device)
    
    return Experience(
        latents = None,
        video = exp_obs,
        critic_state = exp_critic_states,
        rewards = exp_rewards,
        terminals = exp_terminals,
        actions = Actions(exp_actions, None),
        log_probs = Actions(exp_log_probs, None),
        values = exp_values,
        step_size = step_sizes,
        is_truncated = is_truncated_tensor,
        lens = lens_tensor
    )

def extract_micro_experience(train_exp, b_idx):
    return Experience(
        latents = None,
        video = train_exp.video[b_idx],
        critic_state = train_exp.critic_state[b_idx] if exists(train_exp.critic_state) else None,
        rewards = train_exp.rewards[b_idx],
        terminals = train_exp.terminals[b_idx],
        actions = Actions(train_exp.actions.discrete[b_idx], None),
        log_probs = Actions(train_exp.log_probs.discrete[b_idx], None),
        values = train_exp.values[b_idx],
        step_size = train_exp.step_size[b_idx],
        is_truncated = train_exp.is_truncated[b_idx],
        lens = train_exp.lens[b_idx] if exists(train_exp.lens) else None
    )

# training loop

def main(
    batch_size = 2,
    grad_accum_every = 16,
    update_episodes = 64,
    update_epochs = 2,
    num_episodes = 50000,
    max_timesteps = 500,
    learning_rate = 3e-4,
    max_grad_norm = 0.5,
    target_return = 100.0,
    use_asym_critic = True,
    max_policy_updates = 250,
    max_memories_factor = 6,
    agent_value_gradient_frac = 0.1,
    agent_policy_gradient_frac = 0.1,
    seed = 42
):
    torch.manual_seed(seed)

    assert divisible_by(update_episodes, batch_size)

    wandb.init(project = "dreamer4-cartpole")

    accelerator = Accelerator(gradient_accumulation_steps = grad_accum_every)
    device = accelerator.device

    def log(msg):
        accelerator.print(msg)

    # envs

    env = make_env(seed, is_first=True)

    # agent and optimizer

    agent = TransformerPPOAgent(
        use_asym_critic = use_asym_critic,
        agent_value_gradient_frac = agent_value_gradient_frac,
        agent_policy_gradient_frac = agent_policy_gradient_frac
    ).to(device)

    optimizer = AdamW(agent.parameters(), lr = learning_rate)

    agent, optimizer = accelerator.prepare(agent, optimizer)

    recent_returns = deque(maxlen = 20)
    memories = deque(maxlen = update_episodes * max_memories_factor)
    episode_lens = deque(maxlen = update_episodes * max_memories_factor)
    is_episode_truncated = deque(maxlen = update_episodes * max_memories_factor)

    pbar_episodes = tqdm(range(num_episodes), desc = 'episodes')
    
    agent.eval()

    num_policy_updates = 0

    for eps in pbar_episodes:
        one_episode_memories = deque([])
        eps_tensor = tensor(eps)

        env_obs, info = env.reset(seed = seed)
        state = tensor(env_obs, dtype = torch.float32, device = 'cpu')
        critic_state = tensor(info['state'], dtype = torch.float32, device = 'cpu')

        time_cache = None
        past_action = None

        @torch.no_grad()
        def state_to_pred_action_and_value(obs, crit_state, last_action, time_cache):
            obs = obs.unsqueeze(0).to(device)
            crit_state = crit_state.unsqueeze(0).to(device)
            
            last_action_tensor = None
            if last_action is not None:
                last_action_tensor = tensor([[[last_action]]], dtype=torch.long, device=device)
                
            action, log_prob, value, time_cache = agent(
                obs, 
                critic_state = crit_state, 
                past_action = last_action_tensor, 
                time_cache = time_cache
            )
            return action.squeeze(), log_prob.squeeze(), value.squeeze(), time_cache

        for timestep in range(max_timesteps):
            action, log_prob, value, time_cache = state_to_pred_action_and_value(state, critic_state, past_action, time_cache)
            
            next_obs, reward, terminated, truncated, infos = env.step(action.item())
            
            next_state = tensor(next_obs, dtype = torch.float32, device = 'cpu')
            next_critic_state = tensor(infos['state'], dtype = torch.float32, device = 'cpu')

            done = terminated or truncated

            memory = Memory(
                eps_tensor.cpu(), 
                state, 
                critic_state, 
                action.cpu(), 
                log_prob.cpu(), 
                tensor(reward, dtype=torch.float32, device='cpu'),
                tensor(False, dtype=torch.bool, device='cpu'),
                value.cpu(), 
                tensor(terminated, dtype=torch.bool, device='cpu')
            )

            one_episode_memories.append(memory)

            past_action = action.item()
            state = next_state
            critic_state = next_critic_state

            if done and not terminated:
                # bootstrap the value safely via placeholder to let parallel target loop compute it later
                bootstrap_value_memory = memory._replace(
                    obs = state,
                    critic_state = critic_state,
                    eps = tensor(-1, device='cpu'),
                    is_boundary = tensor(True, dtype=torch.bool, device='cpu'),
                    value = tensor(0., dtype=torch.float32, device='cpu'),
                    reward = tensor(0., dtype=torch.float32, device='cpu'),
                    done = tensor(False, dtype=torch.bool, device='cpu')
                )

                one_episode_memories.append(bootstrap_value_memory)

            if done:
                if exists(infos.get('episode')):
                    ep_ret = infos['episode']['r']
                    ep_ret = ep_ret.item() if is_tensor(ep_ret) else float(ep_ret)
                    recent_returns.append(ep_ret)
                break

        episode_lens.append(tensor(len(one_episode_memories)))
        is_episode_truncated.append(tensor(truncated, dtype=torch.bool, device='cpu'))
        memories.append(one_episode_memories)

        avg_ret = np.mean(recent_returns) if len(recent_returns) > 0 else 0.0

        if avg_ret >= target_return:
            log(f'\nTarget average return of {target_return} reached! Stopping training.')
            break

        loss_metrics = dict(avg_return = f'{avg_ret:.1f}')
        
        if divisible_by(eps + 1, update_episodes):
            agent.train()

            exp = construct_padded_experience(memories, episode_lens, is_episode_truncated, device)
            total_envs = exp.video.shape[0]

            # recalculate values for old memories

            @torch.no_grad()
            def recalculate_targets(micro_exp):
                latents = agent.tokenizer(micro_exp.video, return_latents=True)
                
                _, (embeds, _) = agent.dynamics(
                    latents = latents,
                    signal_levels = agent.dynamics.max_steps - 1,
                    step_sizes = micro_exp.step_size,
                    discrete_actions = micro_exp.actions.discrete,
                    latent_is_noised = True,
                    return_pred_only = True,
                    return_intermediates = True
                )
                
                agent_embeds = embeds.agent[..., 0, :]
                value_agent_embeds = agent_embeds

                if exists(agent.dynamics.critic_state_embedder) and exists(micro_exp.critic_state):
                    critic_embeds = agent.dynamics.critic_state_embedder(micro_exp.critic_state)
                    value_agent_embeds = value_agent_embeds + critic_embeds
                    
                value_bins = agent.dynamics.value_head(value_agent_embeds)
                values = agent.dynamics.reward_encoder.bins_to_scalar_value(value_bins)
                
                policy_embed = agent.dynamics.policy_head(agent_embeds)
                log_probs, _ = agent.dynamics.action_embedder.log_probs(
                    policy_embed,
                    pred_head_index=0,
                    discrete_targets=micro_exp.actions.discrete
                )
                
                return values, log_probs

            agent.eval()

            splits = torch.arange(total_envs, device=device).split(batch_size * 2) 
            recalculated = [recalculate_targets(extract_micro_experience(exp, b_idx)) for b_idx in splits]
            
            new_values, new_log_probs = tuple(map(torch.cat, zip(*recalculated)))
            
            exp.values = new_values
            exp.log_probs = Actions(new_log_probs, None)
            
            agent.train()

            epoch_pi_loss = 0.0
            epoch_v_loss  = 0.0

            for _ in range(update_epochs):
                shuffled_indices = torch.randperm(total_envs, device = device)
                batches = shuffled_indices.split(batch_size)
                
                for i, b_idx in enumerate(batches):
                    micro_exp = extract_micro_experience(exp, b_idx)

                    policy_loss, value_loss = agent.dynamics.learn_from_experience(
                        experience = micro_exp,
                        only_learn_policy_value_heads = False,
                        use_pmpo = False
                    )
                    
                    total_loss = (policy_loss + value_loss) / grad_accum_every
                    accelerator.backward(total_loss)
                    
                    is_last_batch = (i + 1) == len(batches)
                    
                    if divisible_by(i + 1, grad_accum_every) or is_last_batch:
                        accelerator.clip_grad_norm_(agent.parameters(), max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    epoch_pi_loss += policy_loss.item() / len(batches)
                    epoch_v_loss  += value_loss.item() / len(batches)

            avg_pi_loss = epoch_pi_loss / update_epochs
            avg_v_loss  = epoch_v_loss / update_epochs

            loss_metrics.update(
                pi_loss = f'{avg_pi_loss:.3f}', 
                v_loss = f'{avg_v_loss:.3f}'
            )

            # log video from the very first memory trace appended
            actual_len = episode_lens[0].item()
            # we need to extract video from the very first episode manually since we no longer have global exp.video
            first_ep_micro = construct_padded_experience([memories[0]], [episode_lens[0]], [is_episode_truncated[0]], device)
            wandb_video = ((first_ep_micro.video[0, :, :actual_len] + 1.0) * 127.5).clamp(0, 255).byte()
            video_numpy = rearrange(wandb_video, 'c t h w -> t c h w').cpu().numpy()

            wandb.log(dict(
                avg_return = avg_ret,
                policy_loss = avg_pi_loss,
                value_loss = avg_v_loss,
                rollout_video = wandb.Video(video_numpy, fps = 15, format = "mp4")
            ))

            # rolling deques automatically manage bounded capacities now
            
            num_policy_updates += 1

            if num_policy_updates >= max_policy_updates:
                log(f'\nReached {num_policy_updates} PPO updates! Stopping training.')
                break

            agent.eval()

        pbar_episodes.set_postfix(loss_metrics)

    wandb.finish()

if __name__ == '__main__':
    fire.Fire(main)
