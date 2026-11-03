# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "gymnasium[classic_control,other]",
#     "fire",
#     "einops",
#     "numpy",
#     "torch-einops-utils",
#     "dreamer4"
# ]
# [tool.uv.sources]
# dreamer4 = { path = "." }
# ///

import shutil
from collections import deque
from pathlib import Path

import fire
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from einops import reduce
from gymnasium.wrappers import RecordVideo
from torch.optim import AdamW
from torch_einops_utils import lens_to_mask

from dreamer4.dreamer4 import DynamicsWorldModel, combine_experiences


def evaluate(model, env, episodes, candidates = 1, horizon = 1, value_coef = 1., verbose = False):
    model.eval()
    returns = []

    for episode in range(episodes):
        experience = model.interact_with_env(
            env,
            seed = 1000 + episode,
            max_timesteps = 200,
            num_steps = 4,
            env_is_vectorized = False,
            store_agent_embed = False,
            store_old_action_unembeds = False,
            num_action_candidates = candidates,
            action_filter_horizon = horizon,
            action_filter_context = 1,
            action_filter_value_coef = value_coef
        )

        episode_return = experience.episode_return.mean().item()
        returns.append(episode_return)

        if verbose:
            print(f'  episode {episode + 1}/{episodes}: {episode_return:.1f}', flush = True)

    returns = np.asarray(returns)
    return returns.mean().item(), returns.std().item()


def record_episode(model, seed, folder, prefix, candidates = 1, horizon = 1):
    folder = Path(folder)
    folder.mkdir(exist_ok = True, parents = True)

    env = gym.make('CartPole-v1', render_mode = 'rgb_array')
    env = RecordVideo(env, video_folder = str(folder), name_prefix = prefix, episode_trigger = lambda episode: True, fps = 30, disable_logger = True)

    experience = model.interact_with_env(
        env,
        seed = seed,
        max_timesteps = 200,
        num_steps = 4,
        env_is_vectorized = False,
        store_agent_embed = False,
        store_old_action_unembeds = False,
        num_action_candidates = candidates,
        action_filter_horizon = horizon,
        action_filter_context = 1
    )

    env.close()
    return experience.episode_return.mean().item()


def main(
    num_iterations = 20,
    episodes_per_iteration = 6,
    max_timesteps = 120,
    replay_size = 64,
    world_model_steps = 8,
    policy_steps = 4,
    eval_episodes = 10,
    learning_rate = 1e-3,
    record_videos = True,
    video_folder = './videos/cartpole_mpc',
    seed = 0
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make('CartPole-v1')
    env.reset(seed = seed)
    env.action_space.seed(seed)

    model = DynamicsWorldModel(
        dim = 32,
        dim_latent = 16,
        max_steps = 4,
        num_register_tokens = 1,
        num_latent_tokens = 1,
        num_spatial_tokens = 1,
        depth = 3,
        time_block_every = 2,
        attn_heads = 4,
        attn_dim_head = 8,
        num_discrete_actions = 2,
        dim_state = 4,
        predict_terminals = True,
        discrete_action_loss_weight = 0.,
        agent_predicts_state = True,
        agent_predicts_state_frac_gradient = 1.,
        value_encoder_kwargs = dict(reward_range = (-20., 220.)),
        prob_shortcut_train = 0.,
        loss_weight_fn = lambda t: (t < 1.).float()
    )

    # fixed random state projection with balanced feature scales, dynamics trained on top

    with torch.no_grad():
        state_scale = torch.tensor([1., 1. / 1.5, 1. / 0.2095, 1. / 2.], device = model.device)
        model.state_to_latents[0].weight.mul_(state_scale)

    model.state_to_latents.requires_grad_(False)

    optimizer = AdamW(model.parameters(), lr = learning_rate)
    memories = deque(maxlen = replay_size)

    for iteration in range(num_iterations):
        model.eval()

        new_experiences = [
            model.interact_with_env(
                env,
                max_timesteps = max_timesteps,
                num_steps = 4,
                env_is_vectorized = False,
                store_agent_embed = True,
                store_old_action_unembeds = True
            ).cpu()
            for _ in range(episodes_per_iteration)
        ]

        memories.extend(new_experiences)
        on_policy_batch = combine_experiences(new_experiences).to(model.device)
        batch = combine_experiences(list(memories)).to(model.device)

        model.train()

        valid_lens = batch.lens - batch.is_truncated.long()

        for step in range(world_model_steps):
            window = (step % 4) + 2
            valid_episodes = (valid_lens >= window).nonzero().squeeze(-1)

            ep_indices = valid_episodes[torch.randint(0, valid_episodes.numel(), (32,), device = model.device)]
            start_times = (torch.rand(32, device = model.device) * (valid_lens[ep_indices] - window + 1)).long()
            time_indices = start_times[:, None] + torch.arange(window, device = model.device)

            loss, losses = model(
                latents = batch.latents[ep_indices[:, None], time_indices],
                discrete_actions = batch.actions.discrete[ep_indices[:, None], time_indices[:, :-1]],
                time_modifier_fn = lambda t: F.pad(t[:, -1:], (window - 1, 0), value = 1.),
                latent_has_view_dim = True,
                return_all_losses = True
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        for _ in range(policy_steps):
            policy_loss, value_loss = model.learn_from_experience(
                on_policy_batch,
                only_learn_policy_value_heads = True,
                objective = 'ppo'
            )

            optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if iteration % 2 == 0:
            base, _ = evaluate(model, env, episodes = 5)
            print(f'iteration {iteration}: world {losses.flow.item():.4f} | policy {policy_loss.item():.3f} | value {value_loss.item():.3f} | base {base:.1f}', flush = True)

    # freeze the dynamics and fit the heads used to score imagined rollouts

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    pooled = reduce(batch.latents[:, 1:], 'b t v n d -> b t d', 'mean')
    valid_lens = batch.lens - batch.is_truncated.long()
    mask = lens_to_mask(valid_lens, batch.rewards.shape[1])[:, :-1]
    reward_targets = model.reward_encoder(batch.rewards[:, 1:]).argmax(dim = -1)
    terminal_targets = torch.zeros_like(mask, dtype = torch.float32)

    for i in range(mask.shape[0]):
        last_transition = int(valid_lens[i]) - 2

        if last_transition >= 0:
            terminal_targets[i, last_transition] = batch.terminals[i].float()

    reward_head = model.to_latent_state_reward_pred
    terminal_head = model.to_latent_state_terminal_pred
    reward_head.requires_grad_(True)
    terminal_head.requires_grad_(True)

    head_optimizer = AdamW([*reward_head.parameters(), *terminal_head.parameters()], lr = 3e-3)
    pos_weight = ((1. - terminal_targets[mask]).sum() / terminal_targets[mask].sum().clamp(min = 1.)).clamp(max = 15.)

    for _ in range(200):
        reward_logits = reward_head(pooled).movedim(-1, 1)
        reward_loss = F.cross_entropy(reward_logits, reward_targets, reduction = 'none')[mask].mean()

        terminal_logits = terminal_head(pooled).squeeze(-1)
        terminal_loss = F.binary_cross_entropy_with_logits(terminal_logits[mask], terminal_targets[mask], pos_weight = pos_weight)

        loss = reward_loss + terminal_loss
        head_optimizer.zero_grad()
        loss.backward()
        head_optimizer.step()

    print(f'heads: reward {reward_loss.item():.3f} | terminal {terminal_loss.item():.3f}', flush = True)

    print('evaluating baseline', flush = True)
    base, base_std = evaluate(model, env, episodes = eval_episodes, verbose = True)
    print(f'base={base:.1f} ± {base_std:.1f}', flush = True)

    for horizon in (2, 4):
        print(f'evaluating mpc horizon {horizon}', flush = True)
        mpc, mpc_std = evaluate(model, env, episodes = eval_episodes, candidates = 4, horizon = horizon, verbose = True)
        print(f'mpc_h{horizon}={mpc:.1f} ± {mpc_std:.1f}', flush = True)

    if record_videos:
        shutil.rmtree(video_folder, ignore_errors = True)

        baseline_return = record_episode(model, seed = 1003, folder = f'{video_folder}/baseline', prefix = 'baseline')
        mpc_return = record_episode(model, seed = 1003, folder = f'{video_folder}/mpc', prefix = 'mpc', candidates = 4, horizon = 4)

        print(f'video baseline return {baseline_return:.1f}', flush = True)
        print(f'video mpc return {mpc_return:.1f}', flush = True)


if __name__ == '__main__':
    fire.Fire(main)
