import pytest
param = pytest.mark.parametrize
import torch

@param('pred_orig_latent', (False, True))
@param('grouped_query_attn', (False, True))
@param('dynamics_with_video_input', (False, True))
@param('prob_no_shortcut_train', (None, 0., 1.))
@param('add_task_embeds', (False, True))
@param('num_spatial_tokens', (2, 8))
@param('signal_and_step_passed_in', (False, True))
@param('condition_on_actions', (False, True))
@param('num_residual_streams', (1, 4))
@param('add_reward_embed_to_agent_token', (False, True))
@param('use_time_kv_cache', (False, True))
def test_e2e(
    pred_orig_latent,
    grouped_query_attn,
    dynamics_with_video_input,
    prob_no_shortcut_train,
    add_task_embeds,
    num_spatial_tokens,
    signal_and_step_passed_in,
    condition_on_actions,
    num_residual_streams,
    add_reward_embed_to_agent_token,
    use_time_kv_cache
):
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel

    tokenizer = VideoTokenizer(
        16,
        encoder_depth = 1,
        decoder_depth = 1,
        dim_latent = 16,
        patch_size = 32,
        attn_dim_head = 16,
        num_latent_tokens = 4,
        num_residual_streams = num_residual_streams
    )

    video = torch.randn(2, 3, 4, 256, 256)

    loss = tokenizer(video)
    assert loss.numel() == 1

    latents = tokenizer(video, return_latents = True)
    assert latents.shape[-1] == 16

    recon = tokenizer.decode(latents, 256, 256)
    assert recon.shape == video.shape

    query_heads, heads = (16, 4) if grouped_query_attn else (8, 8)

    dynamics = DynamicsWorldModel(
        dim = 16,
        video_tokenizer = tokenizer,
        dim_latent = 16,
        max_steps = 64,
        num_tasks = 4,
        num_latent_tokens = 4,
        depth = 4,
        num_spatial_tokens = num_spatial_tokens,
        pred_orig_latent = pred_orig_latent,
        num_discrete_actions = 4,
        attn_dim_head = 16,
        attn_kwargs = dict(
            heads = heads,
            query_heads = query_heads,
        ),
        prob_no_shortcut_train = prob_no_shortcut_train,
        add_reward_embed_to_agent_token = add_reward_embed_to_agent_token,
        num_residual_streams = num_residual_streams
    )

    signal_levels = step_sizes_log2 = None

    if signal_and_step_passed_in:
        signal_levels = torch.randint(0, 32, (2, 4))
        step_sizes_log2 = torch.randint(1, 5, (2,))

    if dynamics_with_video_input:
        dynamics_input = dict(video = video)
    else:
        dynamics_input = dict(latents = latents)

    tasks = None
    if add_task_embeds:
        tasks = torch.randint(0, 4, (2,))

    actions = None
    if condition_on_actions:
        actions = torch.randint(0, 4, (2, 3, 1))

    flow_loss = dynamics(
        **dynamics_input,
        tasks = tasks,
        signal_levels = signal_levels,
        step_sizes_log2 = step_sizes_log2,
        discrete_actions = actions,
        add_autoregressive_action_loss = True
    )

    assert flow_loss.numel() == 1

    # generating

    generations = dynamics.generate(
        time_steps = 10,
        image_height = 128,
        image_width = 128,
        batch_size = 2,
        return_rewards_per_frame = True,
        use_time_kv_cache = use_time_kv_cache
    )

    assert generations.video.shape == (2, 3, 10, 128, 128)
    assert generations.rewards.shape == (2, 10)

    # rl

    rewards = torch.randn((2, 4)) * 100.

    flow_loss = dynamics(
        **dynamics_input,
        tasks = tasks,
        rewards = rewards
    )

def test_symexp_two_hot():
    import torch
    from dreamer4.dreamer4 import SymExpTwoHot

    two_hot_encoder = SymExpTwoHot(
        (-3., 3.),
        num_bins = 20,
        learned_embedding = True,
        dim_embed = 512
    )

    values = torch.randn((10))

    two_hot_encoded = two_hot_encoder(values)
    recon_values = two_hot_encoder.bins_to_scalar_value(two_hot_encoded)

    assert torch.allclose(recon_values, values, atol = 1e-6)

    reward_embeds = two_hot_encoder.embed(two_hot_encoded)
    assert reward_embeds.shape == (10, 512)

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'no cuda')
@param('causal', (False, True))
@param('softclamp_value', (50., None))
@param('num_agent_tokens', (0, 1))
@param('causal_block_size', (1, 8))
@param('block_size_per_special', (1, 8))
@param('special_attend_only_itself', (False, True))
def test_attend_factory(
    causal,
    softclamp_value,
    num_agent_tokens,
    causal_block_size,
    block_size_per_special,
    special_attend_only_itself
):

    from dreamer4.dreamer4 import get_attend_fn

    q = torch.randn(1, 8, 1024, 512).cuda()
    k = torch.randn(1, 4, 1024, 512).cuda()
    v = torch.randn(1, 4, 1024, 512).cuda()

    attend_kwargs = dict(
        seq_len = 1024,
        k_seq_len = 1024,
        causal = causal,
        causal_block_size = causal_block_size,
        softclamp_value = softclamp_value,
        device = q.device,
        num_agent_tokens = num_agent_tokens,
        block_size_per_special = block_size_per_special,
        special_attend_only_itself = special_attend_only_itself
    )

    attend = get_attend_fn(True, **attend_kwargs)
    flex_out = attend(q, k, v)

    attend = get_attend_fn(False, **attend_kwargs)
    out = attend(q, k, v)

    assert torch.allclose(flex_out, out, atol = 1e-6)

def test_action_with_world_model():
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel

    tokenizer = VideoTokenizer(
        512,
        dim_latent = 32,
        patch_size = 32,
        encoder_depth = 1,
        decoder_depth = 1,
        attn_heads = 8,
        image_height = 256,
        image_width = 256,
        attn_kwargs = dict(
            query_heads = 16
        )
    )

    dynamics = DynamicsWorldModel(
        512,
        num_agents = 1,
        video_tokenizer = tokenizer,
        dim_latent = 32,
        num_discrete_actions = 4
    )

    rewards = torch.randn(1, 4)
    discrete_actions = torch.randint(0, 4, (1, 4, 1))

    gen = dynamics.generate(
        16,
        batch_size = 4,
        return_rewards_per_frame = True,
        return_agent_actions = True,
        return_log_probs_and_values = True
    )

    assert gen.video.shape == (4, 3, 16, 256, 256)
    assert gen.rewards.shape == (4, 16)

    discrete_actions, continuous_actions = gen.actions

    assert discrete_actions.shape == (4, 16, 1)
    assert continuous_actions is None

    discrete_log_probs, _ = gen.log_probs

    assert discrete_log_probs.shape == (4, 16, 1)
    assert gen.values.shape == (4, 16)

    # take a reinforcement learning step

    actor_loss, critic_loss = dynamics.learn_from_experience(gen)

    actor_loss.backward(retain_graph = True)
    critic_loss.backward()

def test_action_embedder():
    from dreamer4.dreamer4 import ActionEmbedder

    # 1 discrete action with 4 choices

    embedder = ActionEmbedder(
        512,
        num_discrete_actions = 4
    )

    actions = torch.randint(0, 4, (2, 3, 1))
    action_embed = embedder(discrete_actions = actions)

    assert action_embed.shape == (2, 3, 512)

    # 2 discrete actions with 4 choices each

    embedder = ActionEmbedder(
        512,
        num_discrete_actions = (4, 4)
    )

    actions = torch.randint(0, 4, (2, 3, 2))
    action_embed = embedder(discrete_actions = actions)

    assert action_embed.shape == (2, 3, 512)

    # picking out only the second discrete action

    actions = torch.randint(0, 4, (2, 3, 1))
    action_embed = embedder(discrete_actions = actions, discrete_action_types = 1)

    assert action_embed.shape == (2, 3, 512)

    # 2 continuous actions

    embedder = ActionEmbedder(
        512,
        num_continuous_actions = 2,
        continuous_norm_stats = ((0., 2.), (1., 1.)) # (mean, std) for normalizing each action
    )

    actions = torch.randn((2, 3, 2))
    action_embed = embedder(continuous_actions = actions)

    assert action_embed.shape == (2, 3, 512)

    # 2 discrete actions with 4 choices each and 2 continuous actions

    embedder = ActionEmbedder(
        512,
        num_discrete_actions = (4, 4),
        num_continuous_actions = 2
    )

    discrete_actions = torch.randint(0, 4, (2, 3, 2))
    continuous_actions = torch.randn(2, 3, 2)

    action_embed = embedder(discrete_actions = discrete_actions, continuous_actions = continuous_actions)
    assert action_embed.shape == (2, 3, 512)

    # picking out one discrete and one continuous

    discrete_actions = torch.randint(0, 4, (2, 3, 1))
    continuous_actions = torch.randn(2, 3, 1)

    action_embed = embedder(discrete_actions = discrete_actions, continuous_actions = continuous_actions, discrete_action_types = 1, continuous_action_types = 0)

    assert action_embed.shape == (2, 3, 512)

    # unembed

    embedder = ActionEmbedder(
        512,
        num_discrete_actions = (4, 4),
        num_continuous_actions = 2,
        can_unembed = True
    )

    discrete_actions = torch.randint(0, 4, (2, 3, 2))
    continuous_actions = torch.randn(2, 3, 2)

    action_embed = embedder(discrete_actions = discrete_actions, continuous_actions = continuous_actions)

    discrete_logits, continuous_mean_log_var = embedder.unembed(action_embed)

    assert discrete_logits.shape == (2, 3, 8)
    assert continuous_mean_log_var.shape == (2, 3, 2, 2)

    # return discrete split by number of actions

    discrete_logits, continuous_mean_log_var = embedder.unembed(action_embed, return_split_discrete = True)
    assert discrete_logits[0].shape == discrete_logits[1].shape == (2, 3, 4)

    # unembed subset of actions

    discrete_logits, continuous_mean_log_var = embedder.unembed(action_embed, discrete_action_types = 1, continuous_action_types = 0)

    assert discrete_logits.shape == (2, 3, 4)
    assert continuous_mean_log_var.shape == (2, 3, 1, 2)

    # sample actions

    sampled_discrete_actions, sampled_continuous_actions = embedder.sample(action_embed, discrete_action_types = 1, continuous_action_types = 0)

    assert sampled_discrete_actions.shape == (2, 3, 1)
    assert sampled_continuous_actions.shape == (2, 3, 1)

    # log probs

    assert discrete_logits.shape == (2, 3, 4)
    assert continuous_mean_log_var.shape == (2, 3, 1, 2)

    discrete_log_probs, continuous_log_probs = embedder.log_probs(
        action_embed,
        discrete_targets = discrete_actions,
        continuous_targets = continuous_actions,
        parallel_discrete_calc = False
    )

    assert discrete_log_probs.shape == (2, 3, 2)
    assert continuous_log_probs.shape == (2, 3, 2)

    _, (discrete_entropies, continuous_entropies) = embedder.log_probs(
        action_embed,
        discrete_targets = discrete_actions,
        continuous_targets = continuous_actions,
        parallel_discrete_calc = True,
        return_entropies = True
    )

    assert discrete_entropies.shape == (2, 3, 2)
    assert continuous_entropies.shape == (2, 3, 2)

    parallel_discrete_log_probs, _ = embedder.log_probs(
        action_embed,
        discrete_targets = discrete_actions,
        continuous_targets = continuous_actions,
        parallel_discrete_calc = True
    )

    assert torch.allclose(discrete_log_probs, parallel_discrete_log_probs, atol = 1e-5)

def test_mtp():
    from dreamer4.dreamer4 import create_multi_token_prediction_targets

    rewards = torch.randn(3, 16) # (b t)

    reward_targets, mask = create_multi_token_prediction_targets(rewards, 3) # say three token lookahead

    assert reward_targets.shape == (3, 15, 3)
    assert mask.shape == (3, 15, 3)

    actions = torch.randint(0, 10, (3, 16, 2))
    action_targets, mask = create_multi_token_prediction_targets(actions, 3)

    assert action_targets.shape == (3, 15, 3, 2)
    assert mask.shape == (3, 15, 3)

    from dreamer4.dreamer4 import ActionEmbedder

    embedder = ActionEmbedder(
        512,
        num_discrete_actions = (4, 4),
        num_continuous_actions = 2,
        can_unembed = True,
        num_unembed_preds = 8
    )

    discrete_actions = torch.randint(0, 4, (2, 3, 2))
    continuous_actions = torch.randn(2, 3, 2)

    action_embed = torch.randn(2, 16, 512)
    discrete_logits, continuous_logits = embedder.unembed(action_embed)

    assert discrete_logits.shape == (8, 2, 16, 8)

    discrete_logits, continuous_logits = embedder.unembed(action_embed, pred_head_index = 0)

    assert discrete_logits.shape == (2, 16, 8)

def test_loss_normalizer():
    from torch import cat
    from dreamer4.dreamer4 import LossNormalizer

    loss_normalizer = LossNormalizer(4, beta = 0.)

    losses = torch.rand(4)

    _ = loss_normalizer(losses)
    normed_losses = loss_normalizer(losses)

    assert (normed_losses == 1.).all()
