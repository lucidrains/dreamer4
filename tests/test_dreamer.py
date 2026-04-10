import pytest
param = pytest.mark.parametrize
import torch

def exists(v):
    return v is not None

@param('pred_orig_latent', (False, True))
@param('grouped_query_attn', (False, True))
@param('dynamics_with_video_input', (False, True))
@param('prob_shortcut_train', (None, 1., 0.))
@param('add_task_embeds', (False, True))
@param('num_spatial_tokens', (2, 8))
@param('signal_and_step_passed_in', (False, True))
@param('condition_on_actions', (False, True))
@param('add_reward_embed_to_agent_token', (False, True))
@param('add_state_pred_head', (False, True))
@param('use_time_cache', (False, True))
@param('var_len', (False, True))
@param('time_attention_use_pope', (False, True))
@param('with_latent_ar_in_dynamics', (False, True))
@param('with_latent_ar_action_conditioned', (False, True))
@param('predict_terminals', (False, True))
def test_e2e(
    pred_orig_latent,
    grouped_query_attn,
    dynamics_with_video_input,
    prob_shortcut_train,
    add_task_embeds,
    num_spatial_tokens,
    signal_and_step_passed_in,
    condition_on_actions,
    add_reward_embed_to_agent_token,
    add_state_pred_head,
    use_time_cache,
    var_len,
    time_attention_use_pope,
    with_latent_ar_in_dynamics,
    with_latent_ar_action_conditioned,
    predict_terminals
):
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel

    tokenizer = VideoTokenizer(
        16,
        encoder_depth = 1,
        decoder_depth = 1,
        time_block_every = 1,
        dim_latent = 16,
        patch_size = 16,
        attn_dim_head = 8,
        num_latent_tokens = 4,
        encoder_add_decorr_aux_loss = True,
        decorr_sample_frac = 1.,
        time_attention_use_pope = time_attention_use_pope
    )

    video = torch.randn(2, 3, 4, 32, 32)

    loss = tokenizer(video)
    assert loss.numel() == 1

    latents = tokenizer(video, return_latents = True)
    assert latents.shape[-1] == 16

    recon = tokenizer.decode(latents, 32, 32)
    assert recon.shape == video.shape

    query_heads, heads = (16, 4) if grouped_query_attn else (8, 8)

    dynamics = DynamicsWorldModel(
        dim = 16,
        video_tokenizer = tokenizer,
        dim_latent = 16,
        max_steps = 64,
        num_tasks = 2,
        num_latent_tokens = 4,
        depth = 1,
        num_spatial_tokens = num_spatial_tokens,
        pred_orig_latent = pred_orig_latent,
        num_discrete_actions = 4,
        attn_dim_head = 8,
        attn_heads = heads,
        attn_kwargs = dict(
            query_heads = query_heads,
        ),
        prob_shortcut_train = prob_shortcut_train,
        add_reward_embed_to_agent_token = add_reward_embed_to_agent_token,
        add_state_pred_head = add_state_pred_head,
        agent_predicts_state = add_state_pred_head,
        time_block_every = 1,
        time_attention_use_pope = time_attention_use_pope,
        latent_ar = with_latent_ar_in_dynamics,
        latent_ar_action_conditioned = with_latent_ar_action_conditioned,
        latent_ar_layer = 0 if with_latent_ar_in_dynamics else None,
        latent_ar_loss_weight = 1. if with_latent_ar_in_dynamics else 0.,
        latent_ar_sigreg_loss_weight = 0.05 if with_latent_ar_in_dynamics else 0.,
        latent_ar_sigreg_loss_kwargs = dict(num_slices = 2) if with_latent_ar_in_dynamics else None,
        predict_terminals = predict_terminals
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
        tasks = torch.randint(0, 2, (2,))

    actions = None
    if condition_on_actions:
        actions = torch.randint(0, 4, (2, 3, 1))

    lens = None
    if var_len:
        lens = torch.randint(1, 4, (2,))

    terminals = None
    if predict_terminals:
        terminals = (torch.randn(2) > 0.).bool()

    flow_loss = dynamics(
        **dynamics_input,
        lens = lens,
        tasks = tasks,
        terminals = terminals,
        signal_levels = signal_levels,
        step_sizes_log2 = step_sizes_log2,
        discrete_actions = actions,
        add_autoregressive_action_loss = True
    )

    assert flow_loss.numel() == 1

    # generating

    generations = dynamics.generate(
        time_steps = 10,
        image_height = 32,
        image_width = 32,
        batch_size = 2,
        return_rewards_per_frame = True,
        return_terminals = predict_terminals,
        use_time_cache = use_time_cache
    )

    assert generations.video.shape == (2, 3, 10, 32, 32)
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
        dim_latent = 8,
        patch_size = 16,
        encoder_depth = 1,
        decoder_depth = 1,
        time_block_every = 1,
        attn_heads = 2,
        image_height = 32,
        image_width = 32,
        attn_kwargs = dict(
            query_heads = 16
        )
    )

    dynamics = DynamicsWorldModel(
        512,
        num_agents = 1,
        video_tokenizer = tokenizer,
        dim_latent = 8,
        depth = 1,
        time_block_every = 1,
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

    assert gen.video.shape == (4, 3, 16, 32, 32)
    assert gen.rewards.shape == (4, 16)

    discrete_actions, continuous_actions = gen.actions

    assert discrete_actions.shape == (4, 16, 1)
    assert continuous_actions is None or continuous_actions.numel() == 0

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

    # test kl div

    discrete_logits_tgt, continuous_mean_log_var_tgt = embedder.unembed(action_embed)

    discrete_kl_div, continuous_kl_div = embedder.kl_div((discrete_logits, continuous_mean_log_var), (discrete_logits_tgt, continuous_mean_log_var_tgt))

    assert discrete_kl_div.shape == (2, 3)
    assert continuous_kl_div.shape == (2, 3)

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

    # 0 discrete actions

    embedder = ActionEmbedder(
        512,
        num_discrete_actions = 0,
        num_continuous_actions = 2,
        can_unembed = True
    )

    continuous_actions = torch.randn(2, 3, 2)
    action_embed = embedder(continuous_actions = continuous_actions)

    assert action_embed.shape == (2, 3, 512)

    discrete_logits, continuous_mean_log_var = embedder.unembed(action_embed)

    assert discrete_logits is None
    assert continuous_mean_log_var.shape == (2, 3, 2, 2)

    sampled_discrete, sampled_continuous = embedder.sample(action_embed)

    assert sampled_discrete is None
    assert sampled_continuous.shape == (2, 3, 2)

def test_mtp():
    from dreamer4.dreamer4 import create_multi_token_prediction_targets

    rewards = torch.randn(3, 16) # (b t)

    reward_targets, mask = create_multi_token_prediction_targets(rewards, 3) # say three token lookahead

    assert reward_targets.shape == (3, 16, 3)
    assert mask.shape == (3, 16, 3)

    actions = torch.randint(0, 10, (3, 16, 2))
    action_targets, mask = create_multi_token_prediction_targets(actions, 3)

    assert action_targets.shape == (3, 16, 3, 2)
    assert mask.shape == (3, 16, 3)

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

def test_tokenizer_trainer():
    from dreamer4.trainers import VideoTokenizerTrainer
    from dreamer4.dreamer4 import VideoTokenizer
    from torch.utils.data import Dataset

    class MockDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return dict(
                video = torch.randn(3, 2, 64, 64),
                time_lens = 2
            )

    dataset = MockDataset()

    tokenizer = VideoTokenizer(
        16,
        encoder_depth = 1,
        decoder_depth = 1,
        time_block_every = 1,
        dim_latent = 16,
        patch_size = 32,
        attn_dim_head = 16,
        num_latent_tokens = 4
    )

    trainer = VideoTokenizerTrainer(
        tokenizer,
        dataset = dataset,
        num_train_steps = 1,
        batch_size = 1,
        cpu = True,
        max_grad_norm = 0.5
    )

    trainer()

@param('with_actions', (True, False))
@param('with_rewards', (True, False))
@param('self_flow', (True, False))
def test_bc_trainer(
    with_actions,
    with_rewards,
    self_flow
):
    from dreamer4.trainers import BehaviorCloneTrainer
    from dreamer4.dreamer4 import DynamicsWorldModel, VideoTokenizer

    from torch.utils.data import Dataset

    class MockDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            state = torch.randn(3, 2, 64, 64)

            pkg = dict(video = state)

            if with_actions:
                pkg.update(discrete_actions = torch.randint(0, 4, (2, 1)))

            if with_rewards:
                pkg.update(rewards = torch.randn(2,))

            return pkg

    dataset = MockDataset()

    tokenizer = VideoTokenizer(
        16,
        encoder_depth = 1,
        decoder_depth = 1,
        time_block_every = 1,
        dim_latent = 16,
        patch_size = 32,
        attn_dim_head = 16,
        num_latent_tokens = 1
    )

    world_model = DynamicsWorldModel(
        video_tokenizer = tokenizer,
        dim = 16,
        dim_latent = 16,
        max_steps = 64,
        num_tasks = 4,
        num_latent_tokens = 1,
        depth = 4,
        time_block_every = 1,
        num_spatial_tokens = 1,
        pred_orig_latent = True,
        num_discrete_actions = 4,
        attn_dim_head = 16,
        prob_shortcut_train = 0.9,
        num_residual_streams = 1
    )

    trainer = BehaviorCloneTrainer(
        world_model,
        dataset = dataset,
        batch_size = 1,
        num_train_steps = 1,
        log_video = False,
        cpu = True,
        self_flow = self_flow,
        self_flow_student_layer = -3,
        self_flow_teacher_layer = -1,
        self_flow_loss_weight = 1.0
    )

    trainer()

def test_dream_trainer():
    from dreamer4.dreamer4 import DynamicsWorldModel

    world_model = DynamicsWorldModel(
        dim = 16,
        dim_latent = 16,
        max_steps = 64,
        num_tasks = 4,
        num_latent_tokens = 1,
        depth = 4,
        time_block_every = 1,
        num_spatial_tokens = 1,
        pred_orig_latent = True,
        num_discrete_actions = 4,
        attn_dim_head = 16,
        prob_shortcut_train = 0.9
    )

    # training from self-generations (dreams)

    from dreamer4.trainers import DreamTrainer

    dream_trainer = DreamTrainer(
        world_model,
        batch_size = 2,
        num_train_steps = 1,
        cpu = True,
    )

    dream_trainer()

def test_cache_generate():
    from dreamer4.dreamer4 import DynamicsWorldModel

    dynamics = DynamicsWorldModel(
        dim = 16,
        dim_latent = 16,
        max_steps = 64,
        num_tasks = 4,
        num_latent_tokens = 4,
        depth = 1,
        time_block_every = 1,
        num_spatial_tokens = 1,
        pred_orig_latent = True,
        num_discrete_actions = 4,
        attn_dim_head = 16,
        prob_shortcut_train = 0.9
    )

    generated, time_cache = dynamics.generate(1, return_time_cache = True)
    generated, time_cache = dynamics.generate(1, time_cache = time_cache, return_time_cache = True)
    generated, time_cache = dynamics.generate(1, time_cache = time_cache, return_time_cache = True)

@param('vectorized', (False, True))
@param('use_pmpo', (False, True))
@param('env_can_terminate', (False, True))
@param('env_can_truncate', (False, True))
@param('store_agent_embed', (False, True))
@param('predict_terminals', (False, True))
def test_online_rl(
    vectorized,
    use_pmpo,
    env_can_terminate,
    env_can_truncate,
    store_agent_embed,
    predict_terminals
):
    from dreamer4.dreamer4 import DynamicsWorldModel, VideoTokenizer

    tokenizer = VideoTokenizer(
        16,
        encoder_depth = 1,
        decoder_depth = 1,
        time_block_every = 1,
        dim_latent = 16,
        patch_size = 32,
        attn_dim_head = 16,
        num_latent_tokens = 1,
        image_height = 32,
        image_width = 32,
    )

    world_model_and_policy = DynamicsWorldModel(
        video_tokenizer = tokenizer,
        dim = 16,
        dim_latent = 16,
        max_steps = 64,
        num_tasks = 4,
        num_latent_tokens = 1,
        depth = 4,
        time_block_every = 1,
        num_spatial_tokens = 1,
        pred_orig_latent = True,
        num_discrete_actions = 4,
        attn_dim_head = 16,
        prob_shortcut_train = 0.9
    )

    from dreamer4.mocks import MockEnv
    from dreamer4.dreamer4 import combine_experiences

    mock_env = MockEnv(
        (32, 32),
        vectorized = vectorized,
        num_envs = 4,
        terminate_after_step = 2 if env_can_terminate else None,
        can_truncate = env_can_truncate,
        rand_terminate_prob = 0.1
    )

    # manually

    dream_experience = world_model_and_policy.generate(10, batch_size = 1, store_agent_embed = store_agent_embed, return_terminals = predict_terminals, return_for_policy_optimization = True)

    one_experience = world_model_and_policy.interact_with_env(mock_env, max_timesteps = 8, env_is_vectorized = vectorized, store_agent_embed = store_agent_embed)
    another_experience = world_model_and_policy.interact_with_env(mock_env, max_timesteps = 16, env_is_vectorized = vectorized, store_agent_embed = store_agent_embed)

    combined_experience = combine_experiences([dream_experience, one_experience, another_experience])

    # quick test moving the experience to different devices

    if torch.cuda.is_available():
        combined_experience = combined_experience.to(torch.device('cuda'))
        combined_experience = combined_experience.to(world_model_and_policy.device)

    if store_agent_embed:
        assert exists(combined_experience.agent_embed)

    actor_loss, critic_loss = world_model_and_policy.learn_from_experience(combined_experience, use_pmpo = use_pmpo)

    actor_loss.backward()
    critic_loss.backward()

    # with trainer

    from dreamer4.trainers import SimTrainer

    trainer = SimTrainer(
        world_model_and_policy,
        batch_size = 4,
        cpu = True
    )

    trainer(mock_env, num_episodes = 2, env_is_vectorized = vectorized)

@param('num_video_views', (1, 2))
def test_proprioception(
    num_video_views
):
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel

    tokenizer = VideoTokenizer(
        512,
        dim_latent = 32,
        patch_size = 32,
        encoder_depth = 2,
        decoder_depth = 2,
        time_block_every = 2,
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
        dim_latent = 8,
        dim_proprio = 21,
        num_tasks = 2,
        num_video_views = num_video_views,
        num_discrete_actions = 4
    )

    if num_video_views > 1:
        video_shape = (2, num_video_views, 3, 4, 32, 32)
    else:
        video_shape = (2, 3, 4, 32, 32)

    video = torch.randn(*video_shape)
    rewards = torch.randn(2, 10)
    proprio = torch.randn(2, 10, 21)
    discrete_actions = torch.randint(0, 4, (2, 10, 1))
    tasks = torch.randint(0, 4, (2,))

    loss = dynamics(
        video = video,
        rewards = rewards,
        tasks = tasks,
        proprio = proprio,
        discrete_actions = discrete_actions
    )

    loss.backward()

    generations = dynamics.generate(
        10,
        batch_size = 2,
        return_decoded_video = True
    )

    assert exists(generations.proprio)
    assert generations.video.shape == video_shape

def test_epo():
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel

    tokenizer = VideoTokenizer(
        512,
        dim_latent = 32,
        patch_size = 32,
        encoder_depth = 2,
        decoder_depth = 2,
        time_block_every = 2,
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
        dim_proprio = 21,
        num_tasks = 4,
        num_latent_genes = 16,
        num_discrete_actions = 4
    )

    fitness = torch.randn(16,)
    dynamics.evolve_(fitness)

def test_cache_token_count():
    from dreamer4 import AxialSpaceTimeTransformer

    model = AxialSpaceTimeTransformer(
        dim = 32,
        depth = 4,
        attn_heads = 4,
        attn_dim_head = 8,
        time_block_every = 2,
        num_special_spatial_tokens = 0,
        rnn_time = False,
        value_residual = False,
        use_attn_pool = False
    )

    #first pass: no cache, two timesteps
    x=torch.randn(1,2,3,32)
    _, cache = model(x, return_intermediates=True)
    assert cache.token_count == 2

    #second pass: full context+cache
    x2 = torch.randn(1,3,3,32)
    _, cache2 = model(x2, cache = cache, return_intermediates = True)
    assert cache2.token_count == 3

def test_images_to_video_tokenizer():
    import torch
    from dreamer4 import VideoTokenizer, DynamicsWorldModel, AxialSpaceTimeTransformer

    tokenizer = VideoTokenizer(
        dim = 512,
        dim_latent = 32,
        patch_size = 32,
        image_height = 256,
        image_width = 256,
        encoder_add_decorr_aux_loss = True
    )

    images = torch.randn(2, 3, 256, 256)
    loss, (losses, recon_images) = tokenizer(images, return_intermediates = True)
    loss.backward()

    assert images.shape == recon_images.shape

@param('vectorized', (False, True))
def test_interact_with_env_dict_obs(vectorized):
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel

    tokenizer = VideoTokenizer(
        512,
        dim_latent=32,
        patch_size=32,
        encoder_depth=2,
        decoder_depth=2,
        time_block_every=2,
        attn_heads=8,
        image_height=256,
        image_width=256,
        attn_kwargs=dict(
            query_heads=16
        )
    )

    dynamics = DynamicsWorldModel(
        512,
        num_agents=1,
        video_tokenizer=tokenizer,
        dim_latent=32,
        dim_proprio=21,
        num_tasks=4,
        num_latent_genes=16,
        num_discrete_actions=4
    )

    from dreamer4.mocks import MockDictEnv

    mock_env = MockDictEnv(
        (256, 256),
        21,
        vectorized = vectorized,
        num_envs = 4 if vectorized else 1,
        terminate_after_step = 3
    )

    experience = dynamics.interact_with_env(
        mock_env,
        max_timesteps = 4,
        env_is_vectorized = vectorized
    )

    assert exists(experience)
    assert experience.proprio.shape[-1] == 21
    assert experience.proprio.shape[1] > 0

@param('prompt_type', ('video', 'image', 'latents'))
@param('with_proprio', (False, True))
@param('with_discrete_actions', (False, True))
@param('with_continuous_actions', (False, True))
@param('with_rewards', (False, True))
def test_prompting_generation(
    prompt_type,
    with_proprio,
    with_discrete_actions,
    with_continuous_actions,
    with_rewards
):
    from einops import rearrange
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel

    tokenizer = VideoTokenizer(
        512,
        dim_latent = 32,
        patch_size = 32,
        encoder_depth = 1,
        decoder_depth = 1,
        time_block_every = 1,
        attn_heads = 2,
        image_height = 256,
        image_width = 256,
    )

    dynamics = DynamicsWorldModel(
        512,
        num_agents = 1,
        video_tokenizer = tokenizer,
        dim_latent = 32,
        num_tasks = 1,
        num_discrete_actions = (4, 4) if with_discrete_actions else 0,
        num_continuous_actions = 2 if with_continuous_actions else 0,
        dim_proprio = 21 if with_proprio else None,
        depth = 1,
        time_block_every = 1
    )

    # 1. generate the required prompt components

    prompt_kwargs = dict()

    if prompt_type == 'video':
        prompt_kwargs['prompt'] = torch.randn(2, 3, 1, 256, 256)
    elif prompt_type == 'image':
        prompt_kwargs['prompt'] = torch.randn(2, 3, 256, 256)
    elif prompt_type == 'latents':
        raw_video = torch.randn(2, 3, 1, 256, 256)
        latents = tokenizer.tokenize(raw_video)
        prompt_kwargs['prompt_latents'] = rearrange(latents, 'b t n d -> b t 1 n d')

    if with_proprio:
        prompt_kwargs['prompt_proprio'] = torch.randn(2, 1, 21)

    if with_discrete_actions:
        prompt_kwargs['prompt_discrete_actions'] = torch.randint(0, 4, (2, 1, 2))

    if with_continuous_actions:
        prompt_kwargs['prompt_continuous_actions'] = torch.randn(2, 1, 2)

    if with_rewards:
        prompt_kwargs['prompt_rewards'] = torch.randn(2, 1)

    # 2. execute generation

    generations = dynamics.generate(
        time_steps = 8,
        batch_size = 2,
        image_height = 256,
        image_width = 256,
        return_rewards_per_frame = with_rewards,
        return_agent_actions = (with_discrete_actions or with_continuous_actions),
        return_decoded_video = True,
        **prompt_kwargs
    )

    has_extras = with_rewards or with_proprio or with_discrete_actions or with_continuous_actions

    if has_extras:
        video = generations.video
    else:
        video = generations

    assert video.shape == (2, 3, 8, 256, 256)

    if with_rewards:
        assert generations.rewards.shape == (2, 8)

    if with_proprio:
        assert generations.proprio.shape == (2, 8, 21)

    if with_discrete_actions or with_continuous_actions:
        discrete_actions, continuous_actions = generations.actions

        if with_discrete_actions:
            assert discrete_actions.shape == (2, 8, 2)

        if with_continuous_actions:
            assert continuous_actions.shape == (2, 8, 2)

@param('steps', (1, 2, 4))
def test_rac_like_tokenizer(steps):
    from dreamer4.dreamer4 import VideoTokenizer

    model = VideoTokenizer(
        dim = 32,
        dim_latent = 32,
        patch_size = 4,
        image_height = 32,
        image_width = 32,
        num_latent_tokens = 16,
        encoder_depth = 2,
        decoder_depth = 2,
        time_block_every = 2,
        attn_dim_head = 16,
        attn_heads = 2,
        decoder_flow_steps = steps
    )

    batch = 2
    channels = 3
    time = 2
    height, width = 32, 32

    video = torch.randn(batch, channels, time, height, width)

    loss = model(video)
    assert loss.numel() == 1

    model.eval()
    latents = torch.randn(batch, time, 16, 32)
    with torch.no_grad():
        recon_video = model.decode(latents, height = height, width = width)

    assert recon_video.shape == (batch, channels, time, height, width)

@param('use_time_rnn', (False, True))
@param('use_causal_conv3d', (False, True))
@param('use_shifted_patch_tokenization', (False, True))
def test_e2e_sequential_parallel_cache(use_time_rnn, use_causal_conv3d, use_shifted_patch_tokenization):
    import torch
    from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel

    tokenizer = VideoTokenizer(
        dim = 16,
        dim_latent = 16,
        patch_size = 32,
        image_height = 64,
        image_width = 64,
        num_latent_tokens = 4,
        encoder_depth = 2,
        decoder_depth = 2,
        time_block_every = 2,
        attn_heads = 4,
        attn_dim_head = 16,
        use_causal_conv3d = use_causal_conv3d,
        use_shifted_patch_tokenization = use_shifted_patch_tokenization
    ).eval()

    dynamics = DynamicsWorldModel(
        dim = 32,
        dim_latent = tokenizer.dim_latent,
        num_latent_tokens = tokenizer.num_latent_tokens,
        num_discrete_actions = 2,
        video_tokenizer = tokenizer,
        depth = 1,
        time_block_every = 1,
        policy_head_mlp_depth = 2,
        value_head_mlp_depth = 2,
        attn_heads = 4,
        attn_dim_head = 16,
        use_time_rnn = use_time_rnn
    ).eval()

    video = torch.randn(1, 3, 4, 64, 64)
    discrete_actions = torch.randint(0, 2, (1, 4, 1))

    with torch.no_grad():
        latents_parallel, tokenizer_cache_parallel = tokenizer(
            video,
            return_latents = True,
            return_time_cache = True
        )

        step_sizes = torch.ones((1,))
        signal_levels = torch.full((1, 4), dynamics.max_steps - 1)

        _, (embeds_parallel, _) = dynamics(
            latents = latents_parallel,
            signal_levels = signal_levels,
            step_sizes = step_sizes,
            discrete_actions = discrete_actions,
            return_pred_only = True,
            return_intermediates = True,
            latent_is_noised = True
        )

        policy_embed_parallel = dynamics.policy_head(embeds_parallel.agent)

        tokenizer_cache_sequential = dynamics_cache_sequential = None
        policy_embed_sequential = []

        for frame_idx in range(4):
            latents_sequential, tokenizer_cache_sequential = tokenizer(
                video[:, :, frame_idx],
                return_latents = True,
                time_cache = tokenizer_cache_sequential,
                return_time_cache = True
            )
            seq_action = None if frame_idx == 0 else discrete_actions[:, frame_idx-1:frame_idx]

            _, (intermediates_sequential, dynamics_cache_sequential) = dynamics(
                latents = latents_sequential,
                signal_levels = signal_levels[:, frame_idx:frame_idx+1],
                step_sizes = step_sizes,
                discrete_actions = seq_action,
                time_cache = dynamics_cache_sequential,
                return_pred_only = True,
                return_intermediates = True,
                latent_is_noised = True
            )

            policy_embed_sequential.append(dynamics.policy_head(intermediates_sequential.agent))

        policy_embed_sequential = torch.cat(policy_embed_sequential, dim = 1)

    assert torch.allclose(policy_embed_parallel, policy_embed_sequential, atol = 1e-4)
