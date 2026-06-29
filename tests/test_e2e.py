import os
import shutil
import pytest
import numpy as np
from memmap_replay_buffer import ReplayBuffer

import torch

from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel
from dreamer4.trainers import (
    VideoTokenizerTrainer,
    BehaviorCloneTrainer,
    DreamTrainer,
    SimTrainer,
    VideoDatasetFromReplayBuffer
)
from dreamer4.env import RecordToReplayBufferEnvWrapper

class ToyEnv:
    def __init__(self, num_steps = 5):
        self.num_steps = num_steps
        self.steps = 0
        self.img_shape = (3, 64, 64)

    def reset(self, **kwargs):
        self.steps = 0
        return dict(image = np.random.rand(*self.img_shape).astype(np.float32))

    def step(self, action):
        self.steps += 1
        obs = dict(image = np.random.rand(*self.img_shape).astype(np.float32))
        reward = 1.0
        done = self.steps >= self.num_steps
        return obs, reward, done, False, {}

@pytest.mark.parametrize('is_continuous', [False, True])
def test_e2e(is_continuous):
    test_dir = f'./test_e2e_buf_{is_continuous}'
    shutil.rmtree(test_dir, ignore_errors = True)

    action_field = dict(continuous_actions=('float', (4,))) if is_continuous else dict(discrete_actions='int')

    fields = dict(
        video=('uint8', (3, 64, 64)),
        rewards='float',
        terminated='bool'
    )
    fields.update(action_field)

    buffer = ReplayBuffer(
        folder = test_dir,
        max_episodes = 10,
        max_timesteps = 10,
        fields = fields
    )

    env = ToyEnv(num_steps = 4)
    wrapped_env = RecordToReplayBufferEnvWrapper(
        env,
        replay_buffer = buffer,
        rewards = True,
        terminated = True
    )

    tokenizer = VideoTokenizer(
        dim = 16,
        dim_latent = 16,
        patch_size = 8,
        image_height = 64,
        image_width = 64,
        encoder_depth = 1,
        decoder_depth = 1,
        decoder_flow_steps = 1
    )

    world_model = DynamicsWorldModel(
        video_tokenizer = tokenizer,
        dim_latent = 16,
        dim = 16,
        depth = 1,
        attn_dim_head = 16,
        attn_heads = 2,
        num_continuous_actions=4 if is_continuous else 0,
        num_discrete_actions=0 if is_continuous else 4,
        max_steps = 4
    )

    # 1. gather initial data from environment

    # the wrapper automatically records all observations and actions to the replay buffer

    world_model.interact_with_env(wrapped_env, num_steps = 4, max_timesteps = 4)

    dataset = VideoDatasetFromReplayBuffer(buffer, image_size = 64, max_num_frames = 4)

    # 2. train tokenizer

    tokenizer_trainer = VideoTokenizerTrainer(
        model = tokenizer,
        dataset = dataset,
        batch_size = 1,
        num_train_steps = 1,
        log_video = False,
        cpu = True,
        log_dir = './test_logs'
    )
    tokenizer_trainer()

    # 3. behavior cloning on world model

    bc_trainer = BehaviorCloneTrainer(
        model = world_model,
        dataset = dataset,
        batch_size = 1,
        num_train_steps = 1,
        log_video = False,
        cpu = True,
        log_dir = './test_logs'
    )

    bc_trainer()

    # 4. rl rollouts inside the world model

    dream_trainer = DreamTrainer(
        model = world_model,
        batch_size = 1,
        generate_timesteps = 4,
        num_train_steps = 1,
        cpu = True,
        log_dir = './test_logs'
    )

    dream_trainer()

    # 5. clear buffer and enact continual learning mechanism (apply_fire)

    buffer.clear()
    world_model.apply_fire_()

    # 6. train world model on real environment interactions

    sim_trainer = SimTrainer(
        model = world_model,
        batch_size = 1,
        generate_timesteps = 4,
        epochs = 1,
        cpu = True,
        log_dir = './test_logs'
    )

    # the sim trainer handles the interaction loop transparently

    sim_trainer(wrapped_env, num_episodes = 1, max_experiences_before_learn = 1)

    # 7. tokenizer training on replay buffer again with fresh data

    fresh_dataset = VideoDatasetFromReplayBuffer(buffer, image_size = 64, max_num_frames = 4)

    tokenizer_trainer2 = VideoTokenizerTrainer(
        model = tokenizer,
        dataset = fresh_dataset,
        batch_size = 1,
        num_train_steps = 1,
        log_video = False,
        cpu = True,
        log_dir = './test_logs'
    )

    tokenizer_trainer2()

    # Cleanup

    shutil.rmtree(test_dir, ignore_errors = True)
    shutil.rmtree('./test_logs', ignore_errors = True)
