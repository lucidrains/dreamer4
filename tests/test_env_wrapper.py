import shutil
import pytest
import numpy as np
from pathlib import Path

from memmap_replay_buffer import ReplayBuffer
from dreamer4.env import RecordToFolderEnvWrapper, RecordToReplayBufferEnvWrapper
from dreamer4.dreamer4 import DynamicsWorldModel, VideoTokenizer

# helpers

def exists(v):
    return v is not None

# mocks

class MockEnv:
    def __init__(
        self,
        num_steps = 5,
        img_shape = (3, 64, 64)
    ):
        self.num_steps = num_steps
        self.img_shape = img_shape
        self.steps = 0
        
    def reset(self, **kwargs):
        self.steps = 0
        return dict(image = np.random.rand(*self.img_shape).astype(np.float32))
        
    def step(self, action):
        self.steps += 1
        img = np.random.rand(*self.img_shape).astype(np.float32)
        
        done = self.steps >= self.num_steps
        return dict(image = img), 1.0, done, False

class VariedMockEnv(MockEnv):
    def __init__(self, mode):
        super().__init__(num_steps = 3)
        self.mode = mode
        
    def step(self, action):
        obs, reward, done, _ = super().step(action)
        
        if self.mode == 2:
            return obs, reward
            
        if self.mode == 3:
            return obs, reward, done
            
        if self.mode == 4:
            return obs, reward, done, {}
            
        return obs, reward, False, done, {}

# fixtures

@pytest.fixture
def temp_record_dir(tmp_path):
    d = tmp_path / "recordings"
    
    yield d
    
    if d.exists():
        shutil.rmtree(d)

@pytest.fixture
def real_replay_buffer(tmp_path):
    d = tmp_path / "replay_buffer"
    
    buffer = ReplayBuffer(
        folder = str(d),
        max_episodes = 2,
        max_timesteps = 10,
        fields = dict(
            video = ('uint8', (3, 64, 64)),
            discrete_actions = 'int',
            continuous_actions = 'float'
        )
    )
    
    yield buffer
    
    if d.exists():
        shutil.rmtree(d)

# tests

def test_record_env_wrapper(temp_record_dir):
    env = MockEnv()
    
    wrapped = RecordToFolderEnvWrapper(
        env,
        folder = str(temp_record_dir)
    )
    
    wrapped.reset()
    
    for _ in range(5):
        wrapped.step(np.array(0))
        
    assert (temp_record_dir / "episode_0.mp4").exists()

def test_interact_with_env(temp_record_dir):
    env = MockEnv()
    
    wrapped = RecordToFolderEnvWrapper(
        env,
        folder = str(temp_record_dir)
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
        num_continuous_actions = 0,
        num_discrete_actions = 4,
        max_steps = 8
    )
    
    exp = world_model.interact_with_env(
        wrapped,
        num_steps = 4,
        max_timesteps = 4
    )
    
    assert exists(exp)
    
    wrapped.reset() # trigger save
    
    assert (temp_record_dir / "episode_0.mp4").exists()

def test_step_variations(temp_record_dir):
    for mode in [2, 3, 4, 5]:
        env = VariedMockEnv(mode)
        
        wrapped = RecordToFolderEnvWrapper(
            env,
            folder = str(temp_record_dir),
            clear_on_start = False
        )
        
        wrapped.reset()
        
        for _ in range(3):
            wrapped.step(np.array([1]))
            
        if mode == 2:
            wrapped.reset()
            
        vid_path = temp_record_dir / "episode_0.mp4"
        
        assert vid_path.exists()
        vid_path.unlink()

def test_stacked_wrappers(temp_record_dir, real_replay_buffer):
    
    # order 1: env -> replay -> folder
    
    buffer1 = real_replay_buffer
    
    env1 = RecordToFolderEnvWrapper(
        RecordToReplayBufferEnvWrapper(MockEnv(), replay_buffer = buffer1),
        folder = str(temp_record_dir / "order1")
    )
    
    env1.reset()
    for _ in range(3):
        env1.step(np.array(0))
    env1.reset()
    
    actions_buf1 = np.asarray(buffer1.dataset(slice_by_episode_len = True)[0]['discrete_actions'])
    actions_fol1 = np.load(temp_record_dir / "order1" / "episode_0.discrete_actions.npy")
    
    assert np.allclose(actions_buf1[:-1], actions_fol1)

    # order 2: env -> folder -> replay
    
    buffer2 = buffer1
    buffer2.clear()
    
    env2 = RecordToReplayBufferEnvWrapper(
        RecordToFolderEnvWrapper(MockEnv(), folder = str(temp_record_dir / "order2")),
        replay_buffer = buffer2
    )
    
    env2.reset()
    for _ in range(3):
        env2.step(np.array(1))
    env2.reset()
    
    actions_buf2 = np.asarray(buffer2.dataset(slice_by_episode_len = True)[0]['discrete_actions'])
    actions_fol2 = np.load(temp_record_dir / "order2" / "episode_0.discrete_actions.npy")
    
    assert np.allclose(actions_buf2[:-1], actions_fol2)
