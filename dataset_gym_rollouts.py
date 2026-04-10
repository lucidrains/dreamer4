import torch
from torch import Tensor
from torch.utils.data import Dataset
from einops import repeat

class GymRolloutDataset(Dataset):
    """Collects random rollouts from a Gymnasium env for tokenizer training."""

    def __init__(
        self,
        env_id = 'CartPole-v1',
        num_rollouts = 5000,
        num_frames = 5,
        image_height = 60,
        image_width = 90,
        action_repeat = 1,
    ):
        super().__init__()
        import gymnasium as gym
        import numpy as np

        env = gym.make(env_id, render_mode = 'rgb_array')

        videos = []

        for _ in range(num_rollouts):
            env.reset()
            frames = []

            for _ in range(num_frames):
                frame = env.render()
                frame = torch.from_numpy(frame).float() / 255.
                frame = frame.permute(2, 0, 1).unsqueeze(0)
                frame = torch.nn.functional.interpolate(frame, size = (image_height, image_width), mode = 'bilinear', align_corners = False)
                frames.append(frame.squeeze(0))

                for _ in range(action_repeat):
                    action = env.action_space.sample()
                    _, _, terminated, truncated, _ = env.step(action)

                    if terminated or truncated:
                        env.reset()
                        break

            videos.append(torch.stack(frames, dim = 1))  # (c, t, h, w)

        env.close()

        self.videos = videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        return dict(video = self.videos[idx])
