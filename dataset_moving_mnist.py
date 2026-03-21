import torch
from torch import tensor
from torch.utils.data import Dataset
import torchvision.transforms as T
from einops import repeat
from numpy.random import randint

class MovingMNISTDataset(Dataset):
    def __init__(
        self,
        root = './data',
        num_frames = 10,
        image_size = 64,
        digit_size = 28,
        min_velocity = -2,
        max_velocity = 3,
        condition_on_actions = False,
        action_type = 'both', # 'continuous', 'discrete', 'random', or 'both'
        num_action_bins = 5,
        download = True
    ):
        super().__init__()
        from torchvision.datasets import MNIST
        self.mnist = MNIST(root = root, train = True, download = download)
        self.num_frames = num_frames
        self.image_size = image_size
        self.digit_size = digit_size
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.condition_on_actions = condition_on_actions
        self.action_type = action_type
        self.num_action_bins = num_action_bins

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        digit, _ = self.mnist[idx]
        digit = T.functional.to_tensor(digit)

        if self.digit_size != 28:
            resizer = T.Resize((self.digit_size, self.digit_size), antialias = True)
            digit = resizer(digit)

        digit = digit.squeeze(0)

        seq = torch.arange(self.num_frames)

        # position and velocity

        start_x, start_y = torch.randint(0, self.image_size - self.digit_size, (2,)).tolist()

        vel_x, vel_y = torch.empty(2).uniform_(self.min_velocity, self.max_velocity).tolist()

        x_positions = (start_x + vel_x * seq).long().clamp(0, self.image_size - self.digit_size)
        y_positions = (start_y + vel_y * seq).long().clamp(0, self.image_size - self.digit_size)

        # generate video tensor

        video = torch.zeros(self.num_frames, self.image_size, self.image_size)

        for f, x, y in zip(range(self.num_frames), x_positions, y_positions):
            dest = video[f, y:(y + self.digit_size), x:(x + self.digit_size)]
            torch.maximum(dest, digit, out = dest)

        video = repeat(video, 'f h w -> c f h w', c = 3)

        res = dict(video = video)

        if not self.condition_on_actions:
            return res

        action_seq_len = self.num_frames - 1

        if self.action_type in ('continuous', 'both'):
            cont_action = torch.tensor([vel_x, vel_y], dtype = torch.float32)
            res['continuous_actions'] = repeat(cont_action, 'd -> t d', t = action_seq_len)

        if self.action_type in ('discrete', 'both'):
            bin_size = (self.max_velocity - self.min_velocity) / self.num_action_bins
            
            disc_action = torch.tensor([vel_x, vel_y])
            disc_action = disc_action.sub_(self.min_velocity).div_(bin_size).long()
            disc_action = disc_action.clamp_(max = self.num_action_bins - 1)
            
            res['discrete_actions'] = repeat(disc_action, 'd -> t d', t = action_seq_len)

        return res
