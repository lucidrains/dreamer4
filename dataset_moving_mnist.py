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

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        digit, _ = self.mnist[idx]
        digit = T.functional.to_tensor(digit)
        
        if self.digit_size != 28:
            resizer = T.Resize((self.digit_size, self.digit_size), antialias=True)
            digit = resizer(digit)
            
        digit = digit.squeeze(0)

        seq = torch.arange(self.num_frames)

        # position and velocity

        start_x, start_y = (randint(0, self.image_size - self.digit_size) for _ in range(2))
        vel_x, vel_y = (randint(self.min_velocity, self.max_velocity) for _ in range(2))

        x_positions = (start_x + vel_x * seq).clamp(0, self.image_size - self.digit_size)
        y_positions = (start_y + vel_y * seq).clamp(0, self.image_size - self.digit_size)

        # generate video tensor

        video = torch.zeros(self.num_frames, self.image_size, self.image_size)

        for f, x, y in zip(range(self.num_frames), x_positions, y_positions):
            dest = video[f, y:y+self.digit_size, x:x+self.digit_size]
            torch.maximum(dest, digit, out = dest)

        return repeat(video, 'f h w -> c f h w', c = 3)
