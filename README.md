<img src="./dreamer4-fig2.png" width="400px"></img>

## Dreamer 4

Implementation of Danijar's [latest iteration](https://arxiv.org/abs/2509.24527v1) for his [Dreamer](https://danijar.com/project/dreamer4/) line of work

[Discord channel](https://discord.gg/PmGR7KRwxq) for collaborating with other researchers interested in this work

## Appreciation

- [@dirkmcpherson](https://github.com/dirkmcpherson) for fixes to typo errors and unpassed arguments!

- [@witherhoard99](https://github.com/@witherhoard99) for finding and submitting a pull request for a small bug in the trainer

## Install

```bash
$ pip install dreamer4
```

## Usage

```python
import torch
from dreamer4 import VideoTokenizer, DynamicsWorldModel

# video tokenizer, learned through MAE + lpips

tokenizer = VideoTokenizer(
    dim = 512,
    dim_latent = 32,
    patch_size = 32,
    image_height = 256,
    image_width = 256
)

video = torch.randn(2, 3, 10, 256, 256)

# learn the tokenizer

loss = tokenizer(video)
loss.backward()

# dynamics world model

world_model = DynamicsWorldModel(
    dim = 512,
    dim_latent = 32,
    video_tokenizer = tokenizer,
    num_discrete_actions = 4,
    num_residual_streams = 1
)

# state, action, rewards

video = torch.randn(2, 3, 10, 256, 256)
discrete_actions = torch.randint(0, 4, (2, 10, 1))
rewards = torch.randn(2, 10)

# learn dynamics / behavior cloned model

loss = world_model(
    video = video,
    rewards = rewards,
    discrete_actions = discrete_actions
)

loss.backward()

# do the above with much data

# then generate dreams

dreams = world_model.generate(
    10,
    batch_size = 2,
    return_decoded_video = True,
    return_for_policy_optimization = True
)

# learn from the dreams

actor_loss, critic_loss = world_model.learn_from_experience(dreams)

(actor_loss + critic_loss).backward()

# learn from environment

from dreamer4.mocks import MockEnv

mock_env = MockEnv((256, 256), vectorized = True, num_envs = 4)

experience = world_model.interact_with_env(mock_env, max_timesteps = 8, env_is_vectorized = True)

actor_loss, critic_loss = world_model.learn_from_experience(experience)

(actor_loss + critic_loss).backward()
```

## Citation

```bibtex
@misc{hafner2025trainingagentsinsidescalable,
    title   = {Training Agents Inside of Scalable World Models}, 
    author  = {Danijar Hafner and Wilson Yan and Timothy Lillicrap},
    year    = {2025},
    eprint  = {2509.24527},
    archivePrefix = {arXiv},
    primaryClass = {cs.AI},
    url     = {https://arxiv.org/abs/2509.24527}, 
}
```

*the conquest of nature is to be achieved through number and measure - angels to Descartes in a dream*
