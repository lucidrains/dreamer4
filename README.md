<img src="./dreamer4-fig2.png" width="400px"></img>

## Dreamer 4

Implementation of Danijar's [latest iteration](https://arxiv.org/abs/2509.24527v1) for his [Dreamer](https://danijar.com/project/dreamer4/) line of work

[Discord channel](https://discord.gg/PmGR7KRwxq) for collaborating with other researchers interested in this work

## Appreciation

- [@dirkmcpherson](https://github.com/dirkmcpherson) for fixes to typo errors and unpassed arguments!

- [@witherhoard99](https://github.com/witherhoard99) and [Vish](https://github.com/humboldt123) for [contributing](https://github.com/lucidrains/dreamer4/pull/10) improvements to video tokenizer convergence, proprioception handling, identifying a bug with no discrete actions, and tensorboard logging with video reconstruction!

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
    num_discrete_actions = 4
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

## Moving MNIST

To train a simple tokenizer on Moving MNIST for 5000 steps and then use it to generate action-conditioned dynamics models (should not take more than an hour):

```bash
$ uv run train_moving_mnist_tokenizer.py --num_train_steps 5000

$ uv run train_moving_mnist_dynamics.py --num_train_steps 5000 --condition_on_actions True
```

The baseline will synthesize unconditionally digits floating in a random direction (with 2 frame prompt to see if it has learnt to continue detected velocity).

Passing `--condition_on_actions True` lets you explicitly prompt with velocity actions to command the digit's trajectory. The conditioned samples display a digit with action velocities arranged in the position of the grid, with center being zerod velocities (staying still).

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

```bibtex
@misc{fang2026racrectifiedflowauto,
    title   = {RAC: Rectified Flow Auto Coder},
    author  = {Sen Fang and Yalin Feng and Yanxin Zhang and Dimitris N. Metaxas},
    year    = {2026},
    eprint  = {2603.05925},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV},
    url     = {https://arxiv.org/abs/2603.05925},
}
```

```bibtex
@misc{chefer2026self,
    title   = {Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis},
    author  = {Hila Chefer and Patrick Esser and Dominik Lorenz and Dustin Podell and Vikash Raja and Vinh Tong and Antonio Torralba and Robin Rombach},
    year    = {2026},
    url     = {https://bfl.ai/research/self-flow},
    note    = {Preprint}
}
```

```bibtex
@misc{li2025basicsletdenoisinggenerative,
    title   = {Back to Basics: Let Denoising Generative Models Denoise},
    author  = {Tianhong Li and Kaiming He},
    year    = {2025},
    eprint  = {2511.13720},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV},
    url     = {https://arxiv.org/abs/2511.13720},
}
```

```bibtex
@misc{kimiteam2026attentionresiduals,
    title   = {Attention Residuals},
    author  = {Kimi Team and Guangyu Chen and Yu Zhang and Jianlin Su and Weixin Xu and Siyuan Pan and Yaoyu Wang and Yucheng Wang and Guanduo Chen and Bohong Yin and Yutian Chen and Junjie Yan and Ming Wei and Y. Zhang and Fanqing Meng and Chao Hong and Xiaotong Xie and Shaowei Liu and Enzhe Lu and Yunpeng Tai and Yanru Chen and Xin Men and Haiqing Guo and Y. Charles and Haoyu Lu and Lin Sui and Jinguo Zhu and Zaida Zhou and Weiran He and Weixiao Huang and Xinran Xu and Yuzhi Wang and Guokun Lai and Yulun Du and Yuxin Wu and Zhilin Yang and Xinyu Zhou},
    year    = {2026},
    eprint  = {2603.15031},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url     = {https://arxiv.org/abs/2603.15031},
}
```

```bibtex
@misc{zhang2026beliefformer,
    title   = {BeliefFormer: Belief Attention in Transformer},
    author  = {Guoqiang Zhang},
    year    = {2026},
    url     = {https://openreview.net/forum?id=Ard2QzPAUK}
}
```

```bibtex
@misc{osband2026delightfulpolicygradient,
    title   = {Delightful Policy Gradient},
    author  = {Ian Osband},
    year    = {2026},
    eprint  = {2603.14608},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2603.14608},
}
```

```bibtex
@misc{gopalakrishnan2025decouplingwhatwherepolar,
    title   = {Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings},
    author  = {Anand Gopalakrishnan and Robert Csordás and Jürgen Schmidhuber and Michael C. Mozer},
    year    = {2025},
    eprint  = {2509.10534},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2509.10534},
}
```

```bibtex
@misc{maes2026leworldmodelstableendtoendjointembedding,
    title   = {LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
    author  = {Lucas Maes and Quentin Le Lidec and Damien Scieur and Yann LeCun and Randall Balestriero},
    year    = {2026},
    eprint  = {2603.19312},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2603.19312},
}
```

```bibtex
@misc{balestriero2025lejepa,
    title   = {LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics},
    author  = {Randall Balestriero and Yann LeCun},
    year    = {2025},
    eprint  = {2511.08544},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2511.08544},
}
```

*the conquest of nature is to be achieved through number and measure - angels to Descartes in a dream*
