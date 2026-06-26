<img src="./dreamer4-fig2.png" width="400px"></img>

## Dreamer 4

Implementation of Danijar's [latest iteration](https://arxiv.org/abs/2509.24527v1) for his [Dreamer](https://danijar.com/project/dreamer4/) line of work

[Discord channel](https://discord.gg/PmGR7KRwxq) for collaborating with other researchers interested in this work

## Appreciation

- [@dirkmcpherson](https://github.com/dirkmcpherson) for fixes to typo errors and unpassed arguments!

- [@witherhoard99](https://github.com/witherhoard99) and [Vish](https://github.com/humboldt123) for [contributing](https://github.com/lucidrains/dreamer4/pull/10) improvements to video tokenizer convergence, proprioception handling, identifying a bug with no discrete actions, and tensorboard logging with video reconstruction!

- [@CarsonBurke](https://github.com/CarsonBurke) for identifying and contributing bug fixes!

- `@njha` in discord channel for finding an issue with the flow loss weight for the dynamics model!

- [@CarsonBurke](https://github.com/CarsonBurke) for his [pull request](https://github.com/lucidrains/dreamer4/pull/25) enabling configurable activations!

- [@CarsonBurke](https://github.com/CarsonBurke) for adding the [HL-Gauss reward encoder](https://github.com/lucidrains/dreamer4/pull/28)!

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

## CLI

You can easily train the video tokenizer using the command line interface:

```bash
dreamer4 train-video-tokenizer /path/to/videos \
    --name experiment_name \
    --image_size 64 \
    --batch_size 4 \
    --grad_accum_every 2 \
    --num_train_steps 100000 \
    --depth 4 \
    --flow_steps 2 \
    --separate_flow_decoder True
```

Once trained, you can load the resulting tokenizer in python with a single line using the saved checkpoint

```python
from dreamer4 import VideoTokenizer

# instantiate and load
tokenizer = VideoTokenizer.init_and_load('./checkpoints/experiment_name/tokenizer.pt')

# loading the ema tokenizer
ema_tokenizer = VideoTokenizer.init_and_load('./checkpoints/experiment_name/tokenizer-ema.pt')

```

You can then train the dynamics world model using the trained tokenizer:

```bash
dreamer4 train-dynamics /path/to/videos \
    --tokenizer_checkpoint ./checkpoints/experiment_name/tokenizer-ema.pt \
    --name experiment_name \
    --batch_size 2
```

If your dataset contains continuous or discrete actions saved as `.action.npy` files alongside your videos, you can train an action-conditioned model:

```bash
dreamer4 train-dynamics /path/to/videos \
    --tokenizer_checkpoint ./checkpoints/experiment_name/tokenizer-ema.pt \
    --name experiment_name \
    --condition_on_actions True \
    --num_continuous_actions 6 \
    --batch_size 2
```

## Moving MNIST

To train a simple tokenizer on Moving MNIST for 20000 steps and then use it to generate action-conditioned dynamics models

```bash
$ uv run train_moving_mnist_tokenizer.py --num_train_steps 20000

$ uv run train_moving_mnist_dynamics.py --num_train_steps 20000 --condition_on_actions True
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

```bibtex
@article{Lee2021VisionTF,
    title   = {Vision Transformer for Small-Size Datasets},
    author  = {Seung Hoon Lee and Seunghyun Lee and Byung Cheol Song},
    journal = {arXiv preprint arXiv:2112.13492},
    year    = {2021}
}
```

```bibtex
@misc{lavoie2022simplicialembeddingsselfsupervisedlearning,
    title   = {Simplicial Embeddings in Self-Supervised Learning and Downstream Classification},
    author  = {Samuel Lavoie and Christos Tsirigotis and Max Schwarzer and Ankit Vani and Michael Noukhovitch and Kenji Kawaguchi and Aaron Courville},
    year    = {2022},
    eprint  = {2204.00616},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2204.00616},
}
```

```bibtex
@misc{schmidt2024learningactactions,
    title   = {Learning to Act without Actions},
    author  = {Dominik Schmidt and Minqi Jiang},
    year    = {2024},
    eprint  = {2312.10812},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2312.10812},
}
```

```bibtex
@misc{whittington2022relatingtransformersmodelsneural,
    title   = {Relating transformers to models and neural representations of the hippocampal formation},
    author  = {James C. R. Whittington and Joseph Warren and Timothy E. J. Behrens},
    year    = {2022},
    eprint  = {2112.04035},
    archivePrefix = {arXiv},
    primaryClass = {cs.NE},
    url     = {https://arxiv.org/abs/2112.04035},
}
```

```bibtex
@misc{xie2025simplepolicyoptimization,
    title   = {Simple Policy Optimization},
    author  = {Zhengpeng Xie and Qiang Zhang and Fan Yang and Marco Hutter and Renjing Xu},
    year    = {2025},
    eprint  = {2401.16025},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2401.16025},
}
```

```bibtex
@misc{zhao2026subjepasubspacegaussianregularization,
    title   = {Sub-JEPA: Subspace Gaussian Regularization for Stable End-to-End World Models},
    author  = {Kai Zhao and Dongliang Nie and Yuchen Lin and Zhehan Luo and Yixiao Gu and Deng-Ping Fan and Dan Zeng},
    year    = {2026},
    eprint  = {2605.09241},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2605.09241},
}
```

```bibtex
@misc{wu2025h3aehighcompressionhigh,
    title   = {H3AE: High Compression, High Speed, and High Quality AutoEncoder for Video Diffusion Models},
    author  = {Yushu Wu and Yanyu Li and Ivan Skorokhodov and Anil Kag and Willi Menapace and Sharath Girish and Aliaksandr Siarohin and Yanzhi Wang and Sergey Tulyakov},
    year    = {2025},
    eprint  = {2504.10567},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV},
    url     = {https://arxiv.org/abs/2504.10567},
}
```

```bibtex
@misc{han2026firefrobeniusisometryreinitializationbalancing,
    title   = {FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability-Plasticity Tradeoff},
    author  = {Isaac Han and Sangyeon Park and Seungwon Oh and Donghu Kim and Hojoon Lee and Kyung-Joong Kim},
    year    = {2026},
    eprint  = {2602.08040},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2602.08040},
}
```

```bibtex
@misc{ash2020warmstartingneuralnetworktraining,
    title   = {On Warm-Starting Neural Network Training},
    author  = {Jordan T. Ash and Ryan P. Adams},
    year    = {2020},
    eprint  = {1910.08475},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/1910.08475},
}
```

```bibtex
@inproceedings{wu2023invertedattention,
    title   = {Inverted-Attention Transformers can Learn Object Representations: Insights from Slot Attention},
    author  = {Yi-Fu Wu and Klaus Greff and Gamaleldin Fathy Elsayed and Michael Curtis Mozer and Thomas Kipf and Sjoerd van Steenkiste},
    booktitle = {UniReps:  the First Workshop on Unifying Representations in Neural Models},
    year    = {2023},
    url     = {https://openreview.net/forum?id=WgQZNoQ5AB}
}
```

```bibtex
@misc{schwarzer2021dataefficientreinforcementlearningselfpredictive,
    title   = {Data-Efficient Reinforcement Learning with Self-Predictive Representations},
    author  = {Max Schwarzer and Ankesh Anand and Rishab Goel and R Devon Hjelm and Aaron Courville and Philip Bachman},
    year    = {2021},
    eprint  = {2007.05929},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2007.05929},
}
```

```bibtex
@misc{teoh2026nextlatentpredictiontransformerslearn,
    title   = {Next-Latent Prediction Transformers Learn Compact World Models},
    author  = {Jayden Teoh and Manan Tomar and Kwangjun Ahn and Edward S. Hu and Tim Pearce and Pratyusha Sharma and Akshay Krishnamurthy and Riashat Islam and Alex Lamb and John Langford},
    year    = {2026},
    eprint  = {2511.05963},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2511.05963},
}
```

*the conquest of nature is to be achieved through number and measure - angels to Descartes in a dream*
