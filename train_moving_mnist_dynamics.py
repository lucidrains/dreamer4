# /// script
# dependencies = [
#   "torch",
#   "torchvision",
#   "fire",
#   "tqdm",
#   "numpy",
#   "einops",
#   "moviepy",
#   "imageio",
#   "requests",
#   "accelerate",
#   "adam-atan2-pytorch",
#   "tensorboard",
#   "torch-einops-utils",
#   "dreamer4"
# ]
# [tool.uv.sources]
# dreamer4 = { path = "." }
# ///

from pathlib import Path

import fire
from tqdm import tqdm

from dataset_moving_mnist import MovingMNISTDataset
from dreamer4.dreamer4 import VideoTokenizer, DynamicsWorldModel
from dreamer4.trainers import BehaviorCloneTrainer

def main(
    tokenizer_checkpoint_path: str,
    num_frames = 10,
    num_train_steps = 100_000,
    batch_size = 32,
    lr = 3e-4,
    dim = 512,
    depth = 6,
    attn_dim_head = 64,
    attn_heads = 8,
    use_ema = True,
    video_fps = 4,
    log_video_every = 100,
    log_dir = './logs_mnist_dynamics',
    checkpoint_every = 5000,
    checkpoint_folder = './checkpoints_mnist_dynamics',
):
    import shutil

    # clear old artifacts

    log_path = Path(log_dir)
    if log_path.exists():
        shutil.rmtree(log_path)

    log_path.mkdir(exist_ok = True, parents = True)

    # instantiate the dataset

    dataset = MovingMNISTDataset(num_frames = num_frames)

    # Load frozen tokenizer

    assert Path(tokenizer_checkpoint_path).exists(), f"Tokenizer checkpoint missing at {tokenizer_checkpoint_path}"
    
    tokenizer = VideoTokenizer.init_and_load(tokenizer_checkpoint_path)
    tokenizer.eval().requires_grad_(False)

    # initialize world model

    model = DynamicsWorldModel(
        video_tokenizer = tokenizer,
        dim_latent = tokenizer.dim_latent,
        dim = dim,
        depth = depth,
        attn_dim_head = attn_dim_head,
        attn_heads = attn_heads,
    )

    # initialize trainer

    trainer = BehaviorCloneTrainer(
        model = model,
        dataset = dataset,
        batch_size = batch_size,
        learning_rate = lr,
        num_train_steps = num_train_steps,
        use_tensorboard_logger = True,
        log_dir = log_dir,
        log_video = True,
        video_fps = video_fps,
        log_video_every = log_video_every,
        checkpoint_every = checkpoint_every,
        checkpoint_folder = checkpoint_folder,
        use_ema = use_ema
    )

    # Train dynamics model

    trainer()

if __name__ == '__main__':
    fire.Fire(main)
