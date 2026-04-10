from __future__ import annotations
import math
from collections import OrderedDict
from random import randint

import torch
from torch import is_tensor, tensor
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torchvision.transforms as T

from einops import rearrange, repeat
from torch_einops_utils import shape_with_replace

from pathlib import Path

from tqdm import tqdm

from accelerate import Accelerator

from adam_atan2_pytorch import MuonAdamAtan2

from dreamer4.dreamer4 import (
    VideoTokenizer,
    DynamicsWorldModel,
    Experience,
    SelfFlow,
    combine_experiences,
    eval_decorator
)

from ema_pytorch import EMA

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def grad_norm(parameters):
    sq_norm = 0.

    for param in parameters:
        if not exists(param.grad):
            continue

        sq_norm = sq_norm + param.grad.detach().float().pow(2).sum().item()

    return sq_norm ** 0.5

def video_tensor_to_gif(
    tensor,
    path,
    duration = 120,
    loop = 0,
    optimize = True
):
    tensor = tensor.clamp(0, 1)
    images = [T.ToPILImage()(img) for img in tensor.unbind(dim = 1)]

    first_img, *rest_imgs = images

    first_img.save(
        path,
        save_all = True,
        append_images = rest_imgs,
        duration = duration,
        loop = loop,
        optimize = optimize
    )

def save_video_grid_as_gif(video, path):
    batch = video.shape[0]
    num_rows = int(math.sqrt(batch))
    num_keep = num_rows ** 2
    video = video[:num_keep]
    grid = rearrange(video, '(row col) c f h w -> c f (row h) (col w)', row = num_rows)
    video_tensor_to_gif(grid.cpu(), str(path))

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

# trainers

class VideoTokenizerTrainer(Module):
    def __init__(
        self,
        model: VideoTokenizer,
        dataset: Dataset,
        *,
        checkpoint_path: str | None = None,
        optim_klass = MuonAdamAtan2,
        batch_size = 16,
        learning_rate = 3e-4,
        max_grad_norm = 0.5,
        num_train_steps = 10_000,
        grad_accum_every = 1,
        weight_decay = 0.,
        accelerate_kwargs: dict = dict(),
        optim_kwargs: dict = dict(),
        cpu = False,
        use_ema = False,
        ema_decay = 0.999,
        use_tensorboard_logger = False,
        log_dir: str | None = None,
        project_name = 'dreamer4',
        log_video = False,
        video_fps = -1,
        log_video_every = 1000,
        checkpoint_every = 2500,
        checkpoint_folder = './checkpoints_tokenizer'
    ):
        super().__init__()
        batch_size = min(batch_size, len(dataset))

        if use_tensorboard_logger:
            accelerate_kwargs.update(log_with = 'tensorboard', project_dir = log_dir)

        self.accelerator = Accelerator(
            cpu = cpu,
            **accelerate_kwargs
        )

        if use_tensorboard_logger:
            self.accelerator.init_trackers(project_name)

        if log_video:
            assert video_fps > 0, "Video fps must be passed in and positive when log_video=True"

            self.fps = video_fps
            self.log_video_every = log_video_every

            self.results_folder = None
            if exists(log_dir):
                self.results_folder = Path(log_dir) / 'results'
                self.results_folder.mkdir(parents = True, exist_ok = True)

            if use_tensorboard_logger:
                self.video_logger = self.accelerator.get_tracker("tensorboard").writer.add_video

        self.log_video_flag = log_video
        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(parents = True, exist_ok = True)

        self.model = model
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.checkpoint_path = checkpoint_path

        self.dataset = dataset
        self.train_dataloader = DataLoader(dataset, batch_size = batch_size, drop_last = True, shuffle = True)

        optim_kwargs = dict(
            lr = learning_rate,
            weight_decay = weight_decay
        )

        if optim_klass is MuonAdamAtan2:
            optim = MuonAdamAtan2(
                model.muon_parameters(),
                model.parameters(),
                **optim_kwargs
            )
        else:
            optim = optim_klass(
                model.parameters(),
                **optim_kwargs
            )

        self.optim = optim

        self.max_grad_norm = max_grad_norm

        self.num_train_steps = num_train_steps
        self.grad_accum_every = grad_accum_every
        self.batch_size = batch_size

        self.register_buffer('step', tensor(0))

        self.ema_model = None
        if self.use_ema and self.accelerator.is_main_process:
            self.ema_model = EMA(
                self.model,
                beta = self.ema_decay
            )

        if exists(self.checkpoint_path):
            self.load(self.checkpoint_path)

        (
            self.model,
            self.train_dataloader,
            self.optim
        ) = self.accelerator.prepare(
            self.model,
            self.train_dataloader,
            self.optim
        )

        if exists(self.ema_model):
            self.ema_model.to(self.device)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main_process(self):
        return self.accelerator.is_main_process

    @property
    def unwrap_model(self):
        return self.accelerator.unwrap_model

    def print(self, *args, **kwargs):
        return self.accelerator.print(*args, **kwargs)

    def log(self, **data):
        self.accelerator.log(data, step = self.step.item())

    def log_video(self, video, tag: str):
        if exists(self.video_logger):
            self.video_logger(
                tag,
                rearrange(video, 'b c t h w -> b t c h w'),
                self.step.item(),
                self.fps
            )

    def load(self, path):
        path = Path(path)
        assert path.exists(), f"checkpoint not found at {path}"

        pkg = torch.load(str(path), map_location = 'cpu', weights_only = True)

        if 'step' in pkg:
            self.step.copy_(tensor(pkg['step']))

        self.unwrap_model(self.model).load_state_dict(pkg.get('model', pkg), strict = False)

        # load ema model if active
        if not self.is_main_process or not self.use_ema or not exists(self.ema_model):
            return

        ema_model = self.ema_model.ema_model

        ema_path = path.parent / f"{path.stem}-ema.pt"

        if not ema_path.exists():
            self.print(f"warning: ema model checkpoint not found at {ema_path}, loading from main model")
            ema_model.load_state_dict(pkg.get('model', pkg), strict = False)
            return

        ema_pkg = torch.load(str(ema_path), map_location = 'cpu', weights_only = True)
        ema_model.load_state_dict(ema_pkg.get('model', ema_pkg), strict = False)

    def forward(
        self
    ):
        iter_train_dl = cycle(self.train_dataloader)

        if self.is_main_process and self.log_video_flag:
            msg = "saving logs to tensorboard" if exists(self.video_logger) else "saving logs"
            if exists(self.results_folder):
                msg += f" and video samples to {self.results_folder}"
            self.print(msg)

        pbar = tqdm(
            range(self.step.item(), self.num_train_steps),
            initial = self.step.item(),
            total = self.num_train_steps,
            disable = not self.is_main_process
        )

        for _ in pbar:

            total_loss = 0.

            for _ in range(self.grad_accum_every):
                data = next(iter_train_dl)

                is_dict = isinstance(data, dict)
                video = data['video'] if is_dict else data
                time_lens = data.get('time_lens') if is_dict else None

                loss, (losses, recon_video) = self.model(
                    video,
                    time_lens = time_lens,
                    update_loss_ema = True,
                    return_intermediates = True
                )

                loss = loss / self.grad_accum_every

                self.accelerator.backward(loss)

                total_loss += loss.item()

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optim.step()
            self.optim.zero_grad()

            if self.use_ema and self.is_main_process:
                self.ema_model.update()

            self.log(
                loss = total_loss,
                recon_loss = losses.recon.item(),
                lpips_loss = losses.lpips.item(),
                time_decorr_loss = losses.time_decorr.item(),
                space_decorr_loss = losses.space_decorr.item(),
                latent_ar_loss = losses.latent_ar.item()
            )

            if self.log_video_flag and divisible_by(self.step.item(), self.log_video_every) and self.is_main_process:

                sample_model = self.ema_model.ema_model if self.use_ema else self.model

                sample_model.eval()

                with torch.no_grad():
                    if self.model.has_flow:
                        latents = sample_model.tokenize(video)
                        recon_video = sample_model.decode(latents, height = video.shape[-2], width = video.shape[-1])
                    else:
                        _, (_, recon_video) = sample_model(video, return_intermediates = True)

                recon_video = recon_video.clamp(0., 1.)
                self.log_video(video, "original_video")
                self.log_video(recon_video, "reconstructed_video")

                if exists(self.results_folder):
                    combined_video = torch.cat((video, recon_video), dim = -1)
                    gif_path = self.results_folder / f'sample-{self.step.item()}.gif'
                    save_video_grid_as_gif(combined_video, gif_path)

            # display active losses in pbar
            postfix_kwargs = dict(loss = f"{total_loss:.4f}")

            recon_loss = losses.recon.item()
            if recon_loss > 0.:
                postfix_kwargs['recon'] = f"{recon_loss:.4f}"

            lpips_loss = losses.lpips.item()
            if lpips_loss > 0.:
                postfix_kwargs['lpips'] = f"{lpips_loss:.4f}"

            time_decorr_loss = losses.time_decorr.item()
            if time_decorr_loss > 0.:
                postfix_kwargs['time_decorr'] = f"{time_decorr_loss:.4f}"

            space_decorr_loss = losses.space_decorr.item()
            if space_decorr_loss > 0.:
                postfix_kwargs['space_decorr'] = f"{space_decorr_loss:.4f}"

            latent_ar_loss = losses.latent_ar.item()
            if latent_ar_loss > 0.:
                postfix_kwargs['latent_ar'] = f"{latent_ar_loss:.4f}"

            pbar.set_postfix(**postfix_kwargs)

            self.step += 1

            self.accelerator.wait_for_everyone()

            if self.checkpoint_every > 0 and self.is_main_process and divisible_by(self.step.item(), self.checkpoint_every):
                ckpt_path = self.checkpoint_folder / f'tokenizer-{self.step.item()}.pt'

                model = self.unwrap_model(self.model)
                config = getattr(model, '_config', None)

                import pickle
                from torch_einops_utils.save_load import dehydrate_config
                pkg = dict(
                    model = model.state_dict(),
                    config = pickle.dumps(dehydrate_config(config, '_config')) if config else None,
                    step = self.step.item()
                )
                torch.save(pkg, str(ckpt_path))

                if self.use_ema:
                    ema_ckpt_path = self.checkpoint_folder / f'tokenizer-{self.step.item()}-ema.pt'
                    ema_model = self.ema_model.ema_model

                    ema_config = getattr(ema_model, '_config', None)
                    ema_pkg = dict(
                        model = ema_model.state_dict(),
                        config = pickle.dumps(dehydrate_config(ema_config, '_config')) if ema_config else None,
                        step = self.step.item()
                    )
                    torch.save(ema_pkg, str(ema_ckpt_path))

                self.print(f"checkpoint saved to {ckpt_path}")

        self.accelerator.end_training()

        self.print('training complete')

# dynamics world model

class BehaviorCloneTrainer(Module):
    def __init__(
        self,
        model: DynamicsWorldModel,
        dataset: Dataset,
        optim_klass = MuonAdamAtan2,
        batch_size = 16,
        learning_rate = 3e-4,
        max_grad_norm = 0.5,
        num_train_steps = 10_000,
        weight_decay = 0.,
        accelerate_kwargs: dict = dict(),
        optim_kwargs: dict = dict(),
        cpu = False,
        use_tensorboard_logger = False,
        log_dir: str | None = None,
        project_name = 'dreamer4',
        log_video = True,
        log_video_every = 100,
        checkpoint_every = 5000,
        checkpoint_folder = './checkpoints',
        video_fps = -1,
        sample_time_steps = 16,
        sample_batch_size = 25,
        sample_prompt_frames = 1,
        sample_sticky_action = False,
        sample_autoregressive_actions = None,
        sample_filename_prefix = 'sample',
        use_ema = False,
        ema_decay = 0.999,
        grad_accum_every = 1,
        self_flow = False,
        self_flow_student_layer = -3,
        self_flow_layer = -1,
        self_flow_loss_weight = 1.0,
        self_flow_kwargs: dict = dict(),
        collate_fn = None,
        custom_sample_fn = None
    ):
        super().__init__()
        batch_size = min(batch_size, len(dataset))

        if use_tensorboard_logger:
            accelerate_kwargs.update(log_with = 'tensorboard', project_dir = log_dir)

        self.accelerator = Accelerator(
            cpu = cpu,
            **accelerate_kwargs
        )

        if use_tensorboard_logger:
            self.accelerator.init_trackers(project_name)

        self.model = model
        self.dataset = dataset
        self.train_dataloader = DataLoader(dataset, batch_size = batch_size, drop_last = True, shuffle = True, collate_fn = collate_fn)

        self.custom_sample_fn = custom_sample_fn

        self.grad_accum_every = grad_accum_every

        # self-flow distillation

        self.self_flow = self_flow
        self.self_flow_loss_weight = self_flow_loss_weight

        self.self_flow_module = None

        if self_flow:
            use_ema = True

            self.self_flow_module = SelfFlow(
                model = model,
                student_layer = self_flow_student_layer,
                teacher_layer = self_flow_teacher_layer,
                **self_flow_kwargs
            )

        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # optimizer

        optim_kwargs = dict(
            lr = learning_rate,
            weight_decay = weight_decay
        )

        model_params = list(model.parameters())
        muon_params = list(model.muon_parameters()) if hasattr(model, 'muon_parameters') else []

        if exists(self.self_flow_module):
            model_params.extend(self.self_flow_module.parameters())

        if optim_klass is MuonAdamAtan2:
            optim = MuonAdamAtan2(
                muon_params,
                model_params,
                **optim_kwargs
            )
        else:
            optim = optim_klass(
                model_params,
                **optim_kwargs
            )

        self.optim = optim

        self.max_grad_norm = max_grad_norm

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size

        self.register_buffer('step', tensor(0))

        self.video_logger = None
        self.results_folder = None

        if log_video:
            assert video_fps > 0, "Video fps must be passed in and positive when log_video=True"

            self.fps = video_fps
            self.log_video_every = log_video_every

            self.results_folder = None
            if exists(log_dir):
                self.results_folder = Path(log_dir) / 'results'
                self.results_folder.mkdir(parents = True, exist_ok = True)

            if use_tensorboard_logger:
                self.video_logger = self.accelerator.get_tracker("tensorboard").writer.add_video

        self.log_video_flag = log_video
        self.sample_time_steps = sample_time_steps
        self.sample_batch_size = sample_batch_size
        self.sample_prompt_frames = sample_prompt_frames
        self.sample_sticky_action = sample_sticky_action
        self.sample_autoregressive_actions = sample_autoregressive_actions
        self.sample_filename_prefix = sample_filename_prefix

        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)

        self.ema_model = None

        if self.use_ema and self.accelerator.is_main_process:
            self.ema_model = EMA(
                self.model,
                beta = self.ema_decay
            )

        # prepare with accelerator

        to_prepare = [self.model, self.train_dataloader, self.optim]

        if exists(self.self_flow_module):
            to_prepare.append(self.self_flow_module)

        self.model, self.train_dataloader, self.optim, *rest = self.accelerator.prepare(*to_prepare)

        if rest:
            self.self_flow_module, = rest

        if exists(self.ema_model):
            self.ema_model.to(self.device)

    @property
    def device(self):
        return self.accelerator.device

    def print(self, *args, **kwargs):
        return self.accelerator.print(*args, **kwargs)

    @property
    def is_main_process(self):
        return self.accelerator.is_main_process

    @property
    def unwrap_model(self):
        return self.accelerator.unwrap_model

    def log(self, **data):
        self.accelerator.log(data, step = self.step.item())

    def log_video(self, video, tag: str):
        if exists(self.video_logger):
            self.video_logger(
                tag,
                rearrange(video, 'b c t h w -> b t c h w'),
                self.step.item(),
                self.fps
            )

    def load(self, path):
        path = Path(path)
        assert path.exists(), f"checkpoint not found at {path}"

        pkg = torch.load(str(path), map_location = 'cpu', weights_only = True)

        if 'step' in pkg:
            self.step.copy_(tensor(pkg['step']))

        self.unwrap_model(self.model).load_state_dict(pkg.get('model', pkg), strict = False)

        # load ema model if active
        if not self.is_main_process or not self.use_ema or not exists(self.ema_model):
            return

        ema_model = self.ema_model.ema_model

        ema_path = path.parent / f"{path.stem}-ema.pt"
        if not ema_path.exists():
            self.print(f"warning: ema model checkpoint not found at {ema_path}, loading from main model")
            ema_model.load_state_dict(pkg.get('model', pkg), strict = False)
            return

        ema_pkg = torch.load(str(ema_path), map_location = 'cpu', weights_only = True)
        ema_model.load_state_dict(ema_pkg.get('model', ema_pkg), strict = False)

    def save_checkpoint(self):
        import pickle
        from torch_einops_utils.save_load import dehydrate_config

        ckpt_path = self.checkpoint_folder / f'dynamics-{self.step.item()}.pt'

        model = self.unwrap_model(self.model)
        config = getattr(model, '_config', None)

        pkg = dict(
            model = model.state_dict(),
            config = pickle.dumps(dehydrate_config(config, '_config')) if config else None,
            step = self.step.item()
        )
        torch.save(pkg, str(ckpt_path))

        if self.use_ema:
            ema_ckpt_path = self.checkpoint_folder / f'dynamics-{self.step.item()}-ema.pt'
            ema_model = self.ema_model.ema_model

            ema_config = getattr(ema_model, '_config', None)
            ema_pkg = dict(
                model = ema_model.state_dict(),
                config = pickle.dumps(dehydrate_config(ema_config, '_config')) if ema_config else None,
                step = self.step.item()
            )
            torch.save(ema_pkg, str(ema_ckpt_path))

        self.print(f"checkpoint saved to {ckpt_path}")

    @eval_decorator
    @torch.no_grad()
    def sample(self, batch_data):
        unwrapped = self.unwrap_model(self.model)

        if is_tensor(batch_data):
            prompt_video = batch_data[:self.sample_batch_size, :, :self.sample_prompt_frames]
            real_video = batch_data[:self.sample_batch_size]
        else:
            prompt_video = batch_data['video'][:self.sample_batch_size, :, :self.sample_prompt_frames]
            real_video = batch_data['video'][:self.sample_batch_size]

        image_size = prompt_video.shape[-1]
        sample_batch_size = prompt_video.shape[0]

        kwargs = dict()
        is_autoregressive = exists(self.sample_autoregressive_actions) and self.sample_autoregressive_actions

        if not is_tensor(batch_data):
            action_idx = max(0, self.sample_prompt_frames - 2)
            for action_key in ('continuous_actions', 'discrete_actions'):
                if action_key not in batch_data:
                    continue

                actions = batch_data[action_key][:self.sample_batch_size]

                # sticky the last provided prompt action and extrapolate it for the rest of generation

                if self.sample_sticky_action:
                    actions = actions[:, action_idx:(action_idx + 1)]
                    actions = repeat(actions, 'b 1 d -> b t d', t = self.sample_time_steps - 1)
                    kwargs[f'prompt_{action_key}'] = actions
                    continue

                seq_len = actions.shape[1]
                target_len = self.sample_time_steps - 1

                if seq_len < target_len and not is_autoregressive:
                    padding = actions[:, -1:]
                    padding = repeat(padding, 'b 1 d -> b t d', t = target_len - seq_len)
                    actions = torch.cat((actions, padding), dim = 1)

                kwargs[f'prompt_{action_key}'] = actions

        generated_video = unwrapped.generate(
            prompt = prompt_video,
            time_steps = self.sample_time_steps,
            batch_size = sample_batch_size,
            image_height = image_size,
            image_width = image_size,
            return_decoded_video = True,
            return_agent_actions = is_autoregressive,
            **kwargs
        )

        generated_video = generated_video.clamp(0., 1.)
        self.log_video(generated_video, 'samples')

        if exists(self.results_folder):
            # Prepend the prompt to the generated video
            generated_video = torch.cat((prompt_video, generated_video), dim = 2)

            gen_len = generated_video.shape[2]
            real_len = real_video.shape[2]

            # pad generated or real video so they match in time length for concat

            if gen_len < real_len:
                pad_shape = shape_with_replace(generated_video, {2: real_len - gen_len})
                padding = generated_video.new_zeros(pad_shape)
                generated_video = torch.cat((generated_video, padding), dim = 2)
            elif real_len < gen_len:
                pad_shape = shape_with_replace(real_video, {2: gen_len - real_len})
                padding = real_video.new_zeros(pad_shape)
                real_video = torch.cat((real_video, padding), dim = 2)

            combined_video = torch.cat((real_video, generated_video), dim = -1)
            gif_path = self.results_folder / f'{self.sample_filename_prefix}-{self.step.item()}.gif'
            save_video_grid_as_gif(combined_video, gif_path)

        if exists(self.custom_sample_fn):
            self.custom_sample_fn(self, batch_data)

    def forward(
        self
    ):
        iter_train_dl = cycle(self.train_dataloader)

        if self.is_main_process and self.log_video_flag:
            msg = "saving logs to tensorboard" if exists(self.video_logger) else "saving logs"
            if exists(self.results_folder):
                msg += f" and video samples to {self.results_folder}"
            self.print(msg)

        pbar = tqdm(
            range(self.step.item(), self.num_train_steps),
            initial = self.step.item(),
            total = self.num_train_steps,
            disable = not self.is_main_process
        )

        last_shortcut_loss = 0.

        for _ in pbar:
            total_loss = 0.
            total_flow_loss = 0.
            total_shortcut_loss = 0.
            total_reward_loss = 0.
            total_discrete_action_loss = 0.
            total_continuous_action_loss = 0.
            total_self_flow_loss = 0.

            for _ in range(self.grad_accum_every):
                batch_data = next(iter_train_dl)

                if is_tensor(batch_data):
                    batch_data = dict(video = batch_data)

                batch_data['return_intermediates'] = True

                if self.self_flow:
                    batch_data['seed'] = randint(0, int(1e7))

                loss, losses, intermediates = self.model(**batch_data, return_all_losses = True)

                loss = loss / self.grad_accum_every

                if self.self_flow:
                    with torch.no_grad():
                        self_flow_loss = self.self_flow_module(self.ema_model.ema_model, intermediates, batch_data)

                    self_flow_loss = self_flow_loss / self.grad_accum_every

                    loss = loss + self_flow_loss * self.self_flow_loss_weight
                    total_self_flow_loss += self_flow_loss.item()

                self.accelerator.backward(loss)

                total_loss += loss.item()
                total_flow_loss += (losses.flow.item() / self.grad_accum_every)
                total_shortcut_loss += (losses.shortcut.item() / self.grad_accum_every)
                total_reward_loss += (losses.rewards.sum().item() / self.grad_accum_every)
                total_discrete_action_loss += (losses.discrete_actions.sum().item() / self.grad_accum_every)
                total_continuous_action_loss += (losses.continuous_actions.sum().item() / self.grad_accum_every)

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optim.step()
            self.optim.zero_grad()

            if self.use_ema and self.is_main_process:
                self.ema_model.update()

            log_dict = dict(
                total_loss = total_loss,
                flow_loss = total_flow_loss,
                shortcut_loss = total_shortcut_loss,
                reward_loss = total_reward_loss,
                discrete_action_loss = total_discrete_action_loss,
                continuous_action_loss = total_continuous_action_loss,
            )

            if self.self_flow:
                log_dict.update(self_flow_loss = total_self_flow_loss)

            self.log(**log_dict)

            postfix = OrderedDict(total = f"{total_loss:.4f}")

            if total_flow_loss > 0.:
                postfix['flow'] = f"{total_flow_loss:.4f}"

            if self.self_flow:
                postfix['self_flow'] = f"{total_self_flow_loss:.4f}"

            if total_shortcut_loss > 0.:
                last_shortcut_loss = total_shortcut_loss

            if last_shortcut_loss > 0.:
                postfix['shortcut'] = f"{last_shortcut_loss:.4f}"

            if total_reward_loss > 0.:
                postfix['reward'] = f"{total_reward_loss:.4f}"

            if total_discrete_action_loss > 0.:
                postfix['disc_act'] = f"{total_discrete_action_loss:.4f}"

            if total_continuous_action_loss > 0.:
                postfix['cont_act'] = f"{total_continuous_action_loss:.4f}"

            pbar.set_postfix(ordered_dict = postfix)

            self.step += 1

            self.accelerator.wait_for_everyone()

            if self.is_main_process and self.log_video_flag and divisible_by(self.step.item(), self.log_video_every):
                self.sample(batch_data)

            if self.checkpoint_every > 0 and self.is_main_process and divisible_by(self.step.item(), self.checkpoint_every):
                self.save_checkpoint()

        self.accelerator.end_training()
        self.print('training complete')

# full dreamer loop - collect from env, train world model, dream, train policy

class DreamTrainer(Module):
    def __init__(
        self,
        model: DynamicsWorldModel,
        optim_klass = AdamW,
        batch_size = 16,
        generate_timesteps = 16,
        learning_rate = 3e-4,
        max_grad_norm = 0.5,
        num_train_steps = 10_000,
        weight_decay = 0.,
        accelerate_kwargs: dict = dict(),
        optim_kwargs: dict = dict(),
        cpu = False,
        use_tensorboard_logger = False,
        log_dir: str | None = None,
        project_name = 'dreamer4'
    ):
        super().__init__()

        if use_tensorboard_logger:
            accelerate_kwargs.update(log_with = 'tensorboard', project_dir = log_dir)

        self.accelerator = Accelerator(
            cpu = cpu,
            **accelerate_kwargs
        )

        if use_tensorboard_logger:
            self.accelerator.init_trackers(project_name)

        self.model = model

        optim_kwargs = dict(
            lr = learning_rate,
            weight_decay = weight_decay
        )

        self.policy_head_optim = optim_klass(model.policy_head_parameters(), **optim_kwargs)
        self.value_head_optim = optim_klass(model.value_head_parameters(), **optim_kwargs)

        self.max_grad_norm = max_grad_norm

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.generate_timesteps = generate_timesteps

        self.register_buffer('step', tensor(0))

        self.unwrapped_model = self.model

        (
            self.model,
            self.policy_head_optim,
            self.value_head_optim,
        ) = self.accelerator.prepare(
            self.model,
            self.policy_head_optim,
            self.value_head_optim
        )

    @property
    def device(self):
        return self.accelerator.device

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    @property
    def is_main_process(self):
        return self.accelerator.is_main_process

    def print(self, *args, **kwargs):
        return self.accelerator.print(*args, **kwargs)

    def log(self, **data):
        self.accelerator.log(data, step = self.step.item())

    def forward(
        self
    ):
        pbar = tqdm(range(self.num_train_steps), disable = not self.is_main_process)

        for _ in pbar:
            dreams = self.unwrapped_model.generate(
                self.generate_timesteps + 1, # plus one for bootstrap value
                batch_size = self.batch_size,
                return_rewards_per_frame = True,
                return_agent_actions = True,
                return_log_probs_and_values = True
            )

            policy_head_loss, value_head_loss = self.model.learn_from_experience(dreams)

            self.print(f'policy head loss: {policy_head_loss.item():.3f} | value head loss: {value_head_loss.item():.3f}')

            # update policy head

            self.accelerator.backward(policy_head_loss)

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.model.policy_head_parameters(), self.max_grad_norm)

            self.policy_head_optim.step()
            self.policy_head_optim.zero_grad()

            # update value head

            self.accelerator.backward(value_head_loss)

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.model.value_head_parameters(), self.max_grad_norm)

            self.value_head_optim.step()
            self.value_head_optim.zero_grad()

            self.log(
                policy_head_loss = policy_head_loss.item(),
                value_head_loss = value_head_loss.item()
            )

            pbar.set_postfix(
                policy_loss = f"{policy_head_loss.item():.4f}",
                value_loss = f"{value_head_loss.item():.4f}"
            )

            self.step += 1

        self.accelerator.end_training()

        self.print('training complete')

# training from sim

class SimTrainer(Module):
    def __init__(
        self,
        model: DynamicsWorldModel,
        optim_klass = AdamW,
        batch_size = 16,
        generate_timesteps = 16,
        learning_rate = 3e-4,
        max_grad_norm = None,
        epochs = 2,
        weight_decay = 0.,
        accelerate_kwargs: dict = dict(),
        optim_kwargs: dict = dict(),
        cpu = False,
        use_tensorboard_logger = False,
        log_dir = None,
        project_name = 'dreamer4'
    ):
        super().__init__()

        if use_tensorboard_logger:
            accelerate_kwargs.update(log_with = 'tensorboard', project_dir = log_dir)

        self.accelerator = Accelerator(
            cpu = cpu,
            **accelerate_kwargs
        )

        if use_tensorboard_logger:
            self.accelerator.init_trackers(project_name)

        self.model = model

        optim_kwargs = dict(
            lr = learning_rate,
            weight_decay = weight_decay
        )

        self.policy_head_optim = optim_klass(model.policy_head_parameters(), **optim_kwargs)
        self.value_head_optim = optim_klass(model.value_head_parameters(), **optim_kwargs)

        self.max_grad_norm = max_grad_norm

        self.epochs = epochs
        self.batch_size = batch_size

        self.generate_timesteps = generate_timesteps

        self.register_buffer('step', torch.tensor(0))

        self.unwrapped_model = self.model

        (
            self.model,
            self.policy_head_optim,
            self.value_head_optim,
        ) = self.accelerator.prepare(
            self.model,
            self.policy_head_optim,
            self.value_head_optim
        )

    @property
    def device(self):
        return self.accelerator.device

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    @property
    def is_main_process(self):
        return self.accelerator.is_main_process

    def log(self, **data):
        self.accelerator.log(data, step = self.step.item())

    def print(self, *args, **kwargs):
        return self.accelerator.print(*args, **kwargs)

    def learn(
        self,
        experience: Experience
    ):

        step_size = experience.step_size
        agent_index = experience.agent_index

        latents = experience.latents
        old_values = experience.values
        rewards = experience.rewards

        has_proprio = exists(experience.proprio)
        proprio = experience.proprio

        has_agent_embed = exists(experience.agent_embed)
        agent_embed = experience.agent_embed

        discrete_actions, continuous_actions = experience.actions
        discrete_log_probs, continuous_log_probs = experience.log_probs

        discrete_old_action_unembeds, continuous_old_action_unembeds = default(experience.old_action_unembeds, (None, None))

        # handle empties

        empty_tensor = torch.empty_like(rewards)

        agent_embed = default(agent_embed, empty_tensor)
        proprio = default(proprio, empty_tensor)

        has_discrete = exists(discrete_actions)
        has_continuous = exists(continuous_actions)

        discrete_actions = default(discrete_actions, empty_tensor)
        continuous_actions = default(continuous_actions, empty_tensor)

        discrete_log_probs = default(discrete_log_probs, empty_tensor)
        continuous_log_probs = default(continuous_log_probs, empty_tensor)

        discrete_old_action_unembeds = default(discrete_old_action_unembeds, empty_tensor)
        continuous_old_action_unembeds = default(continuous_old_action_unembeds, empty_tensor)

        # create the dataset and dataloader

        dataset = TensorDataset(
            latents,
            discrete_actions,
            continuous_actions,
            discrete_log_probs,
            continuous_log_probs,
            agent_embed,
            proprio,
            discrete_old_action_unembeds,
            continuous_old_action_unembeds,
            old_values,
            rewards
        )

        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

        for epoch in range(self.epochs):

            for (
                latents,
                discrete_actions,
                continuous_actions,
                discrete_log_probs,
                continuous_log_probs,
                agent_embed,
                proprio,
                discrete_old_action_unembeds,
                continuous_old_action_unembeds,
                old_values,
                rewards
            ) in dataloader:

                actions = (
                    discrete_actions if has_discrete else None,
                    continuous_actions if has_continuous else None
                )

                log_probs = (
                    discrete_log_probs if has_discrete else None,
                    continuous_log_probs if has_continuous else None
                )

                old_action_unembeds = (
                    discrete_old_action_unembeds if has_discrete else None,
                    continuous_old_action_unembeds if has_continuous else None
                )

                batch_experience = Experience(
                    latents = latents,
                    actions = actions,
                    log_probs = log_probs,
                    agent_embed = agent_embed if has_agent_embed else None,
                    proprio = proprio if has_proprio else None,
                    old_action_unembeds = old_action_unembeds,
                    values = old_values,
                    rewards = rewards,
                    step_size = step_size,
                    agent_index = agent_index
                )

                policy_head_loss, value_head_loss = self.model.learn_from_experience(batch_experience)

                self.print(f'policy head loss: {policy_head_loss.item():.3f} | value head loss: {value_head_loss.item():.3f}')

                self.log(
                    policy_head_loss = policy_head_loss.item(),
                    value_head_loss = value_head_loss.item()
                )

                self.step += 1

                # update policy head

                self.accelerator.backward(policy_head_loss)

                if exists(self.max_grad_norm):
                    self.accelerator.clip_grad_norm_(self.model.policy_head_parameters(), self.max_grad_norm)

                self.policy_head_optim.step()
                self.policy_head_optim.zero_grad()

                # update value head

                self.accelerator.backward(value_head_loss)

                if exists(self.max_grad_norm):
                    self.accelerator.clip_grad_norm_(self.model.value_head_parameters(), self.max_grad_norm)

                self.value_head_optim.step()
                self.value_head_optim.zero_grad()

        self.accelerator.end_training()

        self.print('training complete')

    def forward(
        self,
        env,
        num_episodes = 50000,
        max_experiences_before_learn = 8,
        env_is_vectorized = False
    ):
        pbar = tqdm(range(num_episodes), disable = not self.is_main_process)

        for _ in pbar:
            total_experience = 0
            experiences = []

            while total_experience < max_experiences_before_learn:

                experience = self.unwrapped_model.interact_with_env(env, env_is_vectorized = env_is_vectorized)

                num_experience = experience.video.shape[0]

                total_experience += num_experience

                experiences.append(experience.cpu())

            combined_experiences = combine_experiences(experiences)

            self.learn(combined_experiences)

            experiences.clear()

        self.print('training complete')

# full dreamer loop - collect from env, train world model, dream, train policy

class DreamerTrainer(Module):
    def __init__(
        self,
        model: DynamicsWorldModel,
        optim_klass = AdamW,
        batch_size = 16,
        wm_max_frames_per_batch = 512,
        learning_rate = 3e-4,
        max_grad_norm = 0.5,
        weight_decay = 0.,
        dream_timesteps = 16,
        dream_prompt_len = 1,
        env_max_timesteps = 16,
        wm_collect_frames = 2048,
        dream_train_steps_per_collect = 16,
        wm_only_steps = 0,
        use_pmpo = True,
        accelerate_kwargs: dict = dict(),
        cpu = False,
        use_tensorboard_logger = False,
        log_dir: str | None = None,
        project_name = 'dreamer4',
        checkpoint_every = 1000,
        checkpoint_folder = './checkpoints',
        start_step = 0,
        log_video_every = 50,
    ):
        super().__init__()

        if use_tensorboard_logger:
            accelerate_kwargs.update(log_with = 'tensorboard', project_dir = log_dir)

        self.accelerator = Accelerator(
            cpu = cpu,
            **accelerate_kwargs
        )

        if use_tensorboard_logger:
            self.accelerator.init_trackers(project_name)

        self.model = model

        # world model optimizer
        # keep the policy/action parameters here because the autoregressive action loss trains them
        # exclude only the value head, which is optimized purely from RL targets

        optim_kwargs = dict(lr = learning_rate, weight_decay = weight_decay)

        value_head_params = set(model.value_head_parameters())
        model_params = [p for p in model.parameters() if p not in value_head_params]
        muon_params = list(model.muon_parameters()) if hasattr(model, 'muon_parameters') else []

        if optim_klass is MuonAdamAtan2:
            self.world_model_optim = MuonAdamAtan2(muon_params, model_params, **optim_kwargs)
        else:
            self.world_model_optim = optim_klass(model_params, **optim_kwargs)

        # policy/value optimizers (separate param groups)

        self.policy_head_optim = optim_klass(model.policy_head_parameters(), **optim_kwargs)
        self.value_head_optim = optim_klass(model.value_head_parameters(), **optim_kwargs)

        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.wm_max_frames_per_batch = wm_max_frames_per_batch
        self.dream_timesteps = dream_timesteps
        self.dream_prompt_len = dream_prompt_len
        self.env_max_timesteps = env_max_timesteps
        self.wm_collect_frames = wm_collect_frames
        self.dream_train_steps_per_collect = dream_train_steps_per_collect
        self.wm_only_steps = wm_only_steps
        self.use_pmpo = use_pmpo

        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)

        self.log_video_every = log_video_every
        self.results_folder = Path(checkpoint_folder).parent / 'results'
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.register_buffer('step', tensor(start_step))

        (
            self.model,
            self.world_model_optim,
            self.policy_head_optim,
            self.value_head_optim,
        ) = self.accelerator.prepare(
            self.model,
            self.world_model_optim,
            self.policy_head_optim,
            self.value_head_optim
        )

    @property
    def device(self):
        return self.accelerator.device

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    @property
    def is_main_process(self):
        return self.accelerator.is_main_process

    def print(self, *args, **kwargs):
        return self.accelerator.print(*args, **kwargs)

    def log(self, **data):
        self.accelerator.log(data, step = self.step.item())

    def save_checkpoint(self):
        import pickle
        from torch_einops_utils.save_load import dehydrate_config

        ckpt_path = self.checkpoint_folder / f'dreamer-{self.step.item()}.pt'

        model = self.unwrapped_model
        config = getattr(model, '_config', None)

        pkg = dict(
            model = model.state_dict(),
            config = pickle.dumps(dehydrate_config(config, '_config')) if config else None,
            step = self.step.item()
        )
        torch.save(pkg, str(ckpt_path))
        self.print(f"checkpoint saved to {ckpt_path}")

    def _slice_experience(self, experience: Experience, indices):
        """Slice an experience along the batch dimension, trimming bootstrap padding to match video length."""
        video = experience.video[indices]
        time = video.shape[2]  # video is (b, c, t, h, w)
        rewards = experience.rewards[indices][..., :time] if exists(experience.rewards) else None
        terminals = experience.terminals[indices] if exists(experience.terminals) else None
        discrete_actions = experience.actions[0][indices][:, :time] if exists(experience.actions[0]) else None
        continuous_actions = experience.actions[1][indices][:, :time] if exists(experience.actions[1]) else None
        lens = experience.lens[indices] if exists(experience.lens) else None

        # clamp lens to video length (bootstrap may have incremented them)
        if exists(lens):
            lens = lens.clamp(max = time)

        return video, rewards, terminals, discrete_actions, continuous_actions, lens

    def _transition_lens(self, experience: Experience):
        if exists(experience.latents):
            payload = experience.latents
            seq_len = payload.shape[1]
        else:
            payload = experience.video
            seq_len = payload.shape[2]

        lens = default(experience.lens, torch.full((payload.shape[0],), seq_len, device = self.device))

        transition_lens = lens.clone()

        if exists(experience.rewards):
            transition_lens = transition_lens.clamp(max = experience.rewards.shape[1])

        if exists(experience.actions):
            discrete_actions, continuous_actions = experience.actions

            if exists(discrete_actions):
                transition_lens = transition_lens.clamp(max = discrete_actions.shape[1])

            if exists(continuous_actions):
                transition_lens = transition_lens.clamp(max = continuous_actions.shape[1])

        return transition_lens

    def _make_frame_batches(self, lens, max_frames):
        """Group episodes into batches respecting a frame budget. Returns list of index tensors."""
        # sort by length so similar-length episodes pack together (less padding waste)
        sorted_indices = lens.argsort()
        batches = []
        current_batch = []
        current_max_len = 0

        for idx in sorted_indices:
            ep_len = lens[idx].item()
            new_max_len = max(current_max_len, ep_len)
            new_frames = new_max_len * (len(current_batch) + 1)

            if current_batch and new_frames > max_frames:
                batches.append(torch.tensor(current_batch, device = lens.device))
                current_batch = [idx.item()]
                current_max_len = ep_len
            else:
                current_batch.append(idx.item())
                current_max_len = new_max_len

        if current_batch:
            batches.append(torch.tensor(current_batch, device = lens.device))

        return batches

    def train_world_model(self, experience: Experience):
        """Train world model on collected experience for 1 epoch with frame-budget batching."""

        ep_lens = experience.lens if exists(experience.lens) else torch.full((experience.video.shape[0],), experience.video.shape[2], device = self.device)
        batches = self._make_frame_batches(ep_lens, self.wm_max_frames_per_batch)

        # shuffle batch order
        batch_order = torch.randperm(len(batches))

        wm_loss_breakdown = None
        total_loss = 0.
        num_batches = 0

        for bi in batch_order:
            batch_indices = batches[bi]
            video, rewards, terminals, discrete_actions, continuous_actions, lens = self._slice_experience(experience, batch_indices)

            loss, losses = self.model(
                video = video,
                rewards = rewards,
                terminals = terminals,
                discrete_actions = discrete_actions,
                continuous_actions = continuous_actions,
                lens = lens,
                return_all_losses = True,
            )

            wm_loss_breakdown = dict(
                flow_loss = losses.flow.item(),
                shortcut_loss = losses.shortcut.item(),
                reward_loss = losses.rewards.sum().item(),
                terminal_loss = losses.terminals.sum().item(),
                **getattr(self.unwrapped_model, '_cont_diagnostics', {}),
                discrete_action_loss = losses.discrete_actions.sum().item(),
                continuous_action_loss = losses.continuous_actions.sum().item(),
                state_pred_loss = losses.state_pred.item(),
                agent_state_pred_loss = losses.agent_state_pred.item(),
                latent_ar_loss = losses.latent_ar.item(),
                latent_ar_sigreg_loss = losses.latent_ar_sigreg.item(),
            )

            self.accelerator.backward(loss)

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.unwrapped_model.parameters(), self.max_grad_norm)

            self.world_model_optim.step()
            self.world_model_optim.zero_grad()
            self.policy_head_optim.zero_grad()
            self.value_head_optim.zero_grad()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        return tensor(avg_loss, device = self.device), wm_loss_breakdown

    def _sample_dream_prompts(self, experience: Experience):
        return self._sample_dream_prompts_with_metadata(experience)[0]

    def _sample_dream_prompts_with_metadata(self, experience: Experience):
        if self.dream_prompt_len <= 0:
            return {}, None

        latents = experience.latents
        lens = self._transition_lens(experience)

        if not exists(latents) or not exists(lens):
            return {}, None

        valid_episode_mask = lens >= self.dream_prompt_len

        if not valid_episode_mask.any():
            return {}, None

        valid_episode_indices = valid_episode_mask.nonzero(as_tuple = True)[0]
        sampled_episode_indices = valid_episode_indices[torch.randint(len(valid_episode_indices), (self.batch_size,), device = latents.device)]

        sampled_lens = lens[sampled_episode_indices]
        max_start = sampled_lens - self.dream_prompt_len
        start_offsets = (torch.rand((self.batch_size,), device = latents.device) * (max_start + 1).float()).floor().long()
        time_offsets = torch.arange(self.dream_prompt_len, device = latents.device)
        time_indices = start_offsets[:, None] + time_offsets

        prompt_kwargs = dict(
            prompt_latents = latents[sampled_episode_indices[:, None], time_indices]
        )

        if exists(experience.proprio):
            prompt_kwargs['prompt_proprio'] = experience.proprio[sampled_episode_indices[:, None], time_indices]

        if exists(experience.rewards):
            prompt_kwargs['prompt_rewards'] = experience.rewards[sampled_episode_indices[:, None], time_indices]

        if exists(experience.actions):
            discrete_actions, continuous_actions = experience.actions

            if exists(discrete_actions):
                prompt_kwargs['prompt_discrete_actions'] = discrete_actions[sampled_episode_indices[:, None], time_indices]

            if exists(continuous_actions):
                prompt_kwargs['prompt_continuous_actions'] = continuous_actions[sampled_episode_indices[:, None], time_indices]

        prompt_metadata = dict(
            episode_indices = sampled_episode_indices,
            start_offsets = start_offsets,
            prompt_len = self.dream_prompt_len
        )

        return prompt_kwargs, prompt_metadata

    @torch.no_grad()
    def _compute_dream_alignment_diagnostics(
        self,
        experience: Experience,
        dreams: Experience,
        prompt_metadata: dict | None
    ):
        if (
            not exists(prompt_metadata) or
            not exists(experience.rewards) or
            not exists(experience.lens) or
            not exists(dreams.rewards) or
            not exists(dreams.lens)
        ):
            return {}

        episode_indices = prompt_metadata['episode_indices']
        start_offsets = prompt_metadata['start_offsets']
        prompt_len = prompt_metadata['prompt_len']
        transition_lens = self._transition_lens(experience)

        real_start = start_offsets + prompt_len
        real_remaining = (transition_lens[episode_indices] - real_start).clamp(min = 0)
        dream_learnable_lens = (dreams.lens - dreams.is_truncated.long()).clamp(min = 0)

        diagnostics = dict(
            dream_rollout_len_mean = dream_learnable_lens.float().mean().item(),
            real_future_len_mean = real_remaining.float().mean().item(),
            dream_rollout_len_minus_real_future_len = (dream_learnable_lens.float() - real_remaining.float()).mean().item(),
            dream_real_len_mae = (dream_learnable_lens.float() - real_remaining.float()).abs().mean().item(),
        )

        compare_horizon = min(dreams.rewards.shape[1], experience.rewards.shape[1])

        if compare_horizon <= 0:
            return diagnostics

        device = dreams.rewards.device
        time_offsets = torch.arange(compare_horizon, device = device)
        future_time = real_start[:, None] + time_offsets
        clamped_future_time = future_time.clamp(max = experience.rewards.shape[1] - 1)

        real_future_rewards = experience.rewards[episode_indices[:, None], clamped_future_time]
        dream_future_rewards = dreams.rewards[:, :compare_horizon]

        real_mask = time_offsets < real_remaining[:, None]
        dream_mask = time_offsets < dream_learnable_lens[:, None]
        shared_mask = real_mask & dream_mask

        diagnostics['dream_shared_horizon_mean'] = shared_mask.sum(dim = -1).float().mean().item()

        if shared_mask.any():
            gamma = self.unwrapped_model.gae_discount_factor
            discount = torch.full((compare_horizon,), gamma, device = device).pow(time_offsets)

            masked_real_rewards = real_future_rewards.masked_fill(~shared_mask, 0.)
            masked_dream_rewards = dream_future_rewards.masked_fill(~shared_mask, 0.)

            real_reward_sums = masked_real_rewards.sum(dim = -1)
            dream_reward_sums = masked_dream_rewards.sum(dim = -1)

            real_discounted_returns = (masked_real_rewards * discount).sum(dim = -1)
            dream_discounted_returns = (masked_dream_rewards * discount).sum(dim = -1)

            diagnostics.update(
                dream_reward_sum_mean = dream_reward_sums.mean().item(),
                real_reward_sum_mean = real_reward_sums.mean().item(),
                dream_minus_real_reward_sum = (dream_reward_sums - real_reward_sums).mean().item(),
                dream_real_reward_sum_mae = (dream_reward_sums - real_reward_sums).abs().mean().item(),
                dream_discounted_return_mean = dream_discounted_returns.mean().item(),
                real_discounted_return_mean = real_discounted_returns.mean().item(),
                dream_minus_real_discounted_return = (dream_discounted_returns - real_discounted_returns).mean().item(),
                dream_real_discounted_return_mae = (dream_discounted_returns - real_discounted_returns).abs().mean().item(),
            )

            if exists(dreams.values):
                diagnostics.update(
                    dream_value0_mean = dreams.values[:, 0].mean().item(),
                    dream_value0_dream_return_mae = (dreams.values[:, 0] - dream_discounted_returns).abs().mean().item(),
                    dream_value0_real_return_mae = (dreams.values[:, 0] - real_discounted_returns).abs().mean().item(),
                )

        if exists(dreams.continuation_probs):
            is_truncated = default(experience.is_truncated, torch.ones_like(experience.lens, dtype = torch.bool))
            known_terminal_mask = ~is_truncated[episode_indices].bool()

            if known_terminal_mask.any():
                continuation_probs = dreams.continuation_probs.clamp(0., 1.)

                for horizon in (5, 10, 20):
                    if horizon > continuation_probs.shape[1]:
                        continue

                    horizon_offsets = time_offsets[:horizon]
                    horizon_valid = horizon_offsets < dream_learnable_lens[:, None]
                    horizon_cont = continuation_probs[:, :horizon].masked_fill(~horizon_valid, 1.)
                    dream_terminal_prob = 1. - horizon_cont.prod(dim = -1)

                    real_terminal_by_horizon = (real_remaining <= horizon).float()

                    masked_pred = dream_terminal_prob[known_terminal_mask]
                    masked_real = real_terminal_by_horizon[known_terminal_mask]

                    diagnostics.update(
                        {f'dream_terminal_prob_h{horizon}_mean': masked_pred.mean().item()},
                        **{f'real_terminal_rate_h{horizon}': masked_real.mean().item()},
                        **{f'dream_terminal_overconfidence_h{horizon}': (masked_pred - masked_real).mean().item()},
                        **{f'dream_terminal_brier_h{horizon}': ((masked_pred - masked_real) ** 2).mean().item()},
                    )

        return diagnostics

    def train_policy_from_dreams(self, experience: Experience):
        """Generate dream rollouts and train policy/value heads."""

        prompt_kwargs, prompt_metadata = self._sample_dream_prompts_with_metadata(experience)
        prompt_len = self.dream_prompt_len if 'prompt_latents' in prompt_kwargs else 0
        dream_time_steps = self.dream_timesteps + 1 + prompt_len
        dream_alignment_diagnostics = {}

        for _ in range(self.dream_train_steps_per_collect):
            dreams = self.unwrapped_model.generate(
                dream_time_steps,
                batch_size = self.batch_size,
                return_for_policy_optimization = True,
                return_decoded_video = False,
                drop_prompt_from_experience = True,
                **prompt_kwargs,
            )

            dream_alignment_diagnostics = self._compute_dream_alignment_diagnostics(experience, dreams, prompt_metadata)
            policy_loss, value_loss = self.model.learn_from_experience(dreams, use_pmpo = self.use_pmpo)

            self.accelerator.backward(policy_loss)

            policy_grad_norm = grad_norm(self.model.policy_head_parameters())

            self.unwrapped_model._rl_diagnostics.update(
                policy_grad_norm = policy_grad_norm
            )

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.model.policy_head_parameters(), self.max_grad_norm)

            self.policy_head_optim.step()
            self.policy_head_optim.zero_grad()

            self.accelerator.backward(value_loss)

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.model.value_head_parameters(), self.max_grad_norm)

            self.value_head_optim.step()
            self.value_head_optim.zero_grad()

        return policy_loss, value_loss, dream_alignment_diagnostics

    @torch.no_grad()
    def save_visualizations(self, env, env_is_vectorized = False):
        """Save dream rollout and policy episode GIFs to results folder."""
        step = self.step.item()
        model = self.unwrapped_model
        was_training = model.training
        model.eval()

        # dream rollout
        try:
            dreams = model.generate(
                self.dream_timesteps + 1,
                batch_size = 1,
                return_decoded_video = True,
                return_rewards_per_frame = True,
                return_agent_actions = True,
            )

            if exists(dreams.video):
                dream_video = dreams.video.clamp(0., 1.)
                gif_path = self.results_folder / f'dream-{step}.gif'
                save_video_grid_as_gif(dream_video, gif_path)
        except Exception as e:
            self.print(f'warning: dream visualization failed: {e}')

        # policy episode in real env
        try:
            experience = model.interact_with_env(
                env,
                max_timesteps = min(self.env_max_timesteps, 200),
                env_is_vectorized = env_is_vectorized,
            )

            video = experience.video[:1].clamp(0., 1.)  # first env only
            ep_len = experience.lens[0].item() if exists(experience.lens) else video.shape[2]
            video = video[:, :, :ep_len]

            gif_path = self.results_folder / f'policy-{step}.gif'
            save_video_grid_as_gif(video, gif_path)
        except Exception as e:
            self.print(f'warning: policy visualization failed: {e}')

        if was_training:
            model.train()

    def forward(
        self,
        env,
        num_episodes = 5000,
        max_steps: int | None = None,
        env_is_vectorized = False,
    ):
        pbar = tqdm(range(num_episodes), disable = not self.is_main_process)

        for _ in pbar:

            # 1. collect experience from env until we have enough frames

            experiences = []
            total_frames = 0

            while total_frames < self.wm_collect_frames:
                experience = self.unwrapped_model.interact_with_env(
                    env,
                    max_timesteps = self.env_max_timesteps,
                    env_is_vectorized = env_is_vectorized,
                    store_final_observation = True
                )

                num_frames = experience.video.shape[0] * experience.video.shape[2]  # batch * time
                total_frames += num_frames
                experiences.append(experience.cpu())

            combined = combine_experiences(experiences)
            combined = combined.to(self.device)

            # log episode stats

            # mask rewards by episode length to get true episode reward
            if exists(combined.lens):
                reward_mask = torch.arange(combined.rewards.shape[-1], device = self.device).unsqueeze(0) < combined.lens.unsqueeze(-1)
                ep_reward = (combined.rewards * reward_mask).sum(dim = -1).mean().item()
            else:
                ep_reward = combined.rewards.sum(dim = -1).mean().item()

            ep_length = self._transition_lens(combined).float().mean().item()

            experiences.clear()

            # 2. train world model on collected experience

            wm_loss, wm_loss_breakdown = self.train_world_model(combined)

            # 3. train policy/value from dream rollouts (skip during WM warmup)

            if self.step.item() >= self.wm_only_steps:
                policy_loss, value_loss, dream_alignment_diagnostics = self.train_policy_from_dreams(combined)
            else:
                policy_loss = value_loss = tensor(0., device = self.device)
                dream_alignment_diagnostics = {}

            # logging

            log_data = dict(
                world_model_loss = wm_loss.item(),
                policy_loss = policy_loss.item(),
                value_loss = value_loss.item(),
                episode_reward = ep_reward,
                episode_length = ep_length,
                collected_frames = total_frames,
                wm_batch_count = combined.video.shape[0],
            )

            if exists(wm_loss_breakdown):
                log_data.update(wm_loss_breakdown)

            log_data.update(getattr(self.unwrapped_model, '_rl_diagnostics', {}))
            log_data.update(dream_alignment_diagnostics)

            self.log(**log_data)

            postfix = OrderedDict(
                wm = f"{wm_loss.item():.4f}",
                policy = f"{policy_loss.item():.4f}",
                value = f"{value_loss.item():.4f}",
                reward = f"{ep_reward:.1f}",
            )

            if exists(wm_loss_breakdown):
                if wm_loss_breakdown['flow_loss'] > 0.:
                    postfix['flow'] = f"{wm_loss_breakdown['flow_loss']:.4f}"

                if wm_loss_breakdown['reward_loss'] > 0.:
                    postfix['wm_reward'] = f"{wm_loss_breakdown['reward_loss']:.4f}"

                if wm_loss_breakdown['discrete_action_loss'] > 0.:
                    postfix['disc_act'] = f"{wm_loss_breakdown['discrete_action_loss']:.4f}"

                if wm_loss_breakdown['continuous_action_loss'] > 0.:
                    postfix['cont_act'] = f"{wm_loss_breakdown['continuous_action_loss']:.4f}"

            if 'dream_real_discounted_return_mae' in dream_alignment_diagnostics:
                postfix['dream_ret_gap'] = f"{dream_alignment_diagnostics['dream_real_discounted_return_mae']:.2f}"

            if 'dream_rollout_len_minus_real_future_len' in dream_alignment_diagnostics:
                postfix['dream_len_bias'] = f"{dream_alignment_diagnostics['dream_rollout_len_minus_real_future_len']:.2f}"

            if 'dream_terminal_overconfidence_h5' in dream_alignment_diagnostics:
                postfix['term_bias5'] = f"{dream_alignment_diagnostics['dream_terminal_overconfidence_h5']:.2f}"

            pbar.set_postfix(ordered_dict = postfix)

            if self.log_video_every > 0 and self.is_main_process and divisible_by(self.step.item(), self.log_video_every):
                self.save_visualizations(env, env_is_vectorized)

            self.step += 1

            if self.checkpoint_every > 0 and self.is_main_process and divisible_by(self.step.item(), self.checkpoint_every):
                self.save_checkpoint()

            if exists(max_steps) and self.step.item() >= max_steps:
                break

        self.accelerator.end_training()
        self.print('training complete')
