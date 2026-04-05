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
                video = data if is_tensor(data) else data['video']

                loss, (losses, recon_video) = self.model(
                    video,
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

# training from dreams

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
