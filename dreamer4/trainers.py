from __future__ import annotations

import torch
from einops import rearrange
from torch import is_tensor, tensor
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import Dataset, TensorDataset, DataLoader

from accelerate import Accelerator

from adam_atan2_pytorch import MuonAdamAtan2

from dreamer4.dreamer4 import (
    VideoTokenizer,
    DynamicsWorldModel,
    Experience,
    combine_experiences,
    maybe_cast
)

from ema_pytorch import EMA

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

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
        optim_klass = MuonAdamAtan2,
        batch_size = 16,
        learning_rate = 3e-4,
        max_grad_norm = None,
        num_train_steps = 10_000,
        use_autocast=False,
        autocast_dtype=torch.bfloat16,
        weight_decay = 0.,
        accelerate_kwargs: dict = dict(),
        optim_kwargs: dict = dict(),
        cpu = False,
        use_tensorboard_logger = False,
        log_dir: str | None = None,
        project_name = 'dreamer4',
        log_video = False,
        video_fps = -1,
        log_video_every = 1000
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
            self.video_logger = self.accelerator.get_tracker("tensorboard").writer.add_video

            # Modern python versions have deprecated "\d" which moviepy 1.0.3 uses
            # We can mute those warnings here
            import warnings
            warnings.filterwarnings("ignore", category=SyntaxWarning, module="moviepy")

        self.log_video_flag = log_video

        self.use_autocast = use_autocast
        self.autocast_dtype = autocast_dtype

        self.model = model
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
        self.batch_size = batch_size

        self.register_buffer('step', tensor(0))

        (
            self.model,
            self.train_dataloader,
            self.optim
        ) = self.accelerator.prepare(
            self.model,
            self.train_dataloader,
            self.optim
        )

    @property
    def device(self):
        return self.accelerator.device

    def print(self, *args, **kwargs):
        return self.accelerator.print(*args, **kwargs)

    def log(self, **data):
        self.accelerator.log(data, step = self.step.item())

    def log_video(self, video, tag: str):
        self.video_logger(
            tag,
            rearrange(video, 'b c t h w -> b t c h w'),
            self.step.item(),
            self.fps
        )

    def forward(
        self
    ):
        iter_train_dl = cycle(self.train_dataloader)

        for _ in range(self.num_train_steps):
            video = next(iter_train_dl)
            maybe_cast(video, self.autocast_dtype, self.use_autocast)

            with torch.autocast(enabled=self.use_autocast, dtype=self.autocast_dtype, device_type=self.device.type):
                loss, (losses, recon_video) = self.model(
                    video,
                    update_loss_ema = True,
                    return_intermediates = True
                )

            self.accelerator.backward(loss)

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optim.step()
            self.optim.zero_grad()

            self.log(
                loss = loss.item(),
                recon_loss = losses.recon.item(),
                lpips_loss = losses.lpips.item(),
                time_decorr_loss = losses.time_decorr.item(),
                space_decorr_loss = losses.space_decorr.item()
            )

            if self.log_video_flag and (self.step % self.log_video_every == 0):
                self.log_video(video, "original_video")
                self.log_video(recon_video, "reconstructed_video")

            self.step += 1

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
        max_grad_norm = None,
        num_train_steps = 10_000,
        use_autocast=False,
        autocast_dtype=torch.bfloat16,
        weight_decay = 0.,
        accelerate_kwargs: dict = dict(),
        optim_kwargs: dict = dict(),
        cpu = False,
        use_tensorboard_logger = False,
        log_dir: str | None = None,
        project_name = 'dreamer4'
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

        self.use_autocast = use_autocast
        self.autocast_dtype = autocast_dtype

        self.model = model
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
        self.batch_size = batch_size

        self.register_buffer('step', tensor(0))

        (
            self.model,
            self.train_dataloader,
            self.optim
        ) = self.accelerator.prepare(
            self.model,
            self.train_dataloader,
            self.optim
        )

    @property
    def device(self):
        return self.accelerator.device

    def print(self, *args, **kwargs):
        return self.accelerator.print(*args, **kwargs)

    def log(self, **data):
        self.accelerator.log(data, step = self.step.item())

    def forward(
        self
    ):
        iter_train_dl = cycle(self.train_dataloader)

        for _ in range(self.num_train_steps):
            batch_data = next(iter_train_dl)

            # just assume raw video dynamics training if batch_data is a tensor
            # else kwargs for video, actions, rewards

            if is_tensor(batch_data):
                maybe_cast(batch_data, self.autocast_dtype, self.use_autocast)

                with torch.autocast(enabled=self.use_autocast, dtype=self.autocast_dtype, device_type=self.device.type):
                    loss = self.model(batch_data)
            else:
                for v in batch_data.values():
                    maybe_cast(v, self.autocast_dtype, self.use_autocast)

                with torch.autocast(enabled=self.use_autocast, dtype=self.autocast_dtype, device_type=self.device.type):
                    loss = self.model(**batch_data)

            self.accelerator.backward(loss)

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optim.step()
            self.optim.zero_grad()

            self.log(loss = loss.item())

            self.step += 1

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
        max_grad_norm = None,
        num_train_steps = 10_000,
        use_autocast=False,
        autocast_dtype=torch.bfloat16,
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

        self.use_autocast = use_autocast
        self.autocast_dtype = autocast_dtype

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

    def print(self, *args, **kwargs):
        return self.accelerator.print(*args, **kwargs)

    def log(self, **data):
        self.accelerator.log(data, step = self.step.item())

    def forward(
        self
    ):

        for _ in range(self.num_train_steps):

            with torch.autocast(enabled=self.use_autocast, dtype=self.autocast_dtype, device_type=self.device.type):
                dreams = self.unwrapped_model.generate(
                    self.generate_timesteps + 1, # plus one for bootstrap value
                    batch_size = self.batch_size,
                    return_rewards_per_frame = True,
                    return_agent_actions = True,
                    return_log_probs_and_values = True
                )

            maybe_cast(dreams, self.autocast_dtype, self.use_autocast)

            with torch.autocast(enabled=self.use_autocast, dtype=self.autocast_dtype, device_type=self.device.type):
                policy_head_loss, value_head_loss = self.model.learn_from_experience(dreams)

            self.print(f'policy head loss: {policy_head_loss.item():.3f} | value head loss: {value_head_loss.item():.3f}')

            # update policy head

            self.accelerator.backward(policy_head_loss)

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.model.policy_head_parameters()(), self.max_grad_norm)

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
        use_autocast=False,
        autocast_dtype=torch.bfloat16,
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

        self.use_autocast = use_autocast
        self.autocast_dtype = autocast_dtype

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

                maybe_cast(batch_experience, self.autocast_dtype, self.use_autocast)

                with torch.autocast(enabled=self.use_autocast, dtype=self.autocast_dtype, device_type=self.device.type):
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
                    self.accelerator.clip_grad_norm_(self.model.policy_head_parameters()(), self.max_grad_norm)

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

        for _ in range(num_episodes):

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
