from __future__ import annotations

import torch
from torch import is_tensor
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator

from adam_atan2_pytorch import MuonAdamAtan2

from dreamer4.dreamer4 import (
    VideoTokenizer,
    DynamicsWorldModel
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
        weight_decay = 0.,
        accelerate_kwargs: dict = dict(),
        optim_kwargs: dict = dict(),
        cpu = False,
    ):
        super().__init__()
        batch_size = min(batch_size, len(dataset))

        self.accelerator = Accelerator(
            cpu = cpu,
            **accelerate_kwargs
        )

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

    def forward(
        self
    ):
        iter_train_dl = cycle(self.train_dataloader)

        for _ in range(self.num_train_steps):
            video = next(iter_train_dl)

            loss = self.model(video)
            self.accelerator.backward(loss)

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optim.step()
            self.optim.zero_grad()

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
        weight_decay = 0.,
        accelerate_kwargs: dict = dict(),
        optim_kwargs: dict = dict(),
        cpu = False,
    ):
        super().__init__()
        batch_size = min(batch_size, len(dataset))

        self.accelerator = Accelerator(
            cpu = cpu,
            **accelerate_kwargs
        )

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

    def forward(
        self
    ):
        iter_train_dl = cycle(self.train_dataloader)

        for _ in range(self.num_train_steps):
            batch_data = next(iter_train_dl)

            # just assume raw video dynamics training if batch_data is a tensor
            # else kwargs for video, actions, rewards

            if is_tensor(batch_data):
                loss = self.model(batch_data)
            else:
                loss = self.model(**batch_data)

            self.accelerator.backward(loss)

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optim.step()
            self.optim.zero_grad()

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
        weight_decay = 0.,
        accelerate_kwargs: dict = dict(),
        optim_kwargs: dict = dict(),
        cpu = False,
    ):
        super().__init__()

        self.accelerator = Accelerator(
            cpu = cpu,
            **accelerate_kwargs
        )

        self.model = model

        optim_kwargs = dict(
            lr = learning_rate,
            weight_decay = weight_decay
        )

        self.policy_head_optim = AdamW(model.policy_head_parameters(), **optim_kwargs)
        self.value_head_optim = AdamW(model.value_head_parameters(), **optim_kwargs)

        self.max_grad_norm = max_grad_norm

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.generate_timesteps = generate_timesteps

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

    def forward(
        self
    ):

        for _ in range(self.num_train_steps):

            dreams = self.unwrapped_model.generate(
                self.generate_timesteps,
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
                self.accelerator.clip_grad_norm_(self.model.policy_head_parameters()(), self.max_grad_norm)

            self.policy_head_optim.step()
            self.policy_head_optim.zero_grad()

            # update value head

            self.accelerator.backward(value_head_loss)

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.model.value_head_parameters(), self.max_grad_norm)

            self.value_head_optim.step()
            self.value_head_optim.zero_grad()

        self.print('training complete')
