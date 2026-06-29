import shutil
import numpy as np
import imageio
from pathlib import Path
from functools import reduce
from typing import Callable

import torch
from torch import is_tensor, tensor
from torch.utils._pytree import tree_map

from einops import rearrange, repeat

from dreamer4.dreamer4 import DynamicsWorldModel, cast_to_tensor

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def get_by_dotpath(d, path):
    if not exists(path):
        return d

    def extract(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    return reduce(extract, path.split('.'), d)

# classes

class BaseRecordEnvWrapper:
    def __init__(
        self,
        env,
        obs_image_dotpath = None,
        rewards = False,
        terminated = False,
        dotpaths = None
    ):
        self.env = env
        self.obs_image_dotpath = obs_image_dotpath
        self.current_episode = 0
        self.frames = []
        self.actions = []

        self.dotpaths = default(dotpaths, dict()).copy()

        if rewards:
            self.dotpaths['rewards'] = 'reward'
        if terminated:
            self.dotpaths['terminated'] = 'terminated'

        self.records = {k: [] for k in self.dotpaths.keys()}

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def _extract_image(self, obs):

        # extract image from observation

        if exists(self.obs_image_dotpath):
            img = get_by_dotpath(obs, self.obs_image_dotpath)
        elif isinstance(obs, dict):
            img = default(obs.get('image'), obs.get('state'))
        else:
            img = obs

        img = img if is_tensor(img) else tensor(np.array(img))
        img = img.detach().cpu()

        # handle batched environments

        if img.ndim == 4 and img.shape[1] in (1, 3, 4):
            img = img[0]

        # channel last

        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = rearrange(img, 'c h w -> h w c')

        # scale to 255

        if img.is_floating_point() and img.amax() <= 1.0:
            img = img * 255

        img = img.byte()

        # ensure rgb

        if img.ndim == 2:
            img = rearrange(img, 'h w -> h w 1')

        if img.shape[-1] == 1:
            img = repeat(img, 'h w 1 -> h w c', c = 3)

        return img.numpy()

    def _save_episode(self):
        raise NotImplementedError

    def _clear_records(self):
        self.frames.clear()
        self.actions.clear()
        for v in self.records.values():
            v.clear()

    def _append_records(self, **step_data):
        for name, dotpath in self.dotpaths.items():
            val = get_by_dotpath(step_data, dotpath)
            if exists(val):
                self.records[name].append(val)

    def close(self):
        self._save_episode()
        self._clear_records()
        if hasattr(self.env, 'close'):
            self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.flush()

    def flush(self):
        self._save_episode()
        self._clear_records()

    def wrap_innermost(self, wrapper_cls, *args, **kwargs):
        curr = self

        while True:
            if isinstance(curr, wrapper_cls) or isinstance(curr.env, wrapper_cls):
                return

            if not hasattr(curr.env, 'env'):
                break

            curr = curr.env

        curr.env = wrapper_cls(curr.env, *args, **kwargs)

    def reset(self, **kwargs):
        self.flush()

        obs = self.env.reset(**kwargs)
        obs_val = obs[0] if isinstance(obs, tuple) else obs
        info_val = obs[1] if isinstance(obs, tuple) and len(obs) == 2 else dict()

        self.frames.append(self._extract_image(obs_val))
        self._append_records(obs=obs_val, info=info_val)

        return obs

    def step(self, action):
        out = self.env.step(action)

        # robustly parse termination signals

        reward = 0.
        terminated = False
        info = dict()

        if not isinstance(out, tuple):
            obs = out
            done = False
        elif len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = (terminated | truncated) if is_tensor(terminated) else (terminated or truncated)
        elif len(out) == 4:
            obs, reward, done, info = out
            terminated = done
        elif len(out) == 3:
            obs, reward, done = out
            terminated = done
        elif len(out) == 2:
            obs, reward = out
            done = False
        else:
            obs = out[0]
            done = False

        if is_tensor(done) or isinstance(done, np.ndarray):
            done = bool(done.flatten()[0])

        action_val = action.detach().cpu().numpy() if is_tensor(action) else np.array(action)

        self.actions.append(action_val)
        self.frames.append(self._extract_image(obs))

        self._append_records(obs=obs, reward=reward, terminated=terminated, info=info)

        # auto-save on episode end

        if done:
            self._save_episode()
            self._clear_records()

        return out

    @property
    def parsed_actions(self):
        discrete, continuous = [], []

        for act in self.actions:
            if isinstance(act, tuple):
                disc, cont = act
            elif np.issubdtype(np.asarray(act).dtype, np.integer):
                disc, cont = act, []
            else:
                disc, cont = [], act

            discrete.append(np.asarray(disc))
            continuous.append(np.asarray(cont))

        discrete = discrete if any(d.size > 0 for d in discrete) else []
        continuous = continuous if any(c.size > 0 for c in continuous) else []

        return discrete, continuous

    @property
    def records_to_save(self):
        disc_actions, cont_actions = self.parsed_actions

        return dict(
            discrete_actions = disc_actions,
            continuous_actions = cont_actions,
            **self.records
        )

class RecordToFolderEnvWrapper(BaseRecordEnvWrapper):
    def __init__(
        self,
        env,
        folder,
        obs_image_dotpath = None,
        fps = 20,
        clear_on_start = True,
        **kwargs
    ):
        super().__init__(env, obs_image_dotpath, **kwargs)
        self.folder = folder
        self.fps = fps

        if clear_on_start:
            shutil.rmtree(folder, ignore_errors = True)

        Path(folder).mkdir(parents = True, exist_ok = True)
        self.current_episode = 0

    def _save_episode(self):
        if not self.frames:
            return

        path = Path(self.folder)
        vid_path = path / f'episode_{self.current_episode}.mp4'
        imageio.mimwrite(str(vid_path), self.frames, fps = self.fps, macro_block_size = 1)

        for key, values in self.records_to_save.items():
            if len(values) == 0:
                continue

            np.save(str(path / f'episode_{self.current_episode}.{key}.npy'), np.stack(values))

        self.current_episode += 1

class RecordToReplayBufferEnvWrapper(BaseRecordEnvWrapper):
    def __init__(
        self,
        env,
        replay_buffer,
        obs_image_dotpath = None,
        clear_on_start = False,
        **kwargs
    ):
        super().__init__(env, obs_image_dotpath, **kwargs)
        self.replay_buffer = replay_buffer

        if clear_on_start:
            self.replay_buffer.clear()

    def _save_episode(self):
        if not self.frames:
            return

        video_len = len(self.frames)
        records_to_save = self.records_to_save

        with self.replay_buffer.one_episode():
            for i in range(video_len):
                frame = rearrange(self.frames[i], 'h w c -> c h w')

                kwargs = dict(
                    video = frame,
                    **{k: v[i] for k, v in records_to_save.items() if i < len(v)}
                )

                self.replay_buffer.store(**kwargs)

        self.current_episode += 1

class ActionTransformWrapper:
    def __init__(
        self,
        env,
        transform_fn: Callable,
        clip = None,
        action_info_key = 'env_received_action'
    ):
        self.env = env
        self.transform_fn = transform_fn
        self.clip = clip
        self.action_info_key = action_info_key

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def step(self, action):
        scaled_action = self.transform_fn(action)

        if exists(self.clip):
            min_val, max_val = self.clip
            scaled_action = np.clip(scaled_action, min_val, max_val)

        out = self.env.step(scaled_action)
        out = list(out) if isinstance(out, tuple) else [out, 0., False, False, dict()]

        while len(out) < 4:
            out.append(dict() if len(out) == 3 else False if len(out) == 2 else 0.)

        info = default(out[-1], dict())
        info[self.action_info_key] = scaled_action
        out[-1] = info

        return tuple(out)

# a way to wrap a dynamics world model for a standard interface like a real env

class DynamicsWorldModelWrapper:

    def __init__(
        self,
        world_model: DynamicsWorldModel,
        prompt = None,
        num_generation_steps = 4,
        max_steps = 1000,
        device = None,
        image_size = None
    ):
        self.world_model = world_model
        self.world_model.eval()
        self.num_generation_steps = num_generation_steps
        self.prompt = prompt
        self.max_steps = max_steps
        self.device = default(device, world_model.device)

        tokenizer = world_model.video_tokenizer
        self.image_size = default(image_size, getattr(tokenizer, 'image_height', None) if exists(tokenizer) else None)

        self.clear()

    def clear(self):
        self._latents = None
        self._discrete_actions = None
        self._continuous_actions = None
        self._rewards = None
        self._time_cache = None
        self._current_step = 0

    def reset(self, batch_size = None, seed = None):

        if exists(self.prompt):
            prompt_batch = self.prompt.shape[0]
            assert not exists(batch_size) or batch_size == prompt_batch, f'batch size {batch_size} must match prompt batch size {prompt_batch}'
            batch_size = prompt_batch

        batch_size = default(batch_size, 1)

        device_kwargs = dict(device = self.device)

        # maybe set seed

        if exists(seed):
            torch.manual_seed(seed)

        self.clear()

        # generate initial frame

        with torch.no_grad():
            gen_out = self.world_model.generate(
                batch_size = batch_size,
                time_steps = 1,
                num_steps = self.num_generation_steps,
                prompt = self.prompt,
                image_height = self.image_size,
                image_width = self.image_size,
                return_decoded_video = True,
                return_rewards_per_frame = True,
                return_terminals = True,
                use_time_cache = True,
                return_time_cache = True
            )

        if isinstance(gen_out, tuple):
            gen_out, self._time_cache = gen_out

        # track latents and rewards

        self._latents = gen_out.latents

        if exists(gen_out.rewards):
            self._rewards = gen_out.rewards

        # initialize action histories

        num_discrete = self.world_model.action_embedder.num_discrete_action_types
        num_continuous = self.world_model.action_embedder.num_continuous_action_types

        batch = self._latents.shape[0]

        if num_discrete > 0:
            self._discrete_actions = torch.empty((batch, 0, num_discrete), dtype=torch.long, **device_kwargs)
        if num_continuous > 0:
            self._continuous_actions = torch.empty((batch, 0, num_continuous), dtype=torch.float, **device_kwargs)

        obs = gen_out.video[:, :, -1] if exists(gen_out.video) else gen_out.latents[:, -1]

        return obs, dict()

    def step(self, action):
        device_kwargs = dict(device = self.device)

        self._current_step += 1

        # parse action

        discrete_act, continuous_act = self._parse_action(action)

        # append to action histories

        if exists(discrete_act) and exists(self._discrete_actions):
            self._discrete_actions = torch.cat((self._discrete_actions, discrete_act), dim=1)
        if exists(continuous_act) and exists(self._continuous_actions):
            self._continuous_actions = torch.cat((self._continuous_actions, continuous_act), dim=1)

        time_steps = self._latents.shape[1] + 1
        batch = self._latents.shape[0]

        # generate next frame

        with torch.no_grad():
            gen_out = self.world_model.generate(
                batch_size = batch,
                time_steps = time_steps,
                num_steps = self.num_generation_steps,
                prompt_latents = self._latents,
                prompt_discrete_actions = self._discrete_actions,
                prompt_continuous_actions = self._continuous_actions,
                prompt_rewards = self._rewards,
                image_height = self.image_size,
                image_width = self.image_size,
                return_decoded_video = True,
                return_rewards_per_frame = True,
                return_terminals = True,
                time_cache = self._time_cache,
                use_time_cache = True,
                return_time_cache = True
            )

        if isinstance(gen_out, tuple):
            gen_out, self._time_cache = gen_out

        self._latents = gen_out.latents

        batch = self._latents.shape[0]

        # extract reward

        if exists(gen_out.rewards):
            self._rewards = gen_out.rewards
            reward = gen_out.rewards[:, -1]
        else:
            reward = torch.zeros((batch,), **device_kwargs)

        # extract terminals and truncation

        terminated = gen_out.terminals if exists(gen_out.terminals) else torch.zeros((batch,), dtype=torch.bool, **device_kwargs)
        truncated = tensor([self._current_step >= self.max_steps] * batch, dtype=torch.bool, **device_kwargs)

        # decode observation

        obs = gen_out.video[:, :, -1] if exists(gen_out.video) else gen_out.latents[:, -1]

        return obs, reward, terminated, truncated, dict(experience = gen_out)

    def _parse_action(self, action):

        # parse out discrete and continuous actions

        discrete_action = None
        continuous_action = None

        num_discrete = self.world_model.action_embedder.num_discrete_action_types
        num_continuous = self.world_model.action_embedder.num_continuous_action_types

        device = self.device
        action = tree_map(lambda t: cast_to_tensor(t, device = device), action)

        if num_discrete > 0 and num_continuous > 0:
            discrete_action, continuous_action = action
        elif num_discrete > 0:
            discrete_action = action
        elif num_continuous > 0:
            continuous_action = action

        batch = self._latents.shape[0] if exists(self._latents) else 1

        # helper to format unbatched and batched actions to (b, 1, d)

        def format_action(action):
            if not exists(action):
                return action

            action = torch.atleast_1d(action)

            if action.ndim == 1:
                action = rearrange(action, 'd -> 1 1 d' if batch == 1 else 'b -> b 1 1')
            elif action.ndim == 2:
                action = rearrange(action, 'b d -> b 1 d')
            else:
                raise ValueError(f'action must be 1D or 2D, but got {action.ndim}D')

            assert action.shape[0] == batch, f'action batch size {action.shape[0]} must match environment batch size {batch}'

            return action

        return format_action(discrete_action), format_action(continuous_action)
