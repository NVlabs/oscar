# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from gym import spaces

from isaacgym import gymtorch
from oscar.utils.torch_utils import to_torch
import torch
import numpy as np


# VecEnv Wrapper for RL training
class VecTask():
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        self.task = task

        self.num_environments = task.n_envs
        self.num_agents = 1  # used for multi-agent environments
        self.num_observations = task.obs_dim
        self.num_states = task.states_dim
        self.num_actions = task.action_dim

        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf)
        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

        self.clip_obs = clip_observations
        self.clip_actions = clip_actions
        self.rl_device = rl_device

    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_number_of_agents(self):
        return self.num_agents

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations


# Python CPU/GPU Class
class VecTaskPython(VecTask):

    def get_state(self):
        return torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def step(self, actions):
        actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        self.task.step(actions_tensor)

        # Get obs dict mapped to correct device
        obs_dict = self._to_device(self.task.obs_dict)

        # Clamp main obs buf and add it to obs dict
        obs_dict["obs"] = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # Return obs dict
        return obs_dict, self.task.rew_buf.to(self.rl_device), self.task.reset_buf.to(self.rl_device), self.task.extras

    def reset(self):
        # Reset the environment
        self.task.reset()
        actions = 0.01 * (1 - 2 * torch.rand([self.task.n_envs, self.task.action_dim], dtype=torch.float32, device=self.rl_device))

        # step the simulator
        self.task.step(actions)

        # Get obs dict mapped to correct device
        obs_dict = self._to_device(self.task.obs_dict)

        # Clamp main obs buf and add it to obs dict
        obs_dict["obs"] = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # Return obs dict
        return obs_dict

    def _to_device(self, inp):
        """
        Maps all tensors in @inp to this object's device.

        Args:
            inp (tensor, iterable, dict): Any primitive data type that includes tensor(s)

        Returns:
            (tensor, iterable, dict): Same type as @inp, with all tensors mapped to self.rl_device
        """
        # Check all cases
        if isinstance(inp, torch.Tensor):
            inp = inp.to(self.rl_device)
        elif isinstance(inp, dict):
            for k, v in inp.items():
                inp[k] = self._to_device(v)
        else:
            # We assume that this is an iterable, so we loop over all entries
            for i, entry in enumerate(inp):
                inp[i] = self._to_device(entry)
        return inp
