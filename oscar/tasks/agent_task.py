# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os
import torch

from oscar.tasks.base.base_task import BaseTask
from oscar.agents import AGENT_MAPPING, AGENTS
from isaacgym import gymtorch
from isaacgym import gymapi
from collections import deque


class AgentTask(BaseTask):
    def __init__(
        self,
        cfg,
    ):
        # Load agent
        device = cfg["sim"]["device"]
        self.agent_cfg = cfg["agent"]
        agent_type = self.agent_cfg["type"].lower()
        assert agent_type in AGENTS, \
            f"Invalid agent type requested. Valid options are: {AGENTS}, got: {agent_type}"
        self.agent = AGENT_MAPPING[agent_type](device=device, agent_config=self.agent_cfg)

        # Setup placeholders that will be filled in later
        self.handles = {}
        self.control_models = {}
        self.agent_asset = None
        self.agent_pos_noise = cfg["task"]["agent_pos_noise"]
        self.agent_rot_noise = cfg["task"]["agent_rot_noise"]
        self._n_frames_stack = cfg["task"]["n_frames_stack"]   # How many frames to stack for observations
        self._should_update_obs_history = False     # Flag for making sure obs history only gets updated once per sim step
        self._obs_history = None                    # History of obs values (dict of deques)

        # Run super init (this includes setting up the simulation)
        super().__init__(cfg=cfg)

    def _create_envs(self):
        # n per row is automatically the rounded square root of the number of envs
        # Load the agent asset
        self.agent_asset, self.agent_dof_props = self.agent.load_asset(gym=self.gym, sim=self.sim, n_envs=self.n_envs)

    def setup_references(self):
        # Always run super method first
        super().setup_references()

        # Setup agent references
        self.agent.setup_references(sim_states=self.sim_states, sim_actions=self.sim_actions,
                                    env_ptrs=self.envs, actor_handle=0)

        # Add contact forces from agent to this env
        self.contact_forces.update(self.agent.contact_forces)

    def _update_states(self, dt=None):
        """
        Updates the internal states for this task (should update self.states dict)

        NOTE: Assumes simulation has already refreshed states!!

        Args:
            dt (None or float): Amount of sim time (in seconds) that has passed since last update. If None, will assume
                that this is simply a refresh (no sim time has passed) and will not update any values dependent on dt
        """
        # Always run super method first
        super()._update_states(dt=dt)

        # Update agent's states
        self.agent.update_states(dt=dt)

        # May update obs history if we're actually taking a forward step (dt is not None)
        if dt is not None:
            if self._n_frames_stack > 1 and self._obs_history is not None:
                # Obs gets updated at once during get_observation call
                self._should_update_obs_history = True

        # Add agents' states to this task's dict
        self.states.update(self.agent.states)

    def _compute_observations(self):
        """
        Computes observations for the current sim step

        This is the private method that should be extended by subclasses

        Returns:
            list of tensor: (n_env, any_dim) Observation tensor(s) from different sources
        """
        # Collect relevant obs from agent and task
        obs, obs_dict = self.agent.get_observations()

        # Optionally stack frames
        if self._n_frames_stack > 1:
            if self._obs_history is None:
                # Initialize obs history
                self._obs_history = {
                    k: deque([torch.zeros_like(self.states[k]) for _ in
                              range(self._n_frames_stack)], maxlen=self._n_frames_stack)
                    for k in self.obs_keys
                }
            task_obs = []
            for k in self.obs_keys:
                # Update obs history only if the update flag is set
                if self._should_update_obs_history:
                    self._obs_history[k].append(self.states[k])
                task_obs.append(torch.cat(tuple(self._obs_history[k]), dim=-1))

            # Clear update flag
            self._should_update_obs_history = False

        else:
            task_obs = [self.states[k] for k in self.obs_keys]

        obs = [obs] + task_obs

        # Add agent obs dict to current dict
        self.obs_dict.update(obs_dict)

        # Return obs
        return obs

    def reset(self, env_ids=None):
        # Refresh the states before resetting the agent
        # Reset the agent
        self.agent.reset(env_ids=env_ids)

        # Reset the obs history
        if self._n_frames_stack > 1:
            for k in self.obs_keys:
                for i in range(self._n_frames_stack):
                    self._obs_history[k][i][env_ids] = 0.0

        # Always run super reset at the end
        super().reset(env_ids=env_ids)

    def _pre_physics_step(self, actions):
        # Control agent
        self.agent.control(u=actions)
        # Always run super step at the end
        super()._pre_physics_step(actions=actions)

    @property
    def action_modes(self):
        return self.agent.control_modes

    @property
    def action_dim(self):
        return self.agent.action_dim

    @property
    def obs_keys(self):
        """
        String names that correspond to observations that we want to gather during self.get_observations()

        Returns:
            list: List of observation key names to gather when collecting observations
        """
        return self.task_cfg["observations"]

    def register_control_model(self, name, model):
        """
        Registers external control models that can be referenced by this env class

        Args:
            name (str): Name of control model to register
            model (ControlModel): model to register
        """
        self.control_models[name] = model
        # Also register this model with agent
        self.agent.register_control_model(name=name, model=model)
