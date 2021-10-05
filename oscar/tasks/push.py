# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import numpy as np
import os
import torch
from copy import deepcopy

from oscar import ASSETS_ROOT
from oscar.utils.object_utils import create_cylinder, create_box
from oscar.tasks.agent_task import AgentTask
from isaacgym import gymtorch
from isaacgym import gymapi
from oscar.utils.torch_utils import axisangle2quat, quat2axisangle, rotate_vec_by_quat, to_torch, tensor_clamp, quat_mul


class Push(AgentTask):
    """
    Robot Manipulation task that involves pushing a small puck up an inclined table along a specific path
    """

    def __init__(
        self,
        cfg,
    ):
        # Store relevant task information
        task_cfg = cfg["task"]
        self.aggregate_mode = task_cfg["aggregateMode"]

        # reward info
        self.reward_settings = {
            "r_reach_scale": task_cfg["r_reach_scale"],
            "r_contact_scale": task_cfg["r_contact_scale"],
            "r_path_scale": task_cfg["r_path_scale"],
            "r_goal_scale": task_cfg["r_goal_scale"],
            "r_press_scale": task_cfg["r_press_scale"],
            "n_segments": None,
            "platform_depth": None,
        }

        # reset info
        self.reset_settings = {
            "height_threshold": None,           # Height below which reset will occur
        }

        # Other info from config we need
        self._tmp_asset_path = "/tmp"            # Temporary directory for storing generated cup files
        self.steps_per_vis_update = 1
        self.max_envs_for_vis = 16          # Maximum number of environments allowable for using path segment visualization
        self._platform_depth = 0.2
        self._thickness = 0.05
        self._path_drop = 0.0005
        self._path_width = task_cfg["path_width"]
        self._path_incline = task_cfg["path_incline"]
        self._path_bounds = task_cfg["path_bounds"]
        self._path_segment_bounds = [int(task_cfg["path_bounds"][0] / self._path_width),
                                     int(task_cfg["path_bounds"][1] / self._path_width), ]

        # Determine how many path segments we'll have based on the path type
        self._n_segments = None
        self._path_shape = task_cfg["path_shape"]
        assert self._path_shape in {"straight", "bend1"}, f"Invalid path shape specified: {self._path_shape}"
        if self._path_shape == "straight":
            self._n_segments = self._path_segment_bounds[0]
        else:       # bend1
            self._bend_size = task_cfg.get("path_kwargs", {}).get("bend_size", 2)                   # Hardcoded bend size for now
            self._n_segments = self._path_segment_bounds[0] + self._bend_size
        self.reward_settings["n_segments"] = self._n_segments

        # Placeholders that will be filled in later
        self.cameras = None

        # Private placeholders
        self._start_platform_surface_pos = None       # (x,y,z) position of start platform state
        self._goal_platform_surface_pos = None        # (x,y,z) position of goal platform state
        self._goal_bin_state = None             # (x,y,z) position of goal bin
        self._path_segment_states = None         # States of path segments
        self._puck_state = None              # Root body state
        self._puck_density = None              # Density of each puck in each env (scaled)
        self._puck_friction = None              # Friction of each puck in each env (scaled)
        self._path_friction = None              # Friction of path in each env (scaled)
        self._puck_size = None              # Size of each puck in each env
        self._init_puck_state = None           # Initial state of puck for the current env
        self._agent_id = None                   # Actor ID corresponding to agent for a given env
        self._puck_id = None                 # Actor ID corresponding to puck for a given env
        self._path_segment_ids = None       # Actor IDs corresponding to path segments for a given env
        self._current_path_segment = None   # Integer ID corresponding to the current path segment to be reached for a given env
        self._last_path_segment = None      # Integer ID corresponding to the (potentially outdated) last path segment ot be reached for a given env
        self._use_path_visualization = None # Boolean, whether we're using colored visualization of the path progress or not
        self._color_frac = None             # Color fraction values for each env
        self._default_puck_rb_props = None  # Default properties for puck rigid body

        # Run super init
        super().__init__(cfg=cfg)

    def _create_envs(self):
        # Always run super method for create_envs first
        super()._create_envs()

        # Define bounds for env spacing
        lower = gymapi.Vec3(-self.env_spacing, -self.env_spacing, 0.0)
        upper = gymapi.Vec3(self.env_spacing, self.env_spacing, self.env_spacing)

        # Create platform asset
        platform_opts = gymapi.AssetOptions()
        platform_opts.fix_base_link = True
        platform_asset = self.gym.create_box(
            self.sim, *[self._platform_depth, self._path_bounds[1], self._thickness], platform_opts
        )
        self.reward_settings["platform_depth"] = self._platform_depth

        # Create table stand asset
        table_stand_height = 0.1
        table_stand_offset = -0.1
        table_stand_pos = [-0.60, 0.0, 1.0 + self._thickness / 2 - table_stand_height / 2 + table_stand_offset]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], platform_opts)
        self.reset_settings["height_threshold"] = table_stand_pos[2] - 0.1

        # Create puck asset(s)
        self._puck_density = torch.ones(self.n_envs, 1, device=self.device)
        self._puck_friction = torch.ones(self.n_envs, 1, device=self.device)
        self._puck_size = torch.ones(self.n_envs, 1, device=self.device)
        puck_asset = None

        # Define randomized values
        puck_sizes = np.linspace(self.task_cfg["puck_size"][0], self.task_cfg["puck_size"][1], self.n_envs)

        # Shuffle ordering for each
        np.random.shuffle(puck_sizes)

        # Iterate over all values to fill in per-env physical parameters
        puck_assets = []
        for i, puck_size in enumerate(puck_sizes):
            # Cube asset
            self._puck_size[i] = puck_size
            puck_opts = gymapi.AssetOptions()
            puck_opts.disable_gravity = False
            puck_opts.collapse_fixed_joints = True
            puck_opts.density = 1.0                 # Dummy value, this will be immediately overridden
            # asset = self.gym.create_box(self.sim, *[puck_size, puck_size, 0.3 * puck_size], puck_opts)
            puck_asset_fpath = create_cylinder(
                name=f"puck",
                size=[0.5 * puck_size, 0.4 * puck_size],
                mass=np.pi * ((0.5 * puck_size) ** 2) * 0.4 * puck_size,
                generate_urdf=True,
                unique_urdf_name=False,
                visual_top_site=False,
                from_mesh=False,
                hollow=False,
                asset_root_path=self._tmp_asset_path,
            )
            asset = self.gym.load_asset(self.sim, self._tmp_asset_path, puck_asset_fpath, puck_opts)
            puck_assets.append(asset)

        puck_color = gymapi.Vec3(0.0, 0.7, 0.9)

        # Create path segment asset
        self._path_friction = torch.ones(self.n_envs, 1, device=self.device)
        path_segment_opts = gymapi.AssetOptions()
        path_segment_opts.fix_base_link = True
        path_segment_asset = self.gym.create_box(self.sim, *[self._path_width, self._path_width, self._thickness], path_segment_opts)

        # Define start pose for agent
        agent_start_pose = gymapi.Transform()
        agent_start_pose.p = gymapi.Vec3(-0.55, 0.0, 1.0 + self._thickness / 2 + table_stand_offset)
        agent_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for start platform
        start_platform_pos = np.array([-0.25, 0.0, 1.0])
        start_platform_start_pose = gymapi.Transform()
        start_platform_start_pose.p = gymapi.Vec3(*start_platform_pos)
        start_platform_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._start_platform_surface_pos = np.array(start_platform_pos) + np.array([0, 0, self._thickness / 2])

        # Define start pose for goal platform
        goal_platform_pos = start_platform_pos + self._path_bounds[0] * np.array([np.cos(self._path_incline), 0, np.sin(self._path_incline)]) + np.array([0, 0,  - self._path_drop * (self._n_segments + 2)])
        goal_platform_pos[0] += self._platform_depth
        goal_platform_start_pose = gymapi.Transform()
        goal_platform_start_pose.p = gymapi.Vec3(*goal_platform_pos)
        goal_platform_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._goal_platform_surface_pos = np.array(goal_platform_pos) + np.array([0, 0, self._thickness / 2])

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define dummy start pose (for actors whose positions get overridden during reset() anyways)
        dummy_start_pose = gymapi.Transform()
        dummy_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        dummy_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        self._use_path_visualization = self.n_envs <= self.max_envs_for_vis
        n_agent_bodies = self.gym.get_asset_rigid_body_count(self.agent_asset)
        n_agent_shapes = self.gym.get_asset_rigid_shape_count(self.agent_asset)
        n_puck_bodies = self.gym.get_asset_rigid_body_count(puck_assets[0])
        n_puck_shapes = self.gym.get_asset_rigid_shape_count(puck_assets[0])
        max_agg_bodies = n_agent_bodies + n_puck_bodies + 3 + self._n_segments     # 1 for table, start / goal platform, path segments
        max_agg_shapes = n_agent_shapes + n_puck_shapes + 3 + self._n_segments     # 1 for table, start / goal platform, path segments

        # Determine number of envs to create
        n_per_row = int(np.sqrt(self.n_envs))

        # Create environments
        self._color_frac = np.zeros(self.n_envs)
        self._default_puck_rb_props = {
            "mass": [],
            "invMass": [],
            "inertia": [],
            "invInertia": [],
        }
        for i in range(self.n_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, n_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: Agent should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create agent
            # Potentially randomize start pose
            if self.agent_pos_noise > 0:
                rand_xy = self.agent_pos_noise * (-1. + np.random.rand(2) * 2.0)
                agent_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + self._thickness / 2 + table_stand_height)
            if self.agent_rot_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.agent_rot_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                agent_start_pose.r = gymapi.Quat(*new_quat)
            agent_actor = self.gym.create_actor(env_ptr, self.agent_asset, agent_start_pose, self.agent.name, i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, agent_actor, self.agent_dof_props)

            # Record agent ID if we haven't done so already
            if self._agent_id is None:
                self._agent_id = agent_actor

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create agent stand and platforms
            start_platform_actor = self.gym.create_actor(env_ptr, platform_asset, start_platform_start_pose, "start_platform", i, 1, 0)
            goal_platform_actor = self.gym.create_actor(env_ptr, platform_asset, goal_platform_start_pose, "goal_platform", i, 1, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand", i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create pucks
            self._puck_id = self.gym.create_actor(env_ptr, puck_assets[i], dummy_start_pose, "puck", i, 2, 0)
            # Store default rigid body props
            puck_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, self._puck_id)[0]
            self._default_puck_rb_props["mass"].append(puck_rb_props.mass)
            self._default_puck_rb_props["invMass"].append(puck_rb_props.invMass)
            self._default_puck_rb_props["inertia"].append(puck_rb_props.inertia)
            self._default_puck_rb_props["invInertia"].append(puck_rb_props.invInertia)

            # Create path segments
            self._path_segment_ids = []
            for path_segment_idx in range(self._n_segments):
                path_segment_id = self.gym.create_actor(env_ptr, path_segment_asset, dummy_start_pose, f"path_segment{path_segment_idx}", i, 0, 0)
                self._path_segment_ids.append(path_segment_id)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)

        # Setup init state buffer
        self._init_puck_state = torch.zeros(self.n_envs, 13, device=self.device)

    def setup_references(self):
        # Always run super method first
        super().setup_references()

        # Fill in relevant handles
        task_handles = {
            "puck": self.gym.find_actor_rigid_body_handle(self.envs[0], self._puck_id, "puck"),
        }
        for i in range(self._n_segments):
            task_handles[f"path_segment{i}"] = self.gym.find_actor_rigid_body_handle(self.envs[0], self._path_segment_ids[i], f"path_segment{i}")
        self.handles.update(task_handles)

        # Store tensors to hold states
        self._puck_state = self.sim_states.actor_root_states[:, self._puck_id, :]
        self._path_segment_states = self.sim_states.actor_root_states[:, self._path_segment_ids[0]:self._path_segment_ids[-1]+1, :]

        # Store other necessary tensors
        self._current_path_segment = torch.zeros(self.n_envs, device=self.device, dtype=torch.long)
        self._last_path_segment = torch.zeros_like(self._current_path_segment)

        # Randomize properties
        self._reset_properties()

        # Store references to the contacts
        self.contact_forces.update({
            "puck": self.sim_states.contact_forces[:, self.handles["puck"], :],
        })

    def _compute_rewards(self, actions):
        # Compose dict of contacts
        contacts = {
            "arm": torch.sum(torch.norm(self.contact_forces["arm"], dim=-1), dim=-1),
            "eef": torch.norm(self.contact_forces["eef"], dim=-1),
            "puck": self.contact_forces["puck"],
        }

        # Compute reward (use jit function for speed)
        return _compute_task_rewards(
            actions=actions,
            contacts=contacts,
            states=self.states,
            reward_settings=self.reward_settings,
        )

    def _compute_resets(self):
        # Compute resets (use jit function for speed)
        return _compute_task_resets(
            reset_buf=self.reset_buf,
            progress_buf=self.progress_buf,
            states=self.states,
            max_episode_length=self.max_episode_length,
            reset_settings=self.reset_settings,
        )

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

        # Update path segments
        self._update_path_segments()

        env_ids = torch.arange(start=0, end=self.n_envs, device=self.device, dtype=torch.long)

        # Update internal states
        z_vec = torch.zeros_like(self._puck_state[:, :3])
        z_vec[:, 2] = 1.0
        z_rot = rotate_vec_by_quat(vec=z_vec, quat=self._puck_state[:, 3:7])
        current_target = torch.where(
            self._current_path_segment.unsqueeze(-1) >= self._n_segments,
            self._path_segment_states[env_ids, -1, :3] + torch.tile(torch.tensor([self._path_width, 0, 0], device=self.device, dtype=torch.float32).unsqueeze(0), (self.n_envs, 1)),
            self._path_segment_states[env_ids, self._current_path_segment.clip(max=self._n_segments - 1), :3],
        )

        self.states.update({
            "puck_quat": self._puck_state[:, 3:7].clone(),
            "puck_pos": self._puck_state[:, :3].clone(),
            "puck_tilt": torch.abs(torch.nn.CosineSimilarity(dim=-1)(z_rot, z_vec).unsqueeze(dim=-1)),
            "puck_to_target": current_target - self._puck_state[:, :3],
            "puck_pos_relative": self._puck_state[:, :3] - self.agent.states["eef_pos"],
            "proportion_completed": (self._current_path_segment / self._n_segments).unsqueeze(-1),
            "current_target": current_target,
        })

    def _update_path_segments(self):
        """
        Updates the path segments based on the current state of the environment
        """
        # Check if the puck is close (within 1/2 the width of a given segment) to the current path segment
        # If so, we update the current path segment
        env_ids = torch.arange(start=0, end=self.n_envs, device=self.device, dtype=torch.long)
        xy_close = torch.norm(self._puck_state[:, :2] - self._path_segment_states[env_ids, self._current_path_segment.clip(max=self._n_segments - 1), :2], dim=-1) < (self._path_width - 0.5 * self._puck_size[env_ids].squeeze(-1)) * np.cos(self._path_incline)
        z_close = self._puck_state[:, 2] - self._path_segment_states[env_ids, self._current_path_segment.clip(max=self._n_segments - 1), 2] > 0.5 * self._path_width * np.sin(self._path_incline)
        self._current_path_segment = torch.where(
            xy_close & z_close & (self._current_path_segment < self._n_segments),
            self._current_path_segment + 1,
            self._current_path_segment,
        )

        # If we're visualizing and at the appropriate step, we update the path visuals if necessary
        if self._use_path_visualization:
            should_update = torch.where(
                self.progress_buf % self.steps_per_vis_update == 0,
                torch.ones_like(self.progress_buf),
                torch.zeros_like(self.progress_buf)
            )
            env_ids = should_update.nonzero(as_tuple=False).squeeze(-1)
            if len(env_ids) > 0:
                # Loop through env IDs, and check if current path segment != last path segment
                for env_id in env_ids:
                    if self._current_path_segment[env_id] != self._last_path_segment[env_id]:
                        # Update last path segment, and update color (we assume the marker has reached this point)
                        env = self.envs[env_id]
                        path_segment_id = self._path_segment_ids[self._last_path_segment[env_id]]
                        path_segment_rs_props = self.gym.get_actor_rigid_shape_properties(env, path_segment_id)
                        path_segment_rs_props[0].friction = self._path_friction[env_id]
                        self.gym.set_actor_rigid_shape_properties(env, path_segment_id, path_segment_rs_props)
                        self.gym.set_rigid_body_color(env, path_segment_id, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.7, 1 - self._color_frac[env_id], self._color_frac[env_id]))
                        self._last_path_segment[env_id] = self._current_path_segment[env_id]

    def reset(self, env_ids=None):
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.n_envs, device=self.device, dtype=torch.long)

        # Reset properties
        self._reset_properties(env_ids=env_ids)

        # We must do a reset here to make sure the changes to the rigid body are propagated correctly
        self.sim_states.set_actor_root_states_indexed(env_ids=env_ids)
        self.sim_states.set_dof_states_indexed(env_ids=env_ids)
        self.sim_states.clear_contact_forces_indexed(env_ids=env_ids)
        self.gym.simulate(self.sim)
        self.sim_states.refresh(contact_forces=False)

        # Reset path
        self._reset_paths(env_ids=env_ids)

        # Reset puck (must occur AFTER resetting properties because something weird happens with states being overridden otherwise)
        self._reset_puck_state(env_ids=env_ids)

        # Always run super reset at the end
        super().reset(env_ids=env_ids)

    def _reset_puck_state(self, env_ids=None):
        """
        Simple method to sample @puck's position based on self.startPositionNoise and self.startRotationNoise, and
        automatically reset the pose internally. Populates self._init_puck_state and automatically writes to the
        corresponding self._puck_state

        Args:
            env_ids (tensor or None): Specific environments to reset puck for
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.n_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        n_resets = len(env_ids)
        sampled_puck_state = torch.zeros(n_resets, 13, device=self.device)

        # Sampling is "centered" around front half of table
        centered_puck_xy_state = torch.tensor(
            self._start_platform_surface_pos[:2] + np.array([0.375 * self._platform_depth, 0]),
            device=self.device, dtype=torch.float32)

        # Set z value, which is fixed height
        sampled_puck_state[:, 2] = self._start_platform_surface_pos[2] + self._puck_size[env_ids].squeeze(-1) * 0.4 * 0.5

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_puck_state[:, 6] = 1.0

        # Sample x y values
        noise_scale = self.start_position_noise * torch.tensor([1, 10], dtype=torch.float, device=self.device).unsqueeze(0)
        sampled_puck_state[:, :2] = centered_puck_xy_state + \
            2.0 * noise_scale * (torch.rand(n_resets, 2, dtype=torch.float, device=self.device) - 0.5)

        # Sample rotation value
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(n_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(n_resets, device=self.device) - 0.5)
            sampled_puck_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_puck_state[:, 3:7])

        # Lastly, set these sampled values as the new init state
        self._init_puck_state[env_ids, :] = sampled_puck_state

        # Write to the sim states
        self._puck_state[env_ids] = self._init_puck_state[env_ids]

    def _reset_paths(self, env_ids=None):
        """
        Method to reset paths in specific environments specified by @env_ids.

        Args:
            env_ids (tensor or None): Specific environments to reset path for
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.n_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        n_resets = len(env_ids)
        sampled_path_segment_location_ids_x = torch.zeros(n_resets, self._n_segments, device=self.device)
        sampled_path_segment_location_ids_y = torch.zeros(n_resets, self._n_segments, device=self.device)
        sampled_path_segment_locations = torch.zeros(n_resets, self._n_segments, 3, device=self.device)

        # Handle different cases of paths
        if self._path_shape == "straight":
            raise NotImplementedError("Straight path resets not implemented yet!")
        else:       # bend1
            # Generate path starts, bend locations, and bend directions
            ones = torch.ones(n_resets, device=self.device)
            path_starts = torch.randint(low=0, high=self._path_segment_bounds[1], size=(n_resets,), device=self.device)

            # Sanity clip the path starts so that we only get valid paths
            path_starts = torch.where(
                path_starts < 0.5 * self._path_segment_bounds[1],
                torch.min((ones * int(0.5 * (self._path_segment_bounds[1] - self._bend_size))).int(), path_starts),
                torch.max((ones * int(0.5 * (self._path_segment_bounds[1] + self._bend_size))).int(), path_starts),
            )

            bend_row = torch.randint(low=1, high=self._path_segment_bounds[0]-1, size=(n_resets,), device=self.device)
            bend_direction = torch.randint(low=0, high=2, size=(n_resets,), device=self.device)

            # Modify the bend direction to be +/- 1, and run sanity check to make sure no direction goes past the bounds
            bend_direction = torch.where(bend_direction > 0, ones, -ones)
            bend_direction = torch.where((self._path_segment_bounds[1] - path_starts <= self._bend_size) & (bend_direction > 0), -ones, bend_direction)
            bend_direction = torch.where((path_starts < self._bend_size) & (bend_direction < 0), ones, bend_direction)

            # Generate graphical location IDs and actual cartesian (x,y,z) locations for the path
            for i in range(self._n_segments):
                if i == 0:
                    # This is the start of the path generation, these values are calculated directly
                    x = 0.0
                    y = path_starts[:]
                else:
                    # Calculate how to modify x and y values
                    incrementer = torch.where(
                        (sampled_path_segment_location_ids_x[:, i-1] == bend_row) &
                        (torch.abs(sampled_path_segment_location_ids_y[:, i-1] - path_starts) < self._bend_size),
                        bend_direction, torch.zeros(n_resets, device=self.device)
                    )
                    x = torch.where(
                        incrementer != 0,
                        sampled_path_segment_location_ids_x[:, i-1],
                        sampled_path_segment_location_ids_x[:, i-1]+1,
                    )
                    y = sampled_path_segment_location_ids_y[:, i-1] + incrementer

                # Store these values
                sampled_path_segment_location_ids_x[:, i] = x
                sampled_path_segment_location_ids_y[:, i] = y
                # Generate cartesian (x,y,z) locations
                # We do this procedurally, first by converting x integer value into global value, and then rotating all
                # values (done at once after this loop)
                sampled_path_segment_locations[:, i, 0] = self._path_width / 2.0 + self._path_width * x
                sampled_path_segment_locations[:, i, 1] = (-self._path_bounds[1] + self._path_width) / 2.0 + self._path_width * y
                sampled_path_segment_locations[:, i, 2] = -self._thickness / 2.0 - self._path_drop * (i + 1)

            # Convert xz values into global values
            c, s = np.cos(self._path_incline), np.sin(self._path_incline)
            rot_mat = torch.tensor([[c, -s],[s, c]], device=self.device, dtype=torch.float).unsqueeze(0).unsqueeze(0)
            sampled_path_segment_locations[:, :, [0, 2]] = torch.matmul(rot_mat, sampled_path_segment_locations[:, :, [0, 2]].unsqueeze(-1)).squeeze(-1)

            # Translate the values to the global coordinates
            sampled_path_segment_locations[:, :, 0] += self._start_platform_surface_pos[0] + self._platform_depth / 2.0
            sampled_path_segment_locations[:, :, 2] += self._start_platform_surface_pos[2]

            # Set the rotation values
            segment_quat = torch.zeros_like(sampled_path_segment_locations)
            segment_quat[:, :, 1] = -self._path_incline
            segment_quat = axisangle2quat(segment_quat)

        # Set the sampled locations in sim
        self._path_segment_states[env_ids, :, :3] = sampled_path_segment_locations
        self._path_segment_states[env_ids, :, 3:7] = segment_quat

        # Reset the current segment ID
        self._current_path_segment[env_ids] = 0
        self._last_path_segment[env_ids] = 0

    def _reset_properties(self, env_ids=None):
        """
        Method to reset properties in specific environments specified by @env_ids.

        Args:
            env_ids (tensor or None): Specific environments to reset env properties for
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.n_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        n_resets = len(env_ids)

        # Reset the puck friction and densities
        f_puck_log_min, f_puck_log_max = np.log10(self.task_cfg["puck_friction"][0]), np.log10(self.task_cfg["puck_friction"][1])
        self._puck_friction[env_ids, :] = to_torch(np.power(10., np.random.uniform(
            low=f_puck_log_min,
            high=f_puck_log_max,
            size=(n_resets, 1)
        )), dtype=torch.float, device=self.device)
        d_puck_log_min, d_puck_log_max = np.log10(self.task_cfg["puck_density"][0]), np.log10(self.task_cfg["puck_density"][1])
        self._puck_density[env_ids, :] = to_torch(np.power(10., np.random.uniform(
            low=d_puck_log_min,
            high=d_puck_log_max,
            size=(n_resets, 1)
        )), dtype=torch.float, device=self.device)

        # Reset the path segment frictions
        f_log_min, f_log_max = np.log10(self.task_cfg["path_friction"][0]), np.log10(self.task_cfg["path_friction"][1])
        self._path_friction[env_ids, :] = to_torch(np.power(10., np.random.uniform(
            low=f_log_min,
            high=f_log_max,
            size=(n_resets, 1)
        )), dtype=torch.float, device=self.device)

        # Update in sim
        for env_id in env_ids:
            env = self.envs[env_id]

            # Set color values
            puck_r_frac = (np.log10(self._puck_friction[env_id].item()) - f_puck_log_min) / (f_puck_log_max - f_puck_log_min)
            puck_g_frac = (np.log10(self._puck_density[env_id].item()) - d_puck_log_min) / (d_puck_log_max - d_puck_log_min)
            path_frac = (np.log10(self._path_friction[env_id].item()) - f_log_min) / (f_log_max - f_log_min) if f_log_max != f_log_min else 0.5
            self._color_frac[env_id] = path_frac
            puck_color = [puck_r_frac, puck_g_frac, 0.0] if (d_puck_log_max != d_puck_log_min) and (f_puck_log_max != f_puck_log_min) else [ 0.7, 0.7, 0.0 ]

            # Set puck values
            puck_rs_props = self.gym.get_actor_rigid_shape_properties(env, self._puck_id)
            puck_rb_props = self.gym.get_actor_rigid_body_properties(env, self._puck_id)
            puck_rs_props[0].friction = self._puck_friction[env_id]
            puck_rb_props[0].mass = self._default_puck_rb_props["mass"][env_id] * self._puck_density[env_id].item()
            self.gym.set_actor_rigid_shape_properties(env, self._puck_id, puck_rs_props)
            self.gym.set_actor_rigid_body_properties(env, self._puck_id, puck_rb_props, recomputeInertia=True)
            self.gym.set_rigid_body_color(env, self._puck_id, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*puck_color))

            # Set path values
            for path_segment_id in self._path_segment_ids:
                path_segment_rs_props = self.gym.get_actor_rigid_shape_properties(env, path_segment_id)
                path_segment_rs_props[0].friction = self._path_friction[env_id]
                self.gym.set_actor_rigid_shape_properties(env, path_segment_id, path_segment_rs_props)
                # If we're within the allowable number of environments, we also update the path segment colors
                if self._use_path_visualization:
                    self.gym.set_rigid_body_color(env, path_segment_id, 0, gymapi.MESH_VISUAL,
                                                  gymapi.Vec3(0.0, 1 - path_frac, path_frac))

        # Update extrinsic states
        self.states.update({
            "puck_density": torch.log(self._puck_density),
            "puck_friction": torch.log(self._puck_friction),
            "puck_size": self._puck_size.clone(),
            "path_friction": torch.log(self._path_friction),
        })

    @property
    def force_sim_step_during_reset(self):
        return False


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def _compute_task_rewards(
    actions,
    contacts,
    states,
    reward_settings,
):
    # type: (Tensor, Dict[str, Tensor], Dict[str, Tensor], Dict[str, float]) -> Tensor

    # Compute distance from hand to puck
    d_hand = torch.norm(states["puck_pos_relative"], dim=-1)
    reach_reward = (0.5 * (1 - torch.tanh(10.0 * d_hand)) + 0.5 * (1 - torch.tanh(10.0 * torch.abs(states["puck_pos_relative"][:, 2]))))

    # Reward for making contact with EEF and any object, with no contact of arm
    # As sanity check, also check for EEF being in relative range of puck
    bad_contact = contacts["arm"] > 0
    good_contact = contacts["eef"] > 0.25
    in_range = d_hand < states["puck_size"].squeeze(-1) * 5.0
    contact_reward = good_contact & ~bad_contact & in_range

    # Penalty for pressing too hard
    puck_contact = contacts["puck"]
    press_penalty = torch.abs(puck_contact[:, 2]) > 10.0

    # Path reward is proportion of path completed
    path_reward = states["proportion_completed"].squeeze(-1)

    # Compute distance from puck to target position
    d_target = torch.norm(states["current_target"] - states["puck_pos"], dim=-1)
    target_reward = 1 - torch.tanh(10.0 * d_target)

    # Compute goal reward (height of puck is above goal surface and path is completed and x value is past threshold)
    goal_reward = (states["proportion_completed"].squeeze(-1) == 1.0) & \
                  (states["puck_pos"][:, 2] > states["current_target"][:, 2]) & \
                  (states["puck_pos"][:, 0] > states["current_target"][:, 0])

    # Compose rewards

    # We provide the path success reward + maximum between (target reward + reach reward + contact reward, goal reward)
    rewards = \
        reward_settings["r_path_scale"] * path_reward - \
        reward_settings["r_press_scale"] * press_penalty + \
        torch.max(
            # target_reward * path_reward / reward_settings["n_segments"] +
            0.5 * target_reward * reward_settings["r_path_scale"] / reward_settings["n_segments"] +
            reward_settings["r_reach_scale"] * reach_reward +
            contact_reward * reward_settings["r_contact_scale"],
            goal_reward * reward_settings["r_goal_scale"]
        )

    return rewards


@torch.jit.script
def _compute_task_resets(
    reset_buf,
    progress_buf,
    states,
    max_episode_length,
    reset_settings,
):
    # type: (Tensor, Tensor, Dict[str, Tensor], int, Dict[str, float]) -> Tensor

    reset_buf = torch.where(
        (progress_buf >= max_episode_length - 1),
        torch.ones_like(reset_buf),
        reset_buf
    )

    return reset_buf
