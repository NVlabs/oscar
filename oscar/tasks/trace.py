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
from oscar.utils.path_utils import PATH_MAPPING
from oscar.tasks.agent_task import AgentTask
from oscar.tasks.base.base_task import BaseTask
from oscar.agents.franka_pitcher import FrankaPitcher
from oscar.controllers import OSCController
from oscar.policies.robot_policy import RobotArmPolicy
from isaacgym import gymtorch
from isaacgym import gymapi
from oscar.utils.torch_utils import quat2mat, to_torch


class Trace(AgentTask):
    """
    Robot Manipulation task that involves tracing a parameterized path in free space
    """

    def __init__(
        self,
        cfg,
    ):
        # Store relevant task information
        task_cfg = cfg["task"]
        self.aggregate_mode = task_cfg["aggregateMode"]
        self.use_pre_post_points = task_cfg.get("use_pre_post_points", False)

        # Update task length if using pre post points
        self._start_points, self._end_points = 0, 0
        if self.use_pre_post_points:
            self._start_points = 50
            self._end_points = 50
            task_cfg["episodeLength"] += self._start_points + self._end_points        # 50 for start and end of path

        # reward info
        self.reward_settings = {
            "r_dist_scale": task_cfg["r_dist_scale"],
            "r_ori_scale": task_cfg["r_ori_scale"],
            "r_max": task_cfg["r_ori_scale"],
            "use_balls": None,                      # Gets filled in later, corresponds to whether we're using balls
            "table_height": None,                   # Gets filled in later, corresponds to surface height of table
            "episode_length": task_cfg["episodeLength"],
            "use_pre_post_points": self.use_pre_post_points,
            "start_points": self._start_points,
            "end_points": self._end_points,
        }

        # reset info
        self.reset_settings = {}                    # No specific reset settings for this env
        self.steps_per_path = task_cfg["steps_per_path"]

        # other task info
        self.steps_per_goal_marker = 1
        self.steps_per_eef_marker = 1
        self.max_envs_for_markers = 16          # Maximum number of environments allowable for using path markers
        self.path_type = task_cfg["path_type"]
        assert self.path_type in PATH_MAPPING, f"Invalid path type specified: {self.path_type}"

        # Placeholders that will be filled in later
        self.cameras = None

        # Private placeholders
        self._path = None                       # Path generator
        self._use_balls = None                  # Boolean, whether we're using balls or not
        self._use_path_markers = None           # Boolean, whether we're using path markers or not
        self._n_balls = None                    # Number of balls (if using franka pitcher agent)
        self._n_goal_path_markers = None        # Number of goal path markers (if using markers)
        self._n_eef_path_markers = None         # Number of eef path markers (if using markers)
        self._sample_surface_pos = None         # (x,y,z) location of center of table
        self._ball_states = None                # Root body state of balls
        self._ball_density = None               # ball density
        self._goal_path_marker_states = None    # Root body state of goal path markers
        self._path_marker_states = None         # Root body state of path markers
        self._goal_marker_state = None          # Root body state of visual goal markers
        self._goal_marker_ori_z_state = None    # Root body state of visual goal z ori markers
        self._goal_marker_ori_y_state = None    # Root body state of visual goal y ori markers
        self._agent_id = None                   # Actor ID corresponding to agent for a given env
        self._table_id = None                   # Actor ID corresponding to table for a given env
        self._ball_ids = None                   # Actor ID corresponding to balls for a given env
        self._path_marker_ids = None            # Actor ID corresponding to path markers for a given env
        self._goal_path_marker_ids = None       # Actor ID corresponding to goal path markers for a given env
        self._goal_marker_id = None             # Actor ID corresponding to goal marker for a given env
        self._goal_marker_ori_z_id = None       # Actor ID corresponding to goal z ori marker for a given env
        self._goal_marker_ori_y_id = None       # Actor ID corresponding to goal y ori marker for a given env
        self._goal_pos_min = None
        self._goal_pos_max = None
        self._goal_rot_max = None
        self._goal_pos = None                   # Gets filled in later, desired (x,y,z) eef pos
        self._goal_ori_eef_axis = None          # Gets filled in later, desired (x,y,z) vec representing desired direction of eef z axis
        self._goal_ori_jaw_axis = None          # Gets filled in later, desired (x,y,z) vec representing desired direction of eef y axis (parallel jaw axis)
        self._goal_ori_vec = None               # Cross product between goal y and z axes
        self._reward_offset = None              # Gets filled in later, rewards added as waypoints are reached within a given episode
        self._default_eef_pos = None
        self._default_eef_quat = None

        # Run super init
        super().__init__(cfg=cfg)

    def _create_envs(self):
        # Always run super method for create_envs first
        super()._create_envs()

        # Define bounds for env spacing
        lower = gymapi.Vec3(-self.env_spacing, -self.env_spacing, 0.0)
        upper = gymapi.Vec3(self.env_spacing, self.env_spacing, self.env_spacing)

        # Create table asset
        table_pos = [-0.5, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[0.25, 0.25, table_thickness], table_opts)

        # Create table stand asset
        table_stand_height = 0.4
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

        # Create path / goal marker asset
        marker_opts = gymapi.AssetOptions()
        marker_opts.disable_gravity = True
        marker_asset = self.gym.create_sphere(self.sim, 0.01, marker_opts)
        marker_color = gymapi.Vec3(0.6, 0.1, 0.0)
        marker_ori_z_color = gymapi.Vec3(0.0, 0.6, 0.1)
        marker_ori_y_color = gymapi.Vec3(0.1, 0.0, 0.6)

        # Create path / goal marker asset
        path_marker_opts = gymapi.AssetOptions()
        path_marker_opts.disable_gravity = True
        path_marker_asset = self.gym.create_sphere(self.sim, 0.005, path_marker_opts)
        path_goal_marker_asset = self.gym.create_sphere(self.sim, 0.0025, path_marker_opts)
        path_marker_color = gymapi.Vec3(0.0, 0.7, 0.9)

        # Possibly create ball asset
        self._use_balls = isinstance(self.agent, FrankaPitcher)
        self.reward_settings["use_balls"] = float(self._use_balls)
        if self._use_balls:
            ball_opts = gymapi.AssetOptions()
            assert self.task_cfg["ball_shape"] in {"box", "sphere"}, \
                f"Invalid ball shape requested: {self.task_cfg['ball_shape']}!"
            if self.task_cfg["ball_shape"] == "sphere":
                ball_asset = self.gym.create_sphere(self.sim, self.task_cfg["ball_size"], ball_opts)
            else:           # box
                ball_asset = self.gym.create_box(self.sim, *([self.task_cfg["ball_size"]] * 3), ball_opts)

            ball_color = gymapi.Vec3(0.0, 0.7, 0.9)
            # Set friction to zero for ball asset
            ball_prop = gymapi.RigidShapeProperties()
            self.gym.set_asset_rigid_shape_properties(ball_asset, [ball_prop])

        # Define start pose for agent
        agent_start_pose = gymapi.Transform()
        agent_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        agent_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._sample_surface_pos = torch.tensor(np.array([0., 0., table_pos[2]]) + np.array([0, 0, table_thickness / 2]), device=self.device)

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for marker (doesn't really matter since it gets overridden during reset() anyways)
        marker_start_pose = gymapi.Transform()
        marker_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        marker_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for balls (doesn't really matter since it gets overridden during reset() anyways)
        ball_start_pose = gymapi.Transform()
        ball_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        ball_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        n_agent_bodies = self.gym.get_asset_rigid_body_count(self.agent_asset)
        n_agent_shapes = self.gym.get_asset_rigid_shape_count(self.agent_asset)
        self._use_path_markers = (self.n_envs <= self.max_envs_for_markers) and (self.task_cfg.get("visualize_path", False))
        max_agg_bodies = n_agent_bodies + 5     # 1 for table, table stand, marker (x3)
        max_agg_shapes = n_agent_shapes + 5     # 1 for table, table stand, marker (x3)
        if self._use_balls:
            self._n_balls = self.task_cfg["n_balls"]
            max_agg_bodies += self._n_balls
            max_agg_shapes += self._n_balls
        if self._use_path_markers:
            self._n_goal_path_markers = (self.max_episode_length - self._start_points - self._end_points) // self.steps_per_goal_marker
            self._n_eef_path_markers = (self.max_episode_length - self._start_points - self._end_points) // self.steps_per_eef_marker
            max_agg_bodies += self._n_goal_path_markers + self._n_eef_path_markers
            max_agg_shapes += self._n_goal_path_markers + self._n_eef_path_markers

        # Determine number of envs to create
        n_per_row = int(np.sqrt(self.n_envs))

        # Create environments
        for i in range(self.n_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, n_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: Agent should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create agent
            agent_actor = self.gym.create_actor(env_ptr, self.agent_asset, agent_start_pose, self.agent.name, i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, agent_actor, self.agent_dof_props)

            # Record agent ID if we haven't done so already
            if self._agent_id is None:
                self._agent_id = agent_actor

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            self._table_id = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand", i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Possibly create balls
            if self._use_balls:
                self._ball_ids = []
                for ball_idx in range(self._n_balls):
                    ball_id = self.gym.create_actor(env_ptr, ball_asset, ball_start_pose, f"ball{ball_idx}", i, 0, 0)
                    self._ball_ids.append(ball_id)
                    # Set color
                    self.gym.set_rigid_body_color(env_ptr, ball_id, 0, gymapi.MESH_VISUAL, ball_color)

            # Possibly create path markers
            if self._use_path_markers:
                self._path_marker_ids = []
                self._goal_path_marker_ids = []
                for marker_idx in range(self._n_eef_path_markers):
                    path_marker_id = self.gym.create_actor(env_ptr, path_marker_asset, marker_start_pose, f"path_marker{marker_idx}", self.n_envs + i, 1, 0)
                    self._path_marker_ids.append(path_marker_id)
                    # Set color
                    self.gym.set_rigid_body_color(
                        env_ptr, path_marker_id, 0, gymapi.MESH_VISUAL,
                        gymapi.Vec3(1 - marker_idx / self._n_eef_path_markers, marker_idx / self._n_eef_path_markers, 0.3)
                    )
                for marker_idx in range(self._n_goal_path_markers):
                    goal_path_marker_id = self.gym.create_actor(env_ptr, path_goal_marker_asset, marker_start_pose, f"goal_path_marker{marker_idx}", self.n_envs + i, 1, 0)
                    self._goal_path_marker_ids.append(goal_path_marker_id)

            # Create goal markers
            self._goal_marker_id = self.gym.create_actor(env_ptr, marker_asset, marker_start_pose, "goal_marker", self.n_envs + i, 1, 0)
            self._goal_marker_ori_z_id = self.gym.create_actor(env_ptr, marker_asset, marker_start_pose, "goal_marker_ori_z", self.n_envs + i, 1, 0)
            self._goal_marker_ori_y_id = self.gym.create_actor(env_ptr, marker_asset, marker_start_pose, "goal_marker_ori_y", self.n_envs + i, 1, 0)

            # Set color
            self.gym.set_rigid_body_color(env_ptr, self._goal_marker_id, 0, gymapi.MESH_VISUAL, marker_color)
            self.gym.set_rigid_body_color(env_ptr, self._goal_marker_ori_z_id, 0, gymapi.MESH_VISUAL, marker_ori_z_color)
            self.gym.set_rigid_body_color(env_ptr, self._goal_marker_ori_y_id, 0, gymapi.MESH_VISUAL, marker_ori_y_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)

        # Create path generator
        path_kwargs = {
            "center_quat": (1.0, 0, 0, 0),
            "n_steps": self.steps_per_path,
            "n_paths": self.n_envs,
            "device": self.device,
        }
        path_kwargs.update(self.task_cfg["path_args"])

        self._path = PATH_MAPPING[self.path_type](**path_kwargs)

    def setup_references(self):
        # Always run super method first
        super().setup_references()

        # Fill in relevant handles
        task_handles = {
            "table": self.gym.find_actor_rigid_body_handle(self.envs[0], self._table_id, "box"),
        }

        if self._use_balls:
            self._ball_states = self.sim_states.actor_root_states[:, self._ball_ids[0]:self._ball_ids[-1]+1, :]
            for i in range(self._n_balls):
                task_handles[f"ball{i}"] = self.gym.find_actor_rigid_body_handle(self.envs[0], self._ball_ids[i], f"ball{i}")

            # Register getter function for grabbing ball info to pass to agent
            def ball_getter():
                return {
                    "n_balls": self._n_balls,
                    "ball_size": self.task_cfg["ball_size"],
                    "ball_density": self._ball_density,
                    "ball_states": self._ball_states,
                }
            self.agent.register_balls_state_getter(getter=ball_getter)

        if self._use_path_markers:
            self._path_marker_states = self.sim_states.actor_root_states[:, self._path_marker_ids[0]:self._path_marker_ids[-1]+1, :]
            self._goal_path_marker_states = self.sim_states.actor_root_states[:, self._goal_path_marker_ids[0]:self._goal_path_marker_ids[-1]+1, :]
            for i in range(self._n_eef_path_markers):
                task_handles[f"path_marker{i}"] = self.gym.find_actor_rigid_body_handle(self.envs[0], self._path_marker_ids[i], f"path_marker{i}")
            for i in range(self._n_goal_path_markers):
                task_handles[f"goal_path_marker{i}"] = self.gym.find_actor_rigid_body_handle(self.envs[0], self._goal_path_marker_ids[i], f"goal_path_marker{i}")
        self.handles.update(task_handles)

        # Store tensors to hold marker-related states
        self._goal_marker_state = self.sim_states.actor_root_states[:, self._goal_marker_id, :]
        self._goal_marker_ori_z_state = self.sim_states.actor_root_states[:, self._goal_marker_ori_z_id, :]
        self._goal_marker_ori_y_state = self.sim_states.actor_root_states[:, self._goal_marker_ori_y_id, :]

        # Setup contact references
        self.contact_forces.update({
            "table": self.sim_states.contact_forces[:, self.handles["table"], :],
        })

        # Initialize tensors for goal pos / ori and reward offset
        self._goal_pos = torch.zeros(self.n_envs, 3, device=self.device)
        self._goal_ori_eef_axis = torch.zeros(self.n_envs, 3, device=self.device)
        self._goal_ori_jaw_axis = torch.zeros(self.n_envs, 3, device=self.device)
        self._goal_ori_vec = torch.zeros(self.n_envs, 3, device=self.device)
        self._reward_offset = torch.zeros(self.n_envs, device=self.device)

        # Initialize eef-related tensors
        self._default_eef_pos = torch.tensor([[0.05, 0.05, 1.7]], device=self.device)
        self._default_eef_quat = torch.zeros(self.n_envs, 4, device=self.device)
        self._default_eef_quat[:, 0] = 1.0  # want 180 rotation wrt x axis

        if self._use_balls:
            # We randomize over the physical parameters
            self._ball_density = torch.ones(self.n_envs, 1, device=self.device)
            volume = (4 * np.pi * np.power(self.task_cfg["ball_size"], 3) / 3)
            d_log_min, d_log_max = np.log10(self.task_cfg["ball_density"][0]), np.log10(
                self.task_cfg["ball_density"][1])
            ball_density = np.power(10., np.random.uniform(
                low=d_log_min,
                high=d_log_max,
                size=(self.n_envs,)
            ))

            # Modify the mass for all the balls in each env
            for i, (b_density, env) in enumerate(zip(ball_density, self.envs)):
                self._ball_density[i] = b_density
                frac = (np.log10(b_density) - d_log_min) / (d_log_max - d_log_min)
                for ball_id in self._ball_ids:
                    ball_rb_props = self.gym.get_actor_rigid_body_properties(env, ball_id)
                    ball_rb_props[0].mass = volume * b_density
                    self.gym.set_actor_rigid_body_properties(env, ball_id, ball_rb_props, recomputeInertia=True)
                    self.gym.set_rigid_body_color(env, ball_id, 0, gymapi.MESH_VISUAL, gymapi.Vec3(frac, 0.1, 1 - frac))

            self.states.update({
                "ball_density": torch.log(self._ball_density) / 10.,
            })

        # Pre-fill some states
        self.states.update({
            "goal_pos": self._goal_pos,
            "goal_ori_eef_axis": self._goal_ori_eef_axis,
            "goal_ori_jaw_axis": self._goal_ori_jaw_axis,
            "last_goal_pos": torch.zeros_like(self._goal_pos),
        })

    def _compute_rewards(self, actions):
        # Compose dict of contacts
        contacts = {}

        # Compute reward (use jit function for speed)
        rewards = _compute_task_rewards(
            actions=actions,
            contacts=contacts,
            states=self.states,
            reward_settings=self.reward_settings,
        )

        # Return rewards
        return rewards

    def _compute_resets(self):
        # Compute resets (use jit function for speed)
        return _compute_task_resets(
            reset_buf=self.reset_buf,
            progress_buf=self.progress_buf,
            states=self.states,
            max_episode_length=self.max_episode_length,
            reset_settings=self.reset_settings,
        )

    def _update_path_markers(self):
        """
        Updates the newest path marker to the location of the EEF
        """
        for visualize_error, marker_actor_ids, steps_per_marker, marker_states, current_states in zip(
            (self.task_cfg["visualize_errors"], False),
            (self._path_marker_ids, self._goal_path_marker_ids),
            (self.steps_per_eef_marker, self.steps_per_goal_marker),
            (self._path_marker_states, self._goal_path_marker_states),
            (self.states["eef_base_pos"], self.states["goal_pos"])
        ):
            should_update = torch.where(
                (self.progress_buf % steps_per_marker == 0) & (self.progress_buf >= self._start_points) & (self.progress_buf < self.max_episode_length - self._end_points),
                torch.ones_like(self.progress_buf),
                torch.zeros_like(self.progress_buf)
            )
            env_ids = should_update.nonzero(as_tuple=False).squeeze(-1)
            if len(env_ids) > 0:
                # Grab ID of markers to update
                marker_ids = ((self.progress_buf - self._start_points) // steps_per_marker)[env_ids]
                # Update locations
                marker_states[env_ids, marker_ids, :3] = current_states[env_ids]

                # Optionally set color as well
                if visualize_error:
                    errs = torch.norm(self.states["last_goal_pos"] - self.states["eef_base_pos"], p=2, dim=-1)
                    fracs = (1 - errs * 50.0).clip(min=0.0)
                    for i in range(len(env_ids)):
                        env_id, marker_id, frac = self.envs[i], marker_ids[i], fracs[i].item()
                        self.gym.set_rigid_body_color(env_id, marker_actor_ids[marker_id], 0, gymapi.MESH_VISUAL, gymapi.Vec3(1 - frac, frac, 0))

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

        # Possibly update path markers in sim
        if self._use_path_markers:
            self._update_path_markers()

        # Update goal locations
        if self.use_pre_post_points:
            goal_pos, goal_quat = self._path.generate_pose(idx=torch.clip(self.progress_buf - self._start_points, min=0))

            past_end = self.progress_buf.unsqueeze(-1) >= self.steps_per_path + self._start_points
            goal_pos = torch.where(past_end, self._default_eef_pos, goal_pos)
            goal_quat = torch.where(past_end, self._default_eef_quat, goal_quat)
        else:
            goal_pos, goal_quat = self._path.generate_pose()
        self._goal_pos = goal_pos
        goal_ori = quat2mat(goal_quat)
        self._goal_ori_eef_axis = goal_ori[:, :, 2]
        self._goal_ori_jaw_axis = goal_ori[:, :, 1]
        self._goal_ori_vec = torch.cross(self._goal_ori_jaw_axis, self._goal_ori_eef_axis, dim=-1)

        # Also set the visualization markers in sim
        self._goal_marker_state[:, :3] = self._goal_pos
        self._goal_marker_ori_z_state[:, :3] = self._goal_pos + self._goal_ori_eef_axis * 0.02
        self._goal_marker_ori_y_state[:, :3] = self._goal_pos + self._goal_ori_jaw_axis * 0.02

        # Manually set these in sim states
        self.sim_states.set_actor_root_states_indexed()

        # Update internal states
        self.states.update({
            "last_goal_pos": self.states["goal_pos"],
            "last_goal_ori_eef_axis": self.states["goal_ori_eef_axis"],
            "last_goal_ori_jaw_axis": self.states["goal_ori_jaw_axis"],
            "goal_pos": self._goal_pos,
            "goal_ori_eef_axis": self._goal_ori_eef_axis,
            "goal_ori_jaw_axis": self._goal_ori_jaw_axis,
            "progress_buf": self.progress_buf.clone(),
        })

    def reset(self, env_ids=None):
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.n_envs, device=self.device, dtype=torch.long)

        n_resets = len(env_ids)

        # Reset reward offsets
        self._reward_offset[env_ids] = 0.0

        # Reset paths
        self._path.reset(path_ids=env_ids)

        # Possibly reset path markers
        if self._use_path_markers:
            self._path_marker_states[:, :, :] = 0.0
            self._path_marker_states[:, :, 6] = 1.0           # quat w value
            self._goal_path_marker_states[:, :, :] = 0.0
            self._goal_path_marker_states[:, :, 6] = 1.0           # quat w value

        if self._use_balls:
            #  Re-randomize extrinsic parameters
            volume = (4 * np.pi * np.power(self.task_cfg["ball_size"], 3) / 3)
            d_log_min, d_log_max = np.log10(self.task_cfg["ball_density"][0]), np.log10(self.task_cfg["ball_density"][1])
            self._ball_density[env_ids, :] = to_torch(np.power(10., np.random.uniform(
                low=d_log_min,
                high=d_log_max,
                size=(n_resets, 1)
            )), dtype=torch.float, device=self.device)

            # Modify the mass for all the balls in each env
            for env_id in env_ids:
                env = self.envs[env_id]
                frac = (np.log10(self._ball_density[env_id].item()) - d_log_min) / (d_log_max - d_log_min)
                color = [frac, 0.1, 1 - frac] if d_log_max != d_log_min else [ 0.7, 0.7, 0.0 ]
                for ball_id in self._ball_ids:
                    ball_rb_props = self.gym.get_actor_rigid_body_properties(env, ball_id)
                    ball_rb_props[0].mass = volume * self._ball_density[env_id]
                    self.gym.set_actor_rigid_body_properties(env, ball_id, ball_rb_props, recomputeInertia=True)
                    self.gym.set_rigid_body_color(env, ball_id, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*color))
                    # Also set the pitcher color
                    self.agent.set_pitcher_color(env_id=env_id, color=color)

            # Update extrinsic states
            self.states.update({
                "ball_density": torch.log(self._ball_density) / 10.,
            })

        # Always run super reset at the end
        super().reset(env_ids=env_ids)

    @property
    def force_sim_step_during_reset(self):
        return True

#####################################################################
###=========================jit functions=========================###
#####################################################################

# No jit here because we use CosineSimilarity
def _compute_task_rewards(
    actions,
    contacts,
    states,
    reward_settings,
):
    # type: (Tensor, Dict[str, Tensor], Dict[str, Tensor], Dict[str, float]) -> Tensor

    # distance from hand to the goal pos
    d = torch.norm(states["last_goal_pos"] - states["eef_base_pos"], p=2, dim=-1)
    dist_reward = 1 - torch.tanh(10.0 * d)

    # orientation mismatch of eef axis to goal ori eef axis
    eef_z_axis, eef_y_axis = states["eef_base_z_axis"], states["eef_base_y_axis"]
    c_eef = 0.5 * (1 + torch.nn.CosineSimilarity(dim=-1)(states["last_goal_ori_eef_axis"], eef_z_axis))
    c_jaw = 0.5 * (1 + torch.nn.CosineSimilarity(dim=-1)(states["last_goal_ori_jaw_axis"], eef_y_axis))
    ori_reward = 0.5 * (c_eef + c_jaw)

    rewards = reward_settings["r_ori_scale"] * ori_reward * dist_reward

    # Scale down rewards if using pre post points
    if reward_settings["use_pre_post_points"]:
        rewards = torch.where(
            (states["progress_buf"] < reward_settings["start_points"]) | (states["progress_buf"] >= reward_settings["episode_length"] - reward_settings["end_points"]),
            rewards * 0.10,
            rewards,
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

    # Reset any envs where we've passed the env horizon
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return reset_buf

