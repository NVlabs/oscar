# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os
import torch

from oscar import ASSETS_ROOT
from oscar.tasks.agent_task import AgentTask
from oscar.utils.object_utils import create_hollow_cylinder
from isaacgym import gymtorch
from isaacgym import gymapi
from oscar.utils.torch_utils import quat2axisangle, rotate_vec_by_axisangle, to_torch, axisangle2quat


class Pour(AgentTask):
    """
    Robot Manipulation task that involves pouring a "liquid" (many small spheres) from a pitcher rigidly attahced
    to the agent's end effector into a small, potentially moving, target cup
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
            "r_dist_scale": task_cfg["r_dist_scale"],
            "r_reach_scale": task_cfg["r_reach_scale"],
            "r_tilt_scale": task_cfg["r_tilt_scale"],
            "r_fill_scale": task_cfg["r_fill_scale"],
            "r_miss_scale": task_cfg["r_miss_scale"],
            "pitcher_radius": cfg["agent"]["pitcher_size"][0],
            "ball_radius": task_cfg["ball_size"],
            "episode_length": task_cfg["episodeLength"],
            "metric_rewards": float(task_cfg.get("metric_rewards", False)),
            "table_height": None,                               # Gets filled in later, corresponds to surface height of table
        }

        # reset info
        self.reset_settings = {}                    # No specific reset settings for this env

        # Placeholders that will be filled in later
        self.cameras = None

        # Internal variables
        self.n_balls = task_cfg["n_balls"]
        self.randomize_cup = task_cfg["randomize_cup"]
        self._tmp_asset_path = "/tmp"            # Temporary directory for storing generated cup files
        self._cup_r = None                       # Cup radius in each env
        self._cup_h = None                       # Cup height in each env

        # Private placeholders
        self._table_surface_pos = None          # (x,y,z) location of center of table
        self._cup_state = None                  # Root body state
        self._init_cup_state = None             # Initial body state at start of episode
        self._ball_states = None                # Root body state
        self._ball_density = None               # ball density
        self._agent_id = None                   # Actor ID corresponding to agent for a given env
        self._table_id = None                   # Actor ID corresponding to table for a given env
        self._cup_id = None                     # Actor ID corresponding to cup for a given env
        self._ball_ids = None                   # Actor ID corresponding to balls for a given env

        # Run super init
        super().__init__(cfg=cfg)

        # GPU doesn't work ): So we're forced to use cpu
        assert self.device == 'cpu', "Must use CPU for Pour task since GPU doesn't work ):"

    def _create_envs(self):
        # Always run super method for create_envs first
        super()._create_envs()

        # Define bounds for env spacing
        lower = gymapi.Vec3(-self.env_spacing, -self.env_spacing, 0.0)
        upper = gymapi.Vec3(self.env_spacing, self.env_spacing, self.env_spacing)

        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[0.6, 0.6, table_thickness], table_opts)

        # Create table stand asset
        table_stand_height = 0.25
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

        # Create cup(s)
        cup_r_min, cup_r_max = self.task_cfg["cup_size"][0]
        cup_h_min, cup_h_max = self.task_cfg["cup_size"][1]
        self._cup_r = torch.ones(self.n_envs, 1, device=self.device)
        self._cup_h = torch.ones(self.n_envs, 1, device=self.device)
        cup_asset = None
        n_r_steps = 32
        if self.randomize_cup:
            cup_assets = []
            for i in range(self.n_envs):
                r = cup_r_min + (cup_r_max - cup_r_min) * (i % n_r_steps) / n_r_steps
                h = cup_h_min + (cup_h_max - cup_h_min) * (i // n_r_steps) / n_r_steps
                self._cup_r[i] = r
                self._cup_h[i] = h
                cup_asset_fpath = create_hollow_cylinder(
                    name=f"cup",
                    size=[r, h],
                    thickness=self.task_cfg["cup_thickness"],
                    mass=self.task_cfg["cup_mass"],
                    n_slices=32,
                    shape="round",
                    use_lid=False,
                    transparent_walls=False,
                    generate_urdf=True,
                    unique_urdf_name=False,
                    asset_root_path=self._tmp_asset_path,
                )
                cup_opts = gymapi.AssetOptions()
                cup_opts.collapse_fixed_joints = True
                cup_assets.append(self.gym.load_asset(self.sim, self._tmp_asset_path, cup_asset_fpath, cup_opts))
                if i == 0:
                    # Store reference to first cup
                    cup_asset = cup_assets[0]

        else:
            r = (cup_r_min + cup_r_max) / 2.0
            h = (cup_h_min + cup_h_max) / 2.0
            self._cup_r *= r
            self._cup_h *= h
            cup_asset_fpath = create_hollow_cylinder(
                name="cup",
                size=[r, h],
                thickness=self.task_cfg["cup_thickness"],
                mass=self.task_cfg["cup_mass"],
                n_slices=32,
                shape="round",
                use_lid=False,
                transparent_walls=False,
                generate_urdf=True,
                unique_urdf_name=False,
                asset_root_path=self._tmp_asset_path
            )
            cup_opts = gymapi.AssetOptions()
            cup_opts.collapse_fixed_joints = True
            cup_asset = self.gym.load_asset(self.sim, self._tmp_asset_path, cup_asset_fpath, cup_opts)
        cup_color = gymapi.Vec3(0.9, 0.5, 0.0)

        # Create ball asset
        ball_opts = gymapi.AssetOptions()
        ball_opts.density = 1000. #self.task_cfg["ball_density"]
        ball_asset = self.gym.create_sphere(self.sim, self.task_cfg["ball_size"], ball_opts)
        ball_color = gymapi.Vec3(0.0, 0.7, 0.9)
        # Set friction to zero for ball asset
        ball_prop = gymapi.RigidShapeProperties()
        ball_prop.friction = 0.0
        ball_prop.restitution = 0.5
        self.gym.set_asset_rigid_shape_properties(ball_asset, [ball_prop])

        # Define start pose for agent
        agent_start_pose = gymapi.Transform()
        agent_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        agent_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        # Also set this setting in the reward settings
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for cup (doesn't really matter since it gets overridden during reset() anyways)
        cup_start_pose = gymapi.Transform()
        cup_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        cup_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for balls (doesn't really matter since it gets overridden during reset() anyways)
        ball_start_pose = gymapi.Transform()
        ball_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        ball_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        n_agent_bodies = self.gym.get_asset_rigid_body_count(self.agent_asset)
        n_agent_shapes = self.gym.get_asset_rigid_shape_count(self.agent_asset)
        n_cup_bodies = self.gym.get_asset_rigid_body_count(cup_asset)
        n_cup_shapes = self.gym.get_asset_rigid_shape_count(cup_asset)
        max_agg_bodies = n_agent_bodies + n_cup_bodies + 2 + self.n_balls     # 1 for table, table stand
        max_agg_shapes = n_agent_shapes + n_cup_shapes + 2 + self.n_balls     # 1 for table, table stand

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
            # Potentially randomize start pose
            if self.agent_pos_noise > 0:
                rand_xy = self.agent_pos_noise * (-1. + np.random.rand(2) * 2.0)
                agent_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2 + table_stand_height)
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

            # Create table
            self._table_id = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand", i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create cup
            if self.randomize_cup:
                cup_asset = cup_assets[i]
            self._cup_id = self.gym.create_actor(env_ptr, cup_asset, cup_start_pose, "cup", i, 2, 0)
            # Set color
            for rb_idx in range(n_cup_bodies):
                self.gym.set_rigid_body_color(env_ptr, self._cup_id, rb_idx, gymapi.MESH_VISUAL, cup_color)

            # Create balls
            self._ball_ids = []
            for ball_idx in range(self.n_balls):
                ball_id = self.gym.create_actor(env_ptr, ball_asset, ball_start_pose, f"ball{ball_idx}", i, 0, 0)
                self._ball_ids.append(ball_id)
                # Set color
                self.gym.set_rigid_body_color(env_ptr, ball_id, 0, gymapi.MESH_VISUAL, ball_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)

        # Setup init state buffer
        self._init_cup_state = torch.zeros(self.n_envs, 13, device=self.device)

    def setup_references(self):
        # Always run super method first
        super().setup_references()

        # Fill in relevant handles
        task_handles = {
            "table": self.gym.find_actor_rigid_body_handle(self.envs[0], self._table_id, "box"),
            "cup": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cup_id, "cup_base"),
        }

        self.handles.update(task_handles)

        # Store tensors to hold cup-related state
        self._cup_state = self.sim_states.actor_root_states[:, self._cup_id, :]
        self._ball_states = self.sim_states.actor_root_states[:, self._ball_ids[0]:self._ball_ids[-1]+1, :]

        # Setup contact references
        self.contact_forces.update({
            "table": self.sim_states.contact_forces[:, self.handles["table"], :],
            "cup": self.sim_states.contact_forces[:, self.handles["cup"], :],
        })

        # Register getter function for grabbing ball info to pass to agent
        def ball_getter():
            return {
                "n_balls": self.n_balls,
                "ball_size": self.task_cfg["ball_size"],
                "ball_density": self._ball_density,
                "ball_states": self._ball_states,
            }
        self.agent.register_balls_state_getter(getter=ball_getter)

        # We randomize over the physical parameters
        self._ball_density = torch.ones(self.n_envs, 1, device=self.device)
        volume = (4 * np.pi * np.power(self.task_cfg["ball_size"], 3) / 3)
        d_log_min, d_log_max = np.log10(self.task_cfg["ball_density"][0]), np.log10(self.task_cfg["ball_density"][1])
        ball_density = np.power(10., np.random.uniform(
            low=d_log_min,
            high=d_log_max,
            size=(self.n_envs,)
        ))

        # Modify the mass for all the balls in each env
        for i, (b_density, env) in enumerate(zip(ball_density, self.envs)):
            self._ball_density[i] = b_density
            frac = (np.log10(b_density) - d_log_min) / (d_log_max - d_log_min)
            color = [frac, 0.1, 1 - frac] if d_log_max != d_log_min else [ 0.7, 0.7, 0.0 ]
            for ball_id in self._ball_ids:
                ball_rb_props = self.gym.get_actor_rigid_body_properties(env, ball_id)
                ball_rb_props[0].mass = volume * b_density
                self.gym.set_actor_rigid_body_properties(env, ball_id, ball_rb_props, recomputeInertia=True)
                self.gym.set_rigid_body_color(env, ball_id, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*color))

        # Store states that are static
        self.states.update({
            "ball_density": torch.log(self._ball_density) / 10.,
        })

    def _compute_rewards(self, actions):
        # Compose dict of contacts
        contacts = {
            "arm": torch.sum(torch.norm(self.contact_forces["arm"], dim=-1), dim=-1),
            "cup": torch.norm(self.contact_forces["cup"], dim=-1),
            "table": torch.norm(self.contact_forces["table"], dim=-1),
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

        # Update internal states
        z_vec = torch.zeros_like(self._cup_state[:, :3])
        z_vec[:, 2] = 1.0
        self.states.update({
            "cup_quat": self._cup_state[:, 3:7].clone(),
            "cup_pos": self._cup_state[:, :3].clone(),
            "cup_tilt": torch.nn.CosineSimilarity(dim=-1)(rotate_vec_by_axisangle(z_vec, quat2axisangle(self._cup_state[:, 3:7])), z_vec).unsqueeze(dim=-1),
            "cup_pos_relative": self._cup_state[:, :3] - self.agent.states["eef_pos"],
            "cup_radius": self._cup_r.clone(),
            "cup_height": self._cup_h.clone(),
            "balls_pos": self._ball_states[:, :, :3].clone(),
            "progress_buf": self.progress_buf.clone().unsqueeze(-1),
        })

        # Add fractions of balls in and out of cup
        fill_frac, miss_frac = _compute_ball_proportions(
            balls_pos=self.states["balls_pos"],
            cup_pos=self.states["cup_pos"],
            cup_radius=self.states["cup_radius"],
            cup_height=self.states["cup_height"],
            ball_radius=self.task_cfg["ball_size"],
        )
        self.states.update({
            "fill_frac": fill_frac.unsqueeze(dim=-1),
            "miss_frac": miss_frac.unsqueeze(dim=-1),
        })

    def reset(self, env_ids=None):
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.n_envs, device=self.device, dtype=torch.long)

        n_resets = len(env_ids)

        # Reset cup
        self._reset_cup_state(env_ids=env_ids)

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

        # Update extrinsic states
        self.states.update({
            "ball_density": torch.log(self._ball_density) / 10.,
        })

        # Always run super reset at the end
        super().reset(env_ids=env_ids)

    def _reset_cup_state(self, env_ids=None):
        """
        Simple method to sample @cup's position based on self.startPositionNoise and self.startRotationNoise, and
        automaticlly reset the pose internally. Populates self._init_cup_state and automatically writes to the
        corresponding self._cup_state

        Args:
            env_ids (tensor or None): Specific environments to reset cup for
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.n_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cup_state = torch.zeros(num_resets, 13, device=self.device)

        # Sampling is "centered" around middle of table
        centered_cup_xy_state = torch.tensor(
            self._table_surface_pos[:2] + np.array(self.task_cfg["xy_offset"]), device=self.device, dtype=torch.float32
        )

        # Set z value, which is fixed height
        sampled_cup_state[:, 2] = self._table_surface_pos[2] + self.task_cfg["cup_thickness"] / 2

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_cup_state[:, 6] = 1.0

        # Sample x y values
        sampled_cup_state[:, :2] = centered_cup_xy_state + \
            2.0 * self.start_position_noise * (torch.rand(num_resets, 2, dtype=torch.float, device=self.device) - 0.5)

        # Lastly, set these sampled values as the new init state
        self._init_cup_state[env_ids, :] = sampled_cup_state

        # Write to the sim states
        self._cup_state[env_ids] = self._init_cup_state[env_ids]

    @property
    def force_sim_step_during_reset(self):
        return True


#####################################################################
###=========================jit functions=========================###
#####################################################################


# @torch.jit.script
def _compute_task_rewards(
    actions,
    contacts,
    states,
    reward_settings,
):
    # type: (Tensor, Dict[str, Tensor], Dict[str, Tensor], Dict[str, float]) -> Tensor

    # x y distance from cup to balls
    dist = torch.norm(states["balls_pos"][:, :, :2] - states["cup_pos"][:, :2].unsqueeze(dim=1), dim=-1)

    # Reward for getting balls close to cup
    dist_reward = torch.mean(1 - torch.tanh(5.0 * dist), dim=-1)

    # Reward for getting balls close to correct height (directly above cup)
    height = states["balls_pos"][:, :, 2] - states["cup_pos"][:, 2].unsqueeze(dim=1) - states["cup_height"]
    reach_reward = torch.mean(1 - torch.tanh(2.0 * torch.clip(height, 0.0, 10.)), dim=-1)

    # Reward for tilting pitcher
    tilt_reward = torch.clip(1 - states["pitcher_tilt"].squeeze(-1), 0.0, 1.0)

    # Reward for getting balls IN cup
    fill_reward = states["fill_frac"].squeeze(dim=-1)

    # Penalty for MISSING cup
    miss_penalty = states["miss_frac"].squeeze(dim=-1)

    # Check for whether cup is upright or not
    cup_up = states["cup_tilt"].squeeze(dim=-1) > 0.95

    # Compose rewards
    if reward_settings["metric_rewards"] > 0:
        rewards = torch.where(
            states["progress_buf"] == reward_settings["episode_length"] - 1,
            states["fill_frac"],
            torch.zeros_like(states["fill_frac"]),
        ).squeeze(-1)
    else:
        rewards = \
            reward_settings["r_dist_scale"] * dist_reward * (1 - fill_reward) * (1 - miss_penalty) * cup_up + \
            reward_settings["r_reach_scale"] * reach_reward + \
            reward_settings["r_tilt_scale"] * tilt_reward + \
            reward_settings["r_fill_scale"] * fill_reward * cup_up - \
            reward_settings["r_miss_scale"] * miss_penalty

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


@torch.jit.script
def _compute_ball_proportions(
    balls_pos,
    cup_pos,
    cup_radius,
    cup_height,
    ball_radius,
):
    # type: (Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # Helper function that returns proportion of balls that are in the cup or missed the cup
    # Returns two tuple: fill_frac, miss_frac

    # x y distance from cup to balls
    dist = torch.norm(balls_pos[:, :, :2] - cup_pos[:, :2].unsqueeze(dim=1), dim=-1)

    in_radius = dist < cup_radius
    relative_height = balls_pos[:, :, 2] - cup_pos[:, 2].unsqueeze(dim=1)
    below_height = relative_height < cup_height
    in_height = ((relative_height - ball_radius) > 0.0) & below_height
    fill_frac = torch.mean((in_radius & in_height).float(), dim=-1)

    # Penalty for MISSING cup
    miss_frac = torch.mean((~in_radius & below_height).float(), dim=-1)

    return fill_frac, miss_frac
