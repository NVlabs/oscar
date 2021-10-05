# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from isaacgym import gymapi
from isaacgym import gymtorch
import torch
from .agent import Agent
from oscar import ASSETS_ROOT
from oscar.controllers import *
from oscar.utils.object_utils import *
from oscar.utils.torch_utils import to_torch, tensor_clamp
import os
from copy import deepcopy
from collections import deque


class FrankaPitcher(Agent):
    """
    7-DOF robot manipulator agent that is equipped with an empty pitcher.

    Args:
        device (str or int): Which device to send tensors to
        agent_config (None or dict): Configuration to use for agent. If None,
            a default one will be used. Otherwise, will update the default with any keys specified in this dict.
    """
    def __init__(
        self,
        device,
        agent_config=None,
    ):
        # Setup internal vars
        self._dof = None                    # Filled during load()
        self.dof_total = None              # Filled during load()
        self.dof_props = None               # Filled during load()
        self.dof_lower_limits = None        # Filled during load()
        self.dof_upper_limits = None        # Filled during load()
        self.dof_range = None               # Filled during load()
        self.dof_middle = None              # Filled during load()
        self.dof_default = None             # Filled during load()
        self.dof_arm_mode = None            # One of gymapi.DOF_MODE_XXX
        self.vel_limits = None
        self.effort_limits = None

        # Tensor placeholders
        self._dof_friction = None           # dof friction
        self._dof_damping = None            # dof damping
        self._dof_armature = None           # dof armature
        self._default_body_inertia = None   # Default rigid body inertias
        self._min_body_inertia = None       # Minimum diagonal inertia for rigid bodies
        self._pitcher_base_state = None     # Base of pitcher
        self._pitcher_wall_state = None     # Wall of pitcher
        self._eef_state = None                # end effector state
        self._eef_base_state = None         # base end effector state (shared amongst all franka variants)
        self._eef_base_y_state = None       # y-axis of base end effector state
        self._eef_base_z_state = None       # z-axis of base end effector state
        self._j_eef = None                  # Jacobian for end effector
        self._mm = None                     # Mass matrix
        self._arm_control = None            # Tensor buffer for controlling arm
        self._arm_control_history = None    # History of arm control values (deque)
        self._link_mass = None              # Link masses
        self._link_com = None               # Link COM relative to local frame
        self._j_link = None                 # Link jacobians

        # Run super init
        super().__init__(
            device=device,
            agent_config=agent_config,
        )

    def load_asset(self, gym, sim, n_envs):
        """
        Loads Franka into the simulation, and also sets up the controller

        Args:
            gym (Gym): Active gym instance
            sim (Sim): Active sim instance
            n_envs (int): Number of environments in simulation

        Returns:
            2-tuple:
                Asset: Processed asset representing this agent
                dof_properties: DOF Properties for this agent
        """
        # Save gym, sim, and n_env references
        self.gym = gym
        self.sim = sim
        self.n_envs = n_envs

        # Procedurally generate relevant franka file
        asset_file = self._generate_urdf()

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False       # This is handled manually later
        # asset_options.armature = 0.01
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset = self.gym.load_asset(self.sim, ASSETS_ROOT, asset_file, asset_options)

        # Store dof
        self.dof_total = self.gym.get_asset_dof_count(asset)
        self._dof = 7       # Hardcoded since we shouldn't include dof of kendama string

        # Grab dof properties
        dof_props = self.gym.get_asset_dof_properties(asset)

        # Store the limits
        self.dof_lower_limits = to_torch(dof_props['lower'][:self.dof], device=self.device)
        self.dof_upper_limits = to_torch(dof_props['upper'][:self.dof], device=self.device)
        self.dof_range = self.dof_upper_limits - self.dof_lower_limits
        self.dof_middle = (self.dof_upper_limits + self.dof_lower_limits) / 2.0

        # Setup agent config
        if self.agent_config is None:
            # We use the default config
            self.agent_config = self.default_agent_config
        else:
            # We want to modify the default agent config with the current agent config
            agent_config = self.default_agent_config
            agent_config.update(self.agent_config)
            self.agent_config = agent_config

        # Save default values and map arrays to torch tensor
        self.dof_arm_mode = self.agent_config["dof_arm_mode"]
        self.dof_default = to_torch(self.agent_config["dof_default"], device=self.device)

        # Modify the franka properties so it aligns with the controller and specified values
        for i in range(self.dof):
            # Set drive mode for each component -- directly control joint torques for the arm itself
            # Arm properties
            dof_props['driveMode'][i] = self.dof_arm_mode
            # Set stiffness and damping
            dof_props['stiffness'][i] = self.agent_config["dof_stiffness"][i]
            # Set limits
            dof_props['velocity'][i] = self.agent_config["dof_max_velocities"][i]
            dof_props['effort'][i] = self.agent_config["dof_max_efforts"][i]

        # Get limits
        self.vel_limits = to_torch(dof_props['velocity'][:self.dof], device=self.device)
        self.effort_limits = to_torch(dof_props['effort'][:self.dof], device=self.device)

        # Store dof props
        self.dof_props = dof_props

        # Save and return this asset
        return asset, dof_props

    def setup_references(self, sim_states, sim_actions, env_ptrs, actor_handle=0):
        """
        Sets up relevant references to agent-specific handles in sim

        Args:
            sim_states (SimStates): States object reference from which we'll grab relevant slices

            sim_actions (SimActions): Actions object reference from which we'll grab relevant slices

            env_ptrs (list of Env): Pointer references to the environment with corresponding @actor_handle representing
                this agent

            actor_handle (int): Handle reference to this agent in sim. By default, we assume this agent is the first
                actor loaded
        """
        # Run super first
        super().setup_references(sim_states=sim_states, sim_actions=sim_actions,
                                 env_ptrs=env_ptrs, actor_handle=actor_handle)

        # We assume the first env is representative of all envs
        env_ptr = env_ptrs[0]

        # Overwrite joint values, since we don't care about the string values
        # dof states
        self._dof_state = sim_states.dof_states[:, :self.dof_total]
        self._q = self._dof_state[:, :, 0]
        self._qd = self._dof_state[:, :, 1]
        self._qd_last = self._qd.clone()
        self._qdd = torch.zeros_like(self._qd_last, device=self.device)

        # Get direct references to arm actions
        if self.dof_arm_mode == gymapi.DOF_MODE_POS:
            self._arm_control = self._pos_control[:, :self.agent_config["dof_arm"]]
        elif self.dof_arm_mode == gymapi.DOF_MODE_VEL:
            self._arm_control = self._vel_control[:, :self.agent_config["dof_arm"]]
        elif self.dof_arm_mode == gymapi.DOF_MODE_EFFORT:
            self._arm_control = self._effort_control[:, :self.agent_config["dof_arm"]]
        else:
            raise ValueError(f"Invalid dof mode specified for arm, got: {self.dof_arm_mode}")

        # Setup handles
        # NOTE: We assume the agent is the first thing loaded, so the reference should be unified throughout all
        # environment instances (so we just use @actor_handle and assume they are representative of all
        # instances in sim)
        self.handles.update({
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, "pitcher_wall"),
            "pitcher_base": self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, "pitcher_base"),
            "pitcher_wall": self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, "pitcher_wall"),
            "eef_base": self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, "panda_hand"),
            "eef_base_y": self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, "panda_hand_y_axis"),
            "eef_base_z": self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, "panda_hand_z_axis"),
        })

        # Setup tensor buffers
        self._pitcher_base_state = self._rigid_body_state[:, self.handles["pitcher_base"], :]
        self._pitcher_wall_state = self._rigid_body_state[:, self.handles["pitcher_wall"], :]
        self._eef_state = self._rigid_body_state[:, self.handles["hand"], :]
        self._eef_base_state = self._rigid_body_state[:, self.handles["eef_base"], :]
        self._eef_base_y_state = self._rigid_body_state[:, self.handles["eef_base_y"], :]
        self._eef_base_z_state = self._rigid_body_state[:, self.handles["eef_base_z"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, self.name)
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, actor_handle)['pitcher_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :self.agent_config["dof_arm"]]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, self.name)
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :self.agent_config["dof_arm"], :self.agent_config["dof_arm"]]

        # Link jacobians
        self._j_link = jacobian[:, :self.agent_config["dof_arm"]+1, :, :self.agent_config["dof_arm"]]

        # optionally history of control
        history_length = self.agent_config["history_length"]
        if history_length > 0:
            self._arm_control_history = deque([torch.zeros(self.n_envs, self.dof, device=self.device).type_as(self._q) for _ in range(history_length)], maxlen=history_length)

        # Initialize list to store default body inertias
        self._default_body_inertia = []

        # We randomize over the physical parameters (occurs during reset, so we don't do anything now)
        self._dof_damping = torch.ones(self.n_envs, self._dof, device=self.device)
        self._dof_friction = torch.ones(self.n_envs, self._dof, device=self.device)
        self._dof_armature = torch.ones(self.n_envs, self._dof, device=self.device)
        self._min_body_inertia = torch.ones(self.n_envs, 1, device=self.device)

        # Sample values for damping and friction
        damping = np.power(10., np.random.uniform(
            low=np.log10(self.agent_config["dof_damping_min"]),
            high=np.log10(self.agent_config["dof_damping_max"]),
            size=(self.n_envs, len(self.agent_config["dof_damping_min"]))
        ))
        friction = np.power(10., np.random.uniform(
            low=np.log10(self.agent_config["dof_friction_min"]),
            high=np.log10(self.agent_config["dof_friction_max"]),
            size=(self.n_envs, len(self.agent_config["dof_friction_min"]))
        ))
        armature = np.power(10., np.random.uniform(
            low=np.log10(self.agent_config["dof_armature_min"]),
            high=np.log10(self.agent_config["dof_armature_max"]),
            size=(self.n_envs, len(self.agent_config["dof_armature_min"]))
        ))
        min_body_inertia = np.power(10., np.random.uniform(
            low=np.log10(self.agent_config["min_body_inertia"][0]),
            high=np.log10(self.agent_config["min_body_inertia"][1]),
            size=(self.n_envs,)
        ))

        # We disable gravity for all components
        self._link_mass = torch.zeros(self.n_envs, self.agent_config["dof_arm"]+1, dtype=torch.float, device=self.device)
        self._link_com = torch.zeros(self.n_envs, self.agent_config["dof_arm"]+1, 3, dtype=torch.float, device=self.device)
        rigid_body_names = self.gym.get_actor_rigid_body_names(env_ptrs[0], 0)
        dof_names = self.gym.get_actor_dof_names(env_ptrs[0], 0)
        for i, env in enumerate(env_ptrs):
            rigid_body_props = self.gym.get_actor_rigid_body_properties(env, 0)
            default_inertias = []
            for j, (name, prop) in enumerate(zip(rigid_body_names, rigid_body_props)):
                # Add minimum inertia values
                default_inertias.append(deepcopy(prop.inertia))
                prop.inertia.x = prop.inertia.x + gymapi.Vec3(min_body_inertia[i], 0, 0)
                prop.inertia.y = prop.inertia.x + gymapi.Vec3(0, min_body_inertia[i], 0)
                prop.inertia.z = prop.inertia.x + gymapi.Vec3(0, 0, min_body_inertia[i])
                # Disable gravity if requested
                if self.agent_config['disable_gravity']:
                    prop.flags = 1  # 0th bit corresponds to gravity disabling
                # Grab the mass and add it to the link mass tensor
                if j < self.agent_config["dof_arm"] + 1:
                    self._link_mass[i, j] = prop.mass
                    self._link_com[i, j, 0] = prop.com.x
                    self._link_com[i, j, 1] = prop.com.y
                    self._link_com[i, j, 2] = prop.com.z
                # Optionally set the pitcher color
                if self.agent_config["pitcher_color"] is not None:
                    if "pitcher" in name:
                        self.gym.set_rigid_body_color(env, 0, j, gymapi.MESH_VISUAL, gymapi.Vec3(*self.agent_config["pitcher_color"]))

            # Set the updated props
            self.gym.set_actor_rigid_body_properties(env, 0, rigid_body_props, recomputeInertia=False)

            # Modify joint properties next
            dof_props = self.gym.get_actor_dof_properties(env, 0)
            for j, (name, prop) in enumerate(zip(dof_names, dof_props)):
                # Randomize dof params
                dof_props['damping'][j] = damping[i][j]
                dof_props['friction'][j] = friction[i][j]
                dof_props['armature'][j] = armature[i][j]
            # Set the updated props
            self.gym.set_actor_dof_properties(env, 0, dof_props)

            # Store the friction and damping values
            self._dof_damping[i] = to_torch(damping[i], device=self.device)
            self._dof_friction[i] = to_torch(friction[i], device=self.device)
            self._dof_armature[i] = to_torch(armature[i], device=self.device)
            self._min_body_inertia[i, :] = min_body_inertia[i]
            # Store the default inertia values
            self._default_body_inertia.append(default_inertias)

        # Store references to the contacts
        self.contact_forces = {
            "arm": sim_states.contact_forces[:, :self.handles["hand"], :],
            "pitcher_base": sim_states.contact_forces[:, self.handles["pitcher_base"], :],
            "pitcher_wall": sim_states.contact_forces[:, self.handles["pitcher_wall"], :],
        }

        # Store states that are static / need to be initialized early
        self.states.update({
            "eef_mass": torch.zeros_like(self._min_body_inertia),       # dummy value initially
            "dof_friction": torch.log(self._dof_friction) / 10.,
            "dof_damping": torch.log(self._dof_damping) / 10.,
            "dof_armature": torch.log(self._dof_armature) / 10.,
            "min_body_inertia": torch.log(self._min_body_inertia) / 10.,
            "pitcher_wall_pos": self._pitcher_wall_state[:, :3].clone(),
        })

    def control(self, u):
        """
        Controls this robot for a single timestep in sim. This method should deploy the outputted controller
        actions in sim.

        Args:
            u (None or tensor): Controls to execute in sim
        """
        # The arm action is all of the controls up to the final index, also clip actions
        u_arm = u

        # Denormalize arm action if requested
        if self.agent_config["denormalize_control"]:
            # Scale action according
            if self.dof_arm_mode == gymapi.DOF_MODE_POS:
                u_arm = self.dof_middle[:self.agent_config["dof_arm"]].unsqueeze(0) + \
                        self.dof_range[:self.agent_config["dof_arm"]].unsqueeze(0) * u_arm * 0.5
                low, high = self.dof_lower_limits.unsqueeze(0), self.dof_upper_limits.unsqueeze(0)
            elif self.dof_arm_mode == gymapi.DOF_MODE_VEL:
                u_arm = self.vel_limits[:self.agent_config["dof_arm"]].unsqueeze(0) * u_arm
                low, high = -self.vel_limits.unsqueeze(0), self.vel_limits.unsqueeze(0)
            elif self.dof_arm_mode == gymapi.DOF_MODE_EFFORT:
                u_arm = self.effort_limits[:self.agent_config["dof_arm"]].unsqueeze(0) * u_arm
                low, high = -self.effort_limits.unsqueeze(0), self.effort_limits.unsqueeze(0)
            else:
                raise ValueError(f"Invalid dof mode specified for arm, got: {self.dof_arm_mode}")

        # Optionally use gravity compensation
        if (not self.agent_config['disable_gravity']) and \
            self.agent_config["use_gravity_compensation"] and \
            (self.dof_arm_mode == gymapi.DOF_MODE_EFFORT):
            u_arm = u_arm + self.calculate_gravity_torques(normalize=False)

        # Clip control
        u_arm = tensor_clamp(u_arm, low, high)

        # Write these commands to the appropriate tensor buffers
        self._arm_control[:, :] = u_arm

    def update_states(self, dt=None):
        """
        Updates the internal states for this agent

        NOTE: Assumes simulation has already refreshed states!!

        Args:
            dt (None or float): Amount of sim time (in seconds) that has passed since last update. If None, will assume
                that this is simply a refresh (no sim time has passed) and will not update any values dependent on dt
        """
        # Run super first
        super().update_states(dt=dt)

        # We always update jacobian and mass matrix tensors since they're only relevant to agents
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Calculate pitcher cosine similarity
        pitcher_vec = self._pitcher_wall_state[:, :3] - self._pitcher_base_state[:, :3]
        up_vec = torch.zeros_like(pitcher_vec)
        up_vec[:, 2] = 1.0
        pitcher_tilt = torch.nn.CosineSimilarity(dim=-1)(pitcher_vec, up_vec).unsqueeze(-1)

        # Grab ball-related states
        ball_info = self._balls_getter()
        volume = 4 * np.pi * np.power(ball_info["ball_size"], 3) / 3
        ball_mass = ball_info["ball_density"] * volume
        proportion_in_pitcher = self.get_proportion_in_pitcher().unsqueeze(dim=-1)

        # Modify arm history if we're actually taking a forward step (dt is not None)
        if dt is not None and self._arm_control_history is not None:
            self._arm_control_history.append(self._arm_control.clone())
            self.states.update({
                "arm_control_history": torch.stack(tuple(self._arm_control_history), dim=1)
            })

        # Update internal states
        self.states.update({
            # Remove non-relevant q states
            "q": self.states["q"][:, :self.dof],
            "qd": torch.clip(self.states["qd"][:, :self.dof], -1000., 1000.),
            "qdd": torch.clip(self.states["qdd"][:, :self.dof], -10000., 10000.),
            # Pitcher relevant states
            "pitcher_base_pos": self._pitcher_base_state[:, :3].clone(),
            "pitcher_base_quat": self._pitcher_base_state[:, 3:7].clone(),
            "pitcher_wall_pos": self._pitcher_wall_state[:, :3].clone(),
            "pitcher_wall_quat": self._pitcher_wall_state[:, 3:7].clone(),
            "pitcher_tilt": pitcher_tilt,
            # Ball relevant states
            "ball_proportion_in_pitcher": proportion_in_pitcher,
            "eef_mass": self.agent_config["pitcher_mass"] + ball_mass * proportion_in_pitcher,
            "eef_state": self._eef_state.clone(),
            "eef_pos": self._eef_state[:, :3].clone(),
            "eef_quat": self._eef_state[:, 3:7].clone(),
            "eef_vel": torch.clip(self._eef_state[:, 7:].clone(), -100., 100.),
            "eef_base_pos": self._eef_base_state[:, :3].clone(),
            "eef_base_quat": self._eef_base_state[:, 3:7].clone(),
            "eef_base_y_axis": self._eef_base_y_state[:, :3] - self._eef_base_state[:, :3],
            "eef_base_z_axis": self._eef_base_z_state[:, :3] - self._eef_base_state[:, :3],
            "j_eef": self._j_eef.clone(),
            "mm": self._mm.clone(),
        })

    def get_observations(self):
        """
        Collects and returns concatenated relevant observations, based on self.obs_keys

        Returns:
            2-tuple:
                tensor: (n_env, obs_dim) array of observations
                dict: additional key-mapped observations that may or may not include the main observation tensor
        """
        # Run super method first
        obs, obs_dict = super().get_observations()
        obs_dict["control_dict"] = self.get_control_dict()
        return obs, obs_dict

    def get_control_dict(self):
        """
        Grabs current control information relevant for computing controls

        Returns:
            dict: Keyword-mapped values potentially necessary for a controller computation
        """
        dic = {
            "q": self.states["q"],
            "qd": self.states["qd"],
            "qdd": self.states["qdd"],
            "mm": self.states["mm"],
            "mm_gt": self.states["mm"],
            "j_eef": self.states["j_eef"],
            "eef_state": self.states["eef_state"], # [pos, quat, lvel, avel] (13 dim)
            "extrinsics": torch.cat(
                [
                    self.states["eef_mass"],
                    self.states["dof_friction"],
                    self.states["dof_damping"],
                    self.states["dof_armature"],
                    self.states["min_body_inertia"],
                ],
                dim=-1),
        }

        if self._arm_control_history is not None:
            dic.update({
                "q_history": self.states["q_history"],
                "qd_history": self.states["qd_history"],
                "arm_control_history": self.states["arm_control_history"],
            })

        return dic

    def reset(self, env_ids=None):
        """
        Executes reset for this robot

        Args:
            env_ids (None or tensor): If specified, should be (integer) IDs corresponding to the
                specific env instances of this robot that should be reset
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.n_envs, device=self.device, dtype=torch.int32)

        n_resets = len(env_ids)

        # Run super method
        super().reset(env_ids=env_ids)

        # Reset agent
        reset_noise = torch.rand((n_resets, self.dof), device=self.device)
        pos = torch.zeros(n_resets, self.dof_total, device=self.device)
        pos_arm = tensor_clamp(
            self.dof_default.unsqueeze(0) +
            self.agent_config["reset_noise"] * 2.0 * (reset_noise - 0.5),
            self.dof_lower_limits, self.dof_upper_limits)
        pos[:, :self.dof] = pos_arm

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])
        self._qd_last[env_ids, :] = torch.zeros_like(self._qd[env_ids])
        self._qdd[env_ids, :] = torch.zeros_like(self._qd[env_ids])
        self._qdd_last[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos_arm
        self._vel_control[env_ids, :] = torch.zeros_like(pos_arm)
        self._effort_control[env_ids, :] = torch.zeros_like(pos_arm)

        # Re-randomize extrinsic parameters
        self._dof_damping[env_ids, :] = to_torch(np.power(10., np.random.uniform(
            low=np.log10(self.agent_config["dof_damping_min"]),
            high=np.log10(self.agent_config["dof_damping_max"]),
            size=(n_resets, len(self.agent_config["dof_damping_min"]))
        )), dtype=torch.float, device=self.device)
        self._dof_friction[env_ids, :] = to_torch(np.power(10., np.random.uniform(
            low=np.log10(self.agent_config["dof_friction_min"]),
            high=np.log10(self.agent_config["dof_friction_max"]),
            size=(n_resets, len(self.agent_config["dof_friction_min"]))
        )), dtype=torch.float, device=self.device)
        self._dof_armature[env_ids, :] = to_torch(np.power(10., np.random.uniform(
            low=np.log10(self.agent_config["dof_armature_min"]),
            high=np.log10(self.agent_config["dof_armature_max"]),
            size=(n_resets, len(self.agent_config["dof_armature_min"]))
        )), dtype=torch.float, device=self.device)
        self._min_body_inertia[env_ids, :] = to_torch(np.power(10., np.random.uniform(
            low=np.log10(self.agent_config["min_body_inertia"][0]),
            high=np.log10(self.agent_config["min_body_inertia"][1]),
            size=(n_resets, 1)
        )), dtype=torch.float, device=self.device)

        # Update values in sim
        rigid_body_names = self.gym.get_actor_rigid_body_names(self.envs[0], 0)
        dof_names = self.gym.get_actor_dof_names(self.envs[0], 0)
        for env_id in env_ids:
            env = self.envs[env_id]
            rigid_body_props = self.gym.get_actor_rigid_body_properties(env, 0)
            for j, (name, prop) in enumerate(zip(rigid_body_names, rigid_body_props)):
                # Randomize min body inertia
                default_inertia = self._default_body_inertia[env_id][j]
                prop.inertia.x = default_inertia.x + gymapi.Vec3(self._min_body_inertia[env_id].item(), 0, 0)
                prop.inertia.y = default_inertia.y + gymapi.Vec3(0, self._min_body_inertia[env_id].item(), 0)
                prop.inertia.z = default_inertia.z + gymapi.Vec3(0, 0, self._min_body_inertia[env_id].item())
            # Set the updated props
            self.gym.set_actor_rigid_body_properties(env, 0, rigid_body_props, recomputeInertia=False)
            # Modify joint properties next
            dof_props = self.gym.get_actor_dof_properties(env, 0)
            for j, (name, prop) in enumerate(zip(dof_names, dof_props)):
                # Randomize dof params
                dof_props['damping'][j] = self._dof_damping[env_id, j].item()
                dof_props['friction'][j] = self._dof_friction[env_id, j].item()
                dof_props['armature'][j] = self._dof_armature[env_id, j].item()
            # Set the updated props
            self.gym.set_actor_dof_properties(env, 0, dof_props)

        # Update extrinsic states
        self.states.update({
            "dof_friction": torch.log(self._dof_friction) / 10.,
            "dof_damping": torch.log(self._dof_damping) / 10.,
            "dof_armature": torch.log(self._dof_armature) / 10.,
            "min_body_inertia": torch.log(self._min_body_inertia) / 10.,
        })

        # We force a refresh here so that we can optionally place "liquid" within the pitcher
        # using updated EEF location if used by an external environment
        self.sim_states.set_actor_root_states_indexed(env_ids=env_ids)
        self.sim_states.set_dof_states_indexed(env_ids=env_ids)
        self.sim_states.clear_contact_forces_indexed(env_ids=env_ids)
        # Make sure env_ids is either None or equal to number of envs
        assert env_ids is None or len(env_ids) == self.n_envs, \
            "Forcing sim step during reset can only occur if resetting all envs at once!"
        self.gym.simulate(self.sim)
        self.sim_states.refresh(contact_forces=False)

        # Place balls in pitcher if getter is not None
        if self._balls_getter is not None:
            self._place_balls_in_pitcher(env_ids=env_ids)

    def calculate_gravity_torques(self, normalize=True):
        """
        Calculates gravity torques based on link masses and jacobian

        Args:
            normalize (bool): If True, will normalize gravity torques

        Returns:
            tensor: (n_envs, n_arm_dof) tensor containing gravity torques to apply
        """
        g = torch.zeros(self.n_envs, self._dof+1, 6, 1, dtype=torch.float, device=self.device)
        g[:, :, 2, :] = 9.81
        g_force = self._link_mass.unsqueeze(-1).unsqueeze(-1) * g
        g_torque = (torch.transpose(self._j_link, 2, 3) @ g_force).squeeze(-1)           # new shape is (n_envs, n_links, 7)
        g_torque = torch.sum(g_torque, dim=1, keepdim=False)

        if normalize:
            g_torque = g_torque / self.effort_limits[:self.dof].unsqueeze(0)

        return g_torque

    def register_balls_state_getter(self, getter):
        """
        Registers an external function that should grab the state of all balls for this agent to poll / modify

        Args:
            getter (function): External function for grabbing relevant info for all balls in env.
                Function should have following signature:

                    def getter():
                    args: None
                    returns:
                        dict:
                            n_balls (int): Number of balls in env
                            ball_size (float): Ball radius
                            ball_states (list of tensor): States of all balls -- this should be the direct reference
                                to writable buffer that belongs to sim_states
        """
        # Store getter
        self._balls_getter = getter

    def get_proportion_in_pitcher(self):
        """
        Gets the proportion of balls that are currently in (or close to) the pitcher.

        NOTE: This is only a rough estimate using a hacky heuristic, NOT accurate.

        Returns:
            tensor: Proportion of balls that are in (or close to) the pitcher
        """
        return _compute_ball_proportion(
            balls_pos=self._balls_getter()["ball_states"][:, :, :3],
            pitcher_pos=self.states["pitcher_wall_pos"],
            distance_threshold=self.agent_config["pitcher_size"][1] + 0.005,
        )

    def set_pitcher_color(self, env_id, color):
        """
        Method to set the pitcher color for specific env @env with color @color

        Args:
            env_id (int): Environment ID to set pitcher color
            color (3-array): (R,G,B) Color to set for the pitcher
        """
        # Get rigid body props
        pitcher_rb_props = self.gym.get_actor_rigid_body_properties(self.envs[env_id], 0)[self.dof + 1:]
        for i, rb_prop in enumerate(pitcher_rb_props):
            self.gym.set_rigid_body_color(self.envs[env_id], 0, self.dof + 1 + i, gymapi.MESH_VISUAL, gymapi.Vec3(*color))

    def _place_balls_in_pitcher(self, env_ids=None):
        """
        Simple method to reset balls' positions into the agent's held pitcher.

        Args:
            env_ids (tensor or None): Specific environments to place balls
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.n_envs, device=self.device, dtype=torch.long)

        # Grab relevant ball data
        ball_info = self._balls_getter()

        # Get new sphere locations
        ball_pos = get_sphere_positions_in_cylindrical_container(
            n_spheres=ball_info["n_balls"],
            sphere_radius=ball_info["ball_size"],
            tolerance=0.002,
            container_radius=self.agent_config["pitcher_size"][0],
            container_height=self.agent_config["pitcher_size"][1],
            container_pos=self._pitcher_base_state[env_ids, :3],
            container_ori_vec=self._pitcher_wall_state[env_ids, :3] - self._pitcher_base_state[env_ids, :3],
        )

        # Fill ball states (clear state, then fill pos with new values)
        ball_info["ball_states"][env_ids] = 0.0
        ball_info["ball_states"][env_ids, :, :3] = ball_pos
        ball_info["ball_states"][env_ids, :, 6] = 1.0       # quat w value

    def _get_control_dict(self):
        """
        Composes control dictionary based on current states for computing control
        """
        return {k: self.states[k] for k in ("eef_state", "q", "qd", "mm", "j_eef")}

    def _generate_urdf(self):
        """
        Helper method to procedurally generate the URDF for this custom franka robot, based on specific values set.

        Returns:
            str: relative fpath (relative to ASSETS_ROOT) to generated URDF
        """
        # Generate pitcher
        pitcher_elements = create_hollow_cylinder(
            name="pitcher",
            size=self.agent_config["pitcher_size"],
            thickness=self.agent_config["pitcher_thickness"],
            mass=self.agent_config["pitcher_mass"],
            n_slices=32,
            shape=self.agent_config["pitcher_shape"],
            use_lid=self.agent_config["use_lid"],
            transparent_walls=self.agent_config["transparent_walls"],
            generate_urdf=False,
        )

        # Compose joint to link pitcher to Franka EEF
        bridge_joint = create_joint(
            name="franka_pitcher_joint",
            parent="panda_link7",
            child=pitcher_elements[0].get("name"),
            pos=(-self.agent_config["pitcher_size"][1] / 2, self.agent_config["pitcher_size"][1] / 2, 0.107 + self.agent_config["pitcher_size"][0]),
            rpy=(1.57079632679, 0, 0.7853981633974483),
        )

        # Open Franka template and merge with custom pitcher
        template_fpath = os.path.join(ASSETS_ROOT, "urdf/franka_description/robots/franka_panda_template.urdf")
        lines = []
        with open(template_fpath, 'r') as f:
            for line in f:
                if "{{CUSTOM_EEF}}" in line:
                    # Convert all xml elements to strings
                    pretty_print_xml(bridge_joint)
                    lines.append(ET.tostring(bridge_joint, encoding="unicode"))
                    for ele in pitcher_elements:
                        pretty_print_xml(ele)
                        lines.append(ET.tostring(ele, encoding="unicode"))
                else:
                    lines.append(line)
        # Open new file and write generated URDF
        new_fpath_rel = "urdf/franka_description/robots/franka_panda_pitcher.urdf"
        new_fpath = os.path.join(ASSETS_ROOT, new_fpath_rel)
        with open(new_fpath, 'w') as f_out:
            for line in lines:
                f_out.write(line)

        # Return relative path to generated URDF
        return new_fpath_rel

    @property
    def default_agent_config(self):
        """
        Default agent configuration to use for this agent

        Returns:
            dict: Keyword-mapped values for the default agent configuration. Should, at the minimum, include the
                following keys:

                dof_default (array): Default joint qpos for this agent
                dof_stiffness (array): Stiffness values for each joint -- corresponds to gains if using position
                    spring constant if using torque control
                dof_damping_min (array): Minimum damping values for each joint
                dof_damping_max (array): Maximum damping values for each joint
                reset_noise (float): Normalized noise proportion in range [0, 1.0] to use when resetting this agent
                denormalize_control (bool): If True, assumes inputted u values are normalized to be in range [-1, 1],
                    and will scale them appropriately before executing them in sim
        """
        return {
            "dof_arm": 7,
            "dof_arm_mode": gymapi.DOF_MODE_EFFORT,
            "dof_default": [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469],
            "dof_max_velocities": [2.6180, 2.6180, 2.6180, 2.6180, 3.1416, 3.1416, 3.1416],
            "dof_max_efforts": [87., 87., 87., 87., 12., 12., 12.],
            "dof_stiffness": [0, 0, 0, 0, 0, 0, 0],
            "dof_damping_min": [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001],
            "dof_damping_max": [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001],
            "dof_friction_min": [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001],
            "dof_friction_max": [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001],
            "dof_armature_min": [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001],
            "dof_armature_max": [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001],
            "reset_noise": 0.25,
            "denormalize_control": True,
            "pitcher_color": None,
        }

    @property
    def dof(self):
        return self._dof

    @property
    def action_dim(self):
        """
        Dimension of this agent's action space -- this is the nDOF arm + 1 (for gripper action)

        Returns:
            int: Dimension of agent's action space
        """
        return self.agent_config["dof_arm"]

    @property
    def control_modes(self):
        """
        Control modes that this agent uses. Should be a subset of
            (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_VEL, gymapi.DOF_MODE_EFFORT)

        Returns:
            set: Mode(s) used to control this agent
        """
        return {self.dof_arm_mode}

    @property
    def name(self):
        """
        Name of this agent.

        Returns:
            str: Agent name
        """
        return "franka"


@torch.jit.script
def _compute_ball_proportion(
    balls_pos,
    pitcher_pos,
    distance_threshold,
):
    # type: (Tensor, Tensor, float) -> Tensor

    # Helper function that returns proportion of balls that are in (or close to) the pitcher

    # Calculate distance
    dist = torch.norm(balls_pos - pitcher_pos.unsqueeze(dim=1), dim=-1)

    return torch.mean((dist < distance_threshold).float(), dim=-1)
