# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
from oscar.utils.sim_utils import SimStates, SimActions
from collections import deque


class Agent:
    """
    Base interface that encapsulates a controllable entity to be used in an environment.

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
        # Store internal references and device
        self.device = device
        self.agent_config = agent_config

        # Placeheolders that will be filled later
        self.gym = None
        self.sim = None
        self.envs = None
        self.n_envs = None

        # Handle references
        self.handles = None                                             # dict that contains keyword-mapped handle references in sim

        # States
        self.contact_forces = {}                                        # dict that contains keyword-mapped contact forces for relevant bodies (should be filled in by subclass)
        self.states = {}                                                # dict that contains keyword-mapped state-related tensors for dof pos, vel, etc.
        self.actions = None                                             # dict that contains keyword-mapped action-related tensors (eg.: action, log prob, etc.)
        self.sim_states = None                                          # SimStates object for directly interfacing with gym obj
        self._n_frames_stack = self.agent_config["n_frames_stack"]      # How many frames to stack for observations
        self._should_update_obs_history = False                         # Flag for making sure obs history only gets updated once per sim step
        self._control_models = {}                                       # Control models for this agent

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._root_pos = None               # Base position             (n_envs, 3)
        self._root_quat = None              # Base quaternion           (n_envs, 4)
        self._dof_state = None              # State of all joints       (n_envs, n_dof)
        self._q = None                      # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._qd_last = None                # Joint velocities at prior timestep    (n_envs, n_dof)
        self._qdd = None                    # (Estimate of) joint accelerations     (n_envs, n_dof)
        self._qdd_last = None               # (Estimate of) joint accelerations at prior timestep     (n_envs, n_dof)
        self._rigid_body_state = None       # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._pos_control = None            # Position control of joints           (n_envs, n_dof)
        self._vel_control = None            # Velocity control of joints           (n_envs, n_dof)
        self._effort_control = None         # Effort control of joints             (n_envs, n_dof)
        self._q_history = None              # History of q values         (deque)
        self._qd_history = None             # History of qd values        (deque)
        self._obs_history = None            # History of obs values (dict of deques)

    def load_asset(self, gym, sim, n_envs):
        """
        Loads this agent into the simulation

        Args:
            gym (Gym): Active gym instance
            sim (Sim): Active sim instance
            n_envs (int): Number of environments in simulation

        Returns:
            2-tuple:
                Asset: Processed asset representing this agent
                dof_properties: DOF Properties for this agent
        """
        raise NotImplementedError

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
        # Store sim states
        self.sim_states = sim_states

        # Store environment pointers
        self.envs = env_ptrs

        # Setup tensor buffers

        # root state
        self._root_state = sim_states.actor_root_states[:, actor_handle]
        self._root_pos = self._root_state[..., 0:3]
        self._root_quat = self._root_state[..., 3:7]

        # dof states
        self._dof_state = sim_states.dof_states[:, :self.dof] # TODO: implicitly hardcoded for actor_handle=0
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._qd_last = self._qd.clone()
        self._qdd = torch.zeros_like(self._qd_last, device=self.device)
        self._qdd_last = torch.zeros_like(self._qd_last, device=self.device)

        # optionally history of q, qd
        history_length = self.agent_config["history_length"]
        if history_length > 0:
            self._q_history = deque([torch.zeros(self.n_envs, self.dof, device=self.device).type_as(self._q) for _ in range(history_length)], maxlen=history_length)
            self._qd_history = deque([torch.zeros(self.n_envs, self.dof, device=self.device).type_as(self._q) for _ in range(history_length)], maxlen=history_length)

        # rigid body statse
        self._rigid_body_state = sim_states.rigid_body_states

        # action buffers
        self._pos_control = sim_actions.pos_actions[:, :self.dof] # TODO: implicitly hardcoded for actor_handle=0
        self._vel_control = sim_actions.vel_actions[:, :self.dof] # TODO: implicitly hardcoded for actor_handle=0
        self._effort_control = sim_actions.effort_actions[:, :self.dof] # TODO: implicitly hardcoded for actor_handle=0

        # Setup any handles necessary
        self.handles = {
            "actor": actor_handle,
        }

    def update_states(self, dt=None):
        """
        Updates the internal states for this agent

        NOTE: Assumes simulation has already refreshed states!!

        Args:
            dt (None or float): Amount of sim time (in seconds) that has passed since last update. If None, will assume
                that this is simply a refresh (no sim time has passed) and will not update any values dependent on dt
        """
        # Update tensors

        # Modify history and qdd if we're actually taking a forward step (dt is not None)
        if dt is not None:
            self._qdd = ((self._qd - self._qd_last) / dt + self._qdd_last) / 2.0
            self._qdd_last = self._qdd.clone()
            if self._q_history is not None:
                self._q_history.append(self.states["q"]) if "q" in self.states else torch.zeros_like(self._q)
                self._qd_history.append(self.states["qd"]) if "qd" in self.states else torch.zeros_like(self._qd)
                self.states.update({
                    "q_history": torch.stack(tuple(self._q_history), dim=1),
                    "qd_history": torch.stack(tuple(self._qd_history), dim=1),
                })
            # May update obs history
            if self._n_frames_stack > 1 and self._obs_history is not None:
                # Obs gets updated at once during get_observation call
                self._should_update_obs_history = True

        self._qd_last = self._qd.clone()

        # Update internal states
        self.states.update({
            "root_pos": self._root_pos.clone(),
            "root_quat": self._root_quat.clone(),
            "q": self._q.clone(),
            "qd": self._qd.clone().clip(-500., 500.),
            "qdd": self._qdd.clone().clip(-5000., 5000.),
        })

    def get_observations(self):
        """
        Collects and returns concatenated relevant observations, based on self.obs_keys

        Returns:
            2-tuple:
                tensor: (n_env, obs_dim) array of observations
                dict: additional key-mapped observations that may or may not include the main observation tensor
        """
        # Optionally stack frames
        if self._n_frames_stack > 1:
            if self._obs_history is None:
                # Initialize obs history
                self._obs_history = {
                    k: deque([torch.zeros_like(self.states[k]) for _ in
                        range(self._n_frames_stack)], maxlen=self._n_frames_stack)
                    for k in self.obs_keys
                }
            obs = []
            for k in self.obs_keys:
                # Update obs history only if the update flag is set
                if self._should_update_obs_history:
                    self._obs_history[k].append(self.states[k])
                obs.append(torch.cat(tuple(self._obs_history[k]), dim=-1))
            obs = torch.cat(obs, dim=-1)

            # Clear update flag
            self._should_update_obs_history = False

        else:
            obs = torch.cat([self.states[k] for k in self.obs_keys], dim=-1)

        return obs, {}

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

        # Reset the q, qd history for the specific envs
        history_length = self.agent_config["history_length"]
        if history_length > 0:
            for i in range(history_length):
                self._q_history[i][env_ids] = 0.0
                self._qd_history[i][env_ids] = 0.0

        # Reset the obs history
        if self._n_frames_stack > 1:
            for k in self.obs_keys:
                for i in range(self._n_frames_stack):
                    self._obs_history[k][i][env_ids] = 0.0

    def register_control_model(self, name, model):
        """
        Registers external control models that can be referenced by this env class

        Args:
            name (str): Name of control model to register
            model (ControlModel): model to register
        """
        self._control_models[name] = model

    @property
    def default_agent_config(self):
        """
        Default agent configuration to use for this agent

        Returns:
            dict: Keyword-mapped values for the default agent configuration. Should, at the minimum, include the
                following keys:

                observations (list): List of observation key names to gather when collecting observations
                dof_default (array): Default joint qpos for this agent
                dof_lower_limits (array): Lower limits for joints
                dof_upper_limits (array): Upper limits for joints
                dof_max_efforts (array): Max effort that can be applied to each joint
                dof_stiffness (array): Stiffness values for each joint -- corresponds to gains if using position
                    spring constant if using torque control
                dof_damping (array): Damping values for each joint
        """
        raise NotImplementedError

    @property
    def dof(self):
        """
        How many degrees of freedom this agent has

        Returns:
            int: Number of DOFs for this agent
        """
        raise NotImplementedError

    @property
    def action_dim(self):
        """
        Dimension of this agent's action space (usually corresponds to the controller's command_dim)

        Returns:
            int: Dimension of agent's action space
        """
        raise NotImplementedError

    @property
    def control_modes(self):
        """
        Control modes that this agent uses. Should be a subset of
            (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_VEL, gymapi.DOF_MODE_EFFORT)

        Returns:
            set: Mode(s) used to control this agent
        """
        raise NotImplementedError

    @property
    def obs_keys(self):
        """
        String names that correspond to observations that we want to gather during self.get_observations()

        Returns:
            list: List of observation key names to gather when collecting observations
        """
        return self.agent_config["observations"]

    @property
    def name(self):
        """
        Name of this agent.

        Returns:
            str: Agent name
        """
        raise NotImplementedError
