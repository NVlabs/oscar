# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""
Helper functions / classes for using Isaac Gym
"""
from isaacgym import gymtorch
from isaacgym import gymapi
import torch
from collections import Iterable


class SimInterface:
    """
    Base class for central interfaces with sim. Subclasses should serve as singular interface points for directly
    interfacing with the sim.

    Args:
        gym (Gym): Active gym object
        sim (Sim): Active sim object
        device (str or int): Device to send action tensors to
        actors_with_dof (int or list): Actor handle(s) corresponding to actors with nDOF > 0
    """
    def __init__(self, gym, sim, device, actors_with_dof):
        # Store internal references
        self._gym = gym
        self._sim = sim
        self.device = device
        self.actors_wtih_dof = [actors_with_dof] if isinstance(actors_with_dof, int) else actors_with_dof

        # Get relevant sim metadata
        self.n_envs = self._gym.get_env_count(self._sim)
        self.n_bodies = self._gym.get_sim_rigid_body_count(self._sim)
        self.n_bodies_per_env = self.n_bodies // self.n_envs
        self.n_actors = self._gym.get_sim_actor_count(self._sim)
        self.n_actors_per_env = self.n_actors // self.n_envs
        self.n_dof = self._gym.get_sim_dof_count(self._sim)
        self.n_dof_per_env = self.n_dof // self.n_envs

    def _ids_to_global_ids(self, actor_ids=None, env_ids=None, only_actors_with_dof=False):
        """
        Converts the requested @actor_ids and @env_ids into a single 1D torch tensor of equivalent global IDs

        Args:
            actor_ids (None or int or list or tensor): Actor (relative) ID(s) corresponding to actors that
                will be modified. If None, we assume that all actors will be modified
            env_ids (None or int or list or tensor): Environment ID(s) corresponding to envs that will be modified.
                If None, we assume that all envs will be modified
            only_actors_with_dof (bool): If True, if actor_ids is None, will only populate actor ids with ids
                corresponding to actors that have nDOF > 0

        Returns:
            tensor: 1D tensor of length len(actor_ids) * len(env_ids)
        """
        # First make sure both inputs are iterables
        if not isinstance(actor_ids, Iterable):
            if actor_ids is None:
                actor_ids = self.actors_wtih_dof if only_actors_with_dof else torch.arange(self.n_actors_per_env)
            else:
                actor_ids = [actor_ids]
        if not isinstance(env_ids, Iterable):
            env_ids = torch.arange(self.n_envs) if env_ids is None else [env_ids]
        else:
            env_ids = env_ids.clone()

        # Compose array
        global_ids = torch.arange(self.n_actors, dtype=torch.int32, device=self.device, requires_grad=False).view(self.n_envs, -1)

        # Grab relevant indices, flatten, and return
        return global_ids[env_ids][:, actor_ids].flatten()


class SimStates(SimInterface):
    """
    Simple class that should serve a singular reference to all relevant simulation states
    (root states, dof states, rigid body states). Only one instance should exist per sim, and
    any external objects should take views / slices of this object's tensor attributes in order
    to maintain the singular reference.

    Main attributes that should be shared with external objects are the following:

    self.actor_root_states      (tensor) (n_env, n_actor_per_env, 13), where 13 = (pos, quat, lin_vel, ang_vel)
    self.dof_states             (tensor) (n_env, total_dof_per_env, 2), where 2 = (pos, vel)
    self.rigid_body_states      (tensor) (n_env, n_rigid_bodies_per_env, 13), where 13 = (pos, quat, lin_vel, ang_vel)
    self.contact_forces         (tensor) (n_env, n_rigid_bodies_per_env, 3), where 3 = (f_x, f_y, f_z)

    Args:
        gym (Gym): Active gym object
        sim (Sim): Active sim object
        device (str or int): Device to send action tensors to
        actors_with_dof (int or list): Actor handle(s) corresponding to actors with nDOF > 0
    """
    def __init__(self, gym, sim, device, actors_with_dof):
        # Run super init first
        super().__init__(gym=gym, sim=sim, device=device, actors_with_dof=actors_with_dof)

        # Setup GPU state tensors
        _actor_root_state_tensor = self._gym.acquire_actor_root_state_tensor(self._sim)
        _dof_state_tensor = self._gym.acquire_dof_state_tensor(self._sim)
        _rigid_body_state_tensor = self._gym.acquire_rigid_body_state_tensor(self._sim)
        _contact_forces_tensor = self._gym.acquire_net_contact_force_tensor(self._sim)

        # Wrap references in an actual tensor that we can call later
        self.actor_root_states = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.n_envs, -1, 13)
        self.dof_states = gymtorch.wrap_tensor(_dof_state_tensor).view(self.n_envs, -1, 2)
        self.rigid_body_states = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.n_envs, -1, 13)
        self.contact_forces = gymtorch.wrap_tensor(_contact_forces_tensor).view(self.n_envs, -1, 3)

    def refresh(self, contact_forces=True):
        """
        Refreshes all internal tensors. Should only occur ONCE per sim.simulate() step

        Args:
            contact_forces (bool): If True, will refresh contact forces. Should be set to True if a sim.simulate() step
                has occurred.
        """
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        if contact_forces:
            self._gym.refresh_net_contact_force_tensor(self._sim)

    def set_actor_root_states(self):
        """
        Sets the actor root states based on the current references. Should only occur ONCE per sim.simulate() step
        """
        self._gym.set_actor_root_state_tensor(self._sim, gymtorch.unwrap_tensor(self.actor_root_states))

    def set_actor_root_states_indexed(self, actor_ids=None, env_ids=None):
        """
        Sets a subset of all actor root states based on the current references. Should only occur ONCE
        per sim.simulate() step.

        Args:
            actor_ids (None or int or list or tensor): Actor (relative) ID(s) corresponding to actors that
                will be modified. If None, we assume that all actors will be modified
            env_ids (None or int or list or tensor): Environment ID(s) corresponding to envs that will be modified.
                If None, we assume that all envs will be modified
        """
        # If both inputs are None, we simply run the non-indexed version for speed
        if actor_ids is None and env_ids is None:
            self.set_actor_root_states()
        else:
            # Convert relative IDs into global ids
            global_ids = self._ids_to_global_ids(actor_ids=actor_ids, env_ids=env_ids, only_actors_with_dof=False)
            self._gym.set_actor_root_state_tensor_indexed(
                self._sim, gymtorch.unwrap_tensor(self.actor_root_states),
                gymtorch.unwrap_tensor(global_ids), len(global_ids))

    def set_dof_states(self):
        """
        Sets the DOF states based on the current references. Should only occur ONCE per sim.simulate() step
        """
        self._gym.set_dof_state_tensor(self._sim, gymtorch.unwrap_tensor(self.dof_states))

    def set_dof_states_indexed(self, actor_ids=None, env_ids=None):
        """
        Sets a subset of all DOF states based on the current references. Should only occur ONCE
        per sim.simulate() step.

        Args:
            actor_ids (None or int or list or tensor): Actor (relative) ID(s) corresponding to actors that
                will be modified. If None, we assume that all actors will be modified
            env_ids (None or int or list or tensor): Environment ID(s) corresponding to envs that will be modified.
                If None, we assume that all envs will be modified
        """
        # If both inputs are None, we simply run the non-indexed version for speed
        if actor_ids is None and env_ids is None:
            self.set_dof_states()
        else:
            # Convert relative IDs into global ids
            global_ids = self._ids_to_global_ids(actor_ids=actor_ids, env_ids=env_ids, only_actors_with_dof=True)
            self._gym.set_dof_state_tensor_indexed(
                self._sim, gymtorch.unwrap_tensor(self.dof_states),
                gymtorch.unwrap_tensor(global_ids), len(global_ids))

    def set_rigid_body_states(self):
        """
        Sets the rigid body states based on the current references. Should only occur ONCE per sim.simulate() step
        """
        self._gym.set_rigid_body_state_tensor(self._sim, gymtorch.unwrap_tensor(self.rigid_body_states))

    def set_rigid_body_states_indexed(self, actor_ids=None, env_ids=None):
        """
        Sets a subset of all rigid body states based on the current references. Should only occur ONCE
        per sim.simulate() step.

        Args:
            actor_ids (None or int or list or tensor): Actor (relative) ID(s) corresponding to actors that
                will be modified. If None, we assume that all actors will be modified
            env_ids (None or int or list or tensor): Environment ID(s) corresponding to envs that will be modified.
                If None, we assume that all envs will be modified
        """
        raise NotImplementedError

    def clear_contact_forces(self):
        """
        Clears the contact forces.

        NOTE: Calling self.refresh(contact_forces=True) will override these values!
        """
        self.contact_forces[:] = torch.zeros_like(self.contact_forces[:])

    def clear_contact_forces_indexed(self, env_ids=None):
        """
        Clears a subset of all contact forces based on the current references.

        NOTE: Calling self.refresh(contact_forces=True) will override these values!

        Args:
            env_ids (None or int or list or tensor): Environment ID(s) corresponding to envs that will be modified.
                If None, we assume that all envs will be modified
        """
        # If both inputs are None, we simply run the non-indexed version for speed
        if env_ids is None:
            self.clear_contact_forces()
        else:
            # Standardize end_ids
            if not isinstance(env_ids, Iterable):
                env_ids = torch.arange(self.n_envs) if env_ids is None else [env_ids]
            else:
                env_ids = env_ids.clone()
            # Clear requested contact forces
            self.contact_forces[env_ids] = torch.zeros_like(self.contact_forces[env_ids])


class SimActions(SimInterface):
    """
    Simple class that should serve a singular reference to all relevant simulation actions
    (dof pos, vel, effort). Only one instance should exist per sim, and
    any external objects should take views / slices of this object's tensor attributes in order
    to maintain the singular reference.

    NOTE: We assume all envs have the same number of DOFs

    Main attributes that should be shared with external objects are the following:

    self.pos_actions        (tensor) (n_env, n_dof_per_env)
    self.vel_actions        (tensor) (n_env, n_dof_per_env)
    self.effort_actions     (tensor) (n_env, n_dof_per_env)

    Args:
        gym (Gym): Active gym object
        sim (Sim): Active sim object
        device (str or int): Device to send action tensors to
        actors_with_dof (int or list): Actor handle(s) corresponding to actors with nDOF > 0
        modes (int or list or set): Modes that actions cover. Should be one / list of
            (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_VEL, gymapi.DOF_MODE_EFFORT)
    """

    def __init__(self, gym, sim, device, actors_with_dof, modes=[gymapi.DOF_MODE_POS]):
        # Run super init first
        super().__init__(gym=gym, sim=sim, device=device, actors_with_dof=actors_with_dof)

        # Store modes
        self.modes = set(modes) if isinstance(modes, Iterable) else {modes}

        # Setup action tensors
        self.pos_actions = torch.zeros((self.n_envs, self.n_dof_per_env), dtype=torch.float, device=self.device)
        self.vel_actions = torch.zeros_like(self.pos_actions)
        self.effort_actions = torch.zeros_like(self.pos_actions)

    def deploy(self):
        """
        Applies the internal actions in sim. Should only occur ONCE per sim.simulate() step
        """
        if gymapi.DOF_MODE_POS in self.modes:
            self._gym.set_dof_position_target_tensor(self._sim, gymtorch.unwrap_tensor(self.pos_actions))
        if gymapi.DOF_MODE_VEL in self.modes:
            self._gym.set_dof_velocity_target_tensor(self._sim, gymtorch.unwrap_tensor(self.vel_actions))
        if gymapi.DOF_MODE_EFFORT in self.modes:
            self._gym.set_dof_actuation_force_tensor(self._sim, gymtorch.unwrap_tensor(self.effort_actions))

    def deploy_indexed(self, actor_ids=None, env_ids=None):
        """
        Applies subset of internal actions in sim. Should only occur ONCE per sim.simulate() step

        Args:
            actor_ids (None or int or list or tensor): Actor (relative) ID(s) corresponding to actors that
                will be modified. If None, we assume that all actors will be modified
            env_ids (None or int or list or tensor): Environment ID(s) corresponding to envs that will be modified.
                If None, we assume that all envs will be modified
        """
        # If both inputs are None, we simply run the non-indexed version for speed
        if actor_ids is None and env_ids is None:
            self.deploy()
        else:
            # Convert relative IDs into global ids
            global_ids = self._ids_to_global_ids(actor_ids=actor_ids, env_ids=env_ids, only_actors_with_dof=True)
            n_ids = len(global_ids)

            # Apply actions
            if gymapi.DOF_MODE_POS in self.modes:
                self._gym.set_dof_position_target_tensor_indexed(
                    self._sim, gymtorch.unwrap_tensor(self.pos_actions), gymtorch.unwrap_tensor(global_ids), n_ids)
            if gymapi.DOF_MODE_VEL in self.modes:
                self._gym.set_dof_velocity_target_tensor_indexed(
                    self._sim, gymtorch.unwrap_tensor(self.vel_actions), gymtorch.unwrap_tensor(global_ids), n_ids)
            if gymapi.DOF_MODE_EFFORT in self.modes:
                self._gym.set_dof_actuation_force_tensor_indexed(
                    self._sim, gymtorch.unwrap_tensor(self.effort_actions), gymtorch.unwrap_tensor(global_ids), n_ids)
