# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import operator
from copy import deepcopy
from collections import Iterable
import random
import numpy as np

from isaacgym import gymapi
from isaacgym.gymutil import get_property_setter_map, get_property_getter_map, apply_random_samples, check_buckets

from oscar.utils.config_utils import get_sim_params
from oscar.utils.sim_utils import SimStates, SimActions

import torch


# Base class for RL tasks
class BaseTask:
    """
    Base interface for all tasks.
    """
    def __init__(
        self,
        cfg,
    ):
        # Setup gym reference
        self.gym = gymapi.acquire_gym()

        # Extract relevant info from the config
        self.cfg = cfg
        self.sim_cfg = cfg["sim"]
        self.task_cfg = cfg["task"]

        # Task configuration
        self.n_envs = self.task_cfg["numEnvs"]
        self.env_spacing = self.task_cfg["envSpacing"]
        self.max_episode_length = self.task_cfg["episodeLength"]
        self.action_scale = self.task_cfg["actionScale"]
        self.start_position_noise = self.task_cfg["startPositionNoise"]
        self.start_rotation_noise = self.task_cfg["startRotationNoise"]
        self.debug_viz = self.task_cfg["enableDebugVis"]
        self.observation_noise = self.task_cfg["observation_noise"]

        # Sim configuration
        self.dt = self.sim_cfg["dt"]
        self.substeps = self.sim_cfg["substeps"]
        self.headless = self.sim_cfg["headless"]
        self.physics_engine = self.sim_cfg["physics_engine"]
        self.graphics_device = self.sim_cfg["graphics_device"]
        self.compute_device = self.sim_cfg["compute_device"]
        self.save_video = self.sim_cfg["save_video"]
        # Set device depending on whether we're receiving a device directly (str) or implicitly (int)
        if isinstance(self.sim_cfg["device"], str):
            self.device = self.sim_cfg["device"]
        else:
            self.device = "cpu" if self.sim_cfg["device"] < 0 else f"cuda:{self.sim_cfg['device']}"
        self.up_axis = self.sim_cfg["up_axis_str"]
        self.control_freq_inv = self.sim_cfg["control_freq_inv"]
        self.enable_viewer_sync = self.sim_cfg["enable_viewer_sync"]
        self.render_every_n_steps = self.sim_cfg["render_every_n_steps"]
        self.img_dim = self.sim_cfg["img_dim"]

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # Placeholders that will be filled in later
        self.sim = None
        self.viewer = None
        self.states = {}
        self.sim_states = None
        self.sim_actions = None
        self.n_bodies = None
        self.n_actors = None
        self.envs = []
        self.actors_with_dof = []               # Actor IDs corresponding to actors that have actuatable joints
        self._i = 0                             # Utility counter, e.g.: keeping track of frames when saving video

        # Create placeholders for buffers
        self.obs_dict = {}
        self.obs_buf = None
        self.rew_buf = None
        self.reset_buf = None
        self._last_reset_buf = None
        self.progress_buf = None
        self.randomize_buf = None
        self.extras = {}
        self.contact_forces = {}                # Holds keyword-mapped contact force magnitudes for relevant bodies

        # Randomization Info
        self.original_props = {}
        self.dr_randomizations = {}
        self.first_randomization = True

        # Create sim params from cfg
        self.sim_params = self.sim_cfg.get("sim_params", get_sim_params(self.sim_cfg))

        # Make sure up axis convention is correct
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        # create envs, sim and viewer
        self.create_sim()

        # Setup counters
        self.last_step = -1
        self.last_rand_step = -1
        self._render_count = 0      # Tracker used to only render occasionally

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.graphics_device != -1 and not self.headless:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                # # For close-up of small num envs
                # cam_pos = gymapi.Vec3(0.7, 1.3, 1.5)
                # cam_target = gymapi.Vec3(0.6, 0, 1.0)

                # For far-away shot of many envs
                cam_pos = gymapi.Vec3(20.0, 30.0, 3.0)
                cam_target = gymapi.Vec3(15.0, 10.0, -7.0)

            else:
                cam_pos = gymapi.Vec3(20.0, 7.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

        # Reset all the envs
        self.reset()

    def setup_references(self):
        """
        Sets up any relevant references in this sim (e.g., handles, states, etc.)
        """
        # Determine which actors can actually be set
        self.n_actors = self.gym.get_actor_count(self.envs[0])
        for actor_id in range(self.n_actors):
            if self.gym.get_actor_dof_count(self.envs[0], actor_id) > 0:
                self.actors_with_dof.append(actor_id)

        # Get singular reference to states and actions
        self.sim_states = SimStates(gym=self.gym, sim=self.sim, device=self.device,
                                    actors_with_dof=self.actors_with_dof)
        self.sim_actions = SimActions(gym=self.gym, sim=self.sim, device=self.device,
                                      actors_with_dof=self.actors_with_dof, modes=self.action_modes)

        # Get total number of bodies (per env)
        self.n_bodies = self.sim_states.n_bodies_per_env

    def setup_buffers(self):
        """
        Sets up the buffers for this task, and also stores the obs_dim and action_dim values internally
        as well.

        Args:
            obs_dim (int): Observation dimension to use for this task
            action_dim (int): Action dimension to use for this task
        """
        # Run a dummy sim step and potentially fetch results if we're on CPU so the buffesr are filled
        self.gym.simulate(self.sim)
        if self.device == "cpu":
            self.gym.fetch_results(self.sim, True)

        # allocate buffers
        self.rew_buf = torch.zeros(
            self.n_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.n_envs, device=self.device, dtype=torch.long)
        self._last_reset_buf = torch.zeros(
            self.n_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.n_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.n_envs, device=self.device, dtype=torch.long)

        # Fill the obs dict and obs buf
        self.compute_observations()

    # set gravity based on up axis and return axis index
    def set_sim_params_up_axis(self, sim_params, axis):
        if axis == 'z':
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
            return 2
        return 1

    def step(self, actions):
        # Copy actions
        actions = actions.clone().to(self.device)

        # Use action randomization if requested
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        # apply actions
        self._pre_physics_step(actions)

        # step physics and render each frame
        if self._render_count == self.render_every_n_steps - 1:
            self.render()
        for i in range(self.control_freq_inv):
            self.gym.simulate(self.sim)

        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # render frame if we're saving video
        if self.save_video and self.viewer is not None and (self._render_count == 0):
            num = str(self._i)
            num = '0' * (6 - len(num)) + num
            self.gym.write_viewer_image_to_file(self.viewer, f"frame{num}.png")
            self._i += 1

        # Update render count
        self._render_count = (self._render_count + 1) % self.render_every_n_steps

        # compute observations, rewards, resets, ...
        self._post_physics_step(actions)

        # Use obs randomization if requested
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

    def get_states(self):
        return self.states_buf

    def render(self, sync_frame_time=False):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)

    # Apply randomizations only on resets, due to current PhysX limitations
    def apply_randomizations(self, dr_params):
        # If we don't have a randomization frequency, randomize every step
        rand_freq = dr_params.get("frequency", 1)

        # First, determine what to randomize:
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        #   - on the first call, randomize everything
        self.last_step = self.gym.get_frame_count(self.sim)
        if self.first_randomization:
            do_nonenv_randomize = True
            env_ids = range(self.n_envs)
        else:
            do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
            rand_envs = torch.where(self.randomize_buf >= rand_freq, torch.ones_like(self.randomize_buf), torch.zeros_like(self.randomize_buf))
            rand_envs = torch.logical_and(rand_envs, self.reset_buf)
            env_ids = torch.nonzero(rand_envs, as_tuple=False)
            self.randomize_buf[rand_envs] = 0

        if do_nonenv_randomize:
            self.last_rand_step = self.last_step

        param_setters_map = get_property_setter_map(self.gym)
        param_getters_map = get_property_getter_map(self.gym)

        # On first iteration, check the number of buckets
        if self.first_randomization:
            check_buckets(self.gym, self.envs, dr_params)

        for nonphysical_param in ["observations", "actions"]:
            if nonphysical_param in dr_params and do_nonenv_randomize:
                dist = dr_params[nonphysical_param]["distribution"]
                op_type = dr_params[nonphysical_param]["operation"]
                sched_type = dr_params[nonphysical_param]["schedule"] if "schedule" in dr_params[nonphysical_param] else None
                sched_step = dr_params[nonphysical_param]["schedule_steps"] if "schedule" in dr_params[nonphysical_param] else None
                op = operator.add if op_type == 'additive' else operator.mul

                if sched_type == 'linear':
                    sched_scaling = 1.0 / sched_step * \
                        min(self.last_step, sched_step)
                elif sched_type == 'constant':
                    sched_scaling = 0 if self.last_step < sched_step else 1
                else:
                    sched_scaling = 1

                if dist == 'gaussian':
                    mu, var = dr_params[nonphysical_param]["range"]
                    mu_corr, var_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        mu *= sched_scaling
                        var *= sched_scaling
                        mu_corr *= sched_scaling
                        var_corr *= sched_scaling
                    elif op_type == 'scaling':
                        var = var * sched_scaling  # scale up var over time
                        mu = mu * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                        var_corr = var_corr * sched_scaling  # scale up var over time
                        mu_corr = mu_corr * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * params['var_corr'] + params['mu_corr']
                        return op(
                            tensor, corr + torch.randn_like(tensor) * params['var'] + params['mu'])

                    self.dr_randomizations[nonphysical_param] = {'mu': mu, 'var': var, 'mu_corr': mu_corr, 'var_corr': var_corr, 'noise_lambda': noise_lambda}

                elif dist == 'uniform':
                    lo, hi = dr_params[nonphysical_param]["range"]
                    lo_corr, hi_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        lo *= sched_scaling
                        hi *= sched_scaling
                        lo_corr *= sched_scaling
                        hi_corr *= sched_scaling
                    elif op_type == 'scaling':
                        lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * (params['hi_corr'] - params['lo_corr']) + params['lo_corr']
                        return op(tensor, corr + torch.rand_like(tensor) * (params['hi'] - params['lo']) + params['lo'])

                    self.dr_randomizations[nonphysical_param] = {'lo': lo, 'hi': hi, 'lo_corr': lo_corr, 'hi_corr': hi_corr, 'noise_lambda': noise_lambda}

        if "sim_params" in dr_params and do_nonenv_randomize:
            prop_attrs = dr_params["sim_params"]
            prop = self.gym.get_sim_params(self.sim)

            if self.first_randomization:
                self.original_props["sim_params"] = {
                    attr: getattr(prop, attr) for attr in dir(prop)}

            for attr, attr_randomization_params in prop_attrs.items():
                apply_random_samples(
                    prop, self.original_props["sim_params"], attr, attr_randomization_params, self.last_step)

            self.gym.set_sim_params(self.sim, prop)

        for actor, actor_properties in dr_params["actor_params"].items():
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor)
                for prop_name, prop_attrs in actor_properties.items():
                    if prop_name == 'color':
                        num_bodies = self.gym.get_actor_rigid_body_count(
                            env, handle)
                        for n in range(num_bodies):
                            self.gym.set_rigid_body_color(env, handle, n, gymapi.MESH_VISUAL,
                                                          gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
                        continue

                    prop = param_getters_map[prop_name](env, handle)
                    if isinstance(prop, list):
                        if self.first_randomization:
                            self.original_props[prop_name] = [
                                {attr: getattr(p, attr) for attr in dir(p)} for p in prop]
                        for p, og_p in zip(prop, self.original_props[prop_name]):
                            for attr, attr_randomization_params in prop_attrs.items():
                                apply_random_samples(
                                    p, og_p, attr, attr_randomization_params, self.last_step)
                    else:
                        if self.first_randomization:
                            self.original_props[prop_name] = deepcopy(prop)
                        for attr, attr_randomization_params in prop_attrs.items():
                            apply_random_samples(
                                prop, self.original_props[prop_name], attr, attr_randomization_params, self.last_step)

                    param_setters_map[prop_name](env, handle, prop)

        self.first_randomization = False

    def create_sim(self):
        self.sim = self.gym.create_sim(self.compute_device, self.graphics_device, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs()

        # Prepare the sim for loading data
        self.gym.prepare_sim(self.sim)

        # Setup internal references and tensors
        self.setup_references()
        self.setup_buffers()

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        raise NotImplementedError

    def _pre_physics_step(self, actions):
        """
        Runs any pre-physics computations necessary (anything before calling sim.simulate()). Usually,
        this means processing and executing @actions in sim.

        Subclasses should extend this method to process actions accordingly; this base method merely deploys
        the actiosn in sim using the SimActions API

        Args:
            action (tensor): (n_env, n_actions) Actions to execute in sim
        """
        # Deploy actions
        self.sim_actions.deploy()

    def _post_physics_step(self, actions):
        """
        Runs any post-physics computations necessary (anything before calling sim.simulate()). Usually,
        this means processing resets, computing rewards, and computing observations.

        Args:
            action (tensor): (n_env, n_actions) Actions executed in sim
        """
        # Increment progress buffer (how many steps we've taken)
        self.progress_buf += 1

        # Determine which environments we need to reset (if any)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)

        # Compute obs
        self.compute_observations()

        # Compute rewards
        self.compute_rewards(actions)

        # Compute resets
        self.compute_resets()

        # Debug viewer if requested
        if self.viewer and self.debug_viz:
            self._viewer_visualization()

    def reset(self, env_ids=None):
        """
        Executes reset for this task
        Args:
            env_ids (None or tensor): If specified, should be (integer) IDs corresponding to the
                specific env instances of this robot that should be reset
        """
        # Propagate changes via SimStates and SimActions
        self.sim_states.set_actor_root_states_indexed(env_ids=env_ids)
        self.sim_states.set_dof_states_indexed(env_ids=env_ids)
        self.sim_states.clear_contact_forces_indexed(env_ids=env_ids)
        self.sim_actions.deploy_indexed(env_ids=env_ids)

        # Take a sim step if forcing a reset
        if self.force_sim_step_during_reset:
            # Make sure env_ids is either None or equal to number of envs
            assert env_ids is None or len(env_ids) == self.n_envs, \
                "Forcing sim step during reset can only occur if resetting all envs at once!"
            self.gym.simulate(self.sim)

        # Refresh all states
        self.sim_states.refresh(contact_forces=False)

        # Clear the corresponding progress and reset bufs
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _viewer_visualization(self):
        """
        Run any necessary visualizations in the viewer, if active. Useful for debugging and/or teleoperation. By
        default, this results in a no-op
        """
        pass

    def _update_states(self, dt=None):
        """
        Updates the internal states for this task (should update self.states dict)

        NOTE: Assumes simulation has already refreshed states!!

        Args:
            dt (None or float): Amount of sim time (in seconds) that has passed since last update. If None, will assume
                that this is simply a refresh (no sim time has passed) and will not update any values dependent on dt
        """
        # By default, this results in a no-op
        pass

    def _check_contact(self, groupA, groupB, mode="any"):
        """
        Helper function to check whether contact between @groupA and @groupB is occurring.

        If mode is set to "any", this will either check if any contact between any member of @groupA is in contact with
        @groupB, else it will check to make sure all members of @groupA are in contact with all members of @groupB

        Args:
            groupA (int or list): Rigid body ID(s) that correspond to the first group of bodies to check contact
            groupB (int or list): Rigid body ID(s) that correspond to the second group of bodies to check contact
            mode (str): Mode when checking contacts; options are "any" or "all"

        Returns:
            tensor: (n_envs,) (1 / 0) tensor corresponding to whether requested contact is occurring in each env
        """
        # Make sure groupA and groupB are lists
        groupA = groupA if isinstance(groupA, Iterable) else [groupA]
        groupB = groupB if isinstance(groupB, Iterable) else [groupB]

        # Grab all contacts
        contacts = self.gym.get_rigid_contacts(self.sim)

        # Define mapping from groupA, groupB to unique idx
        n_A = len(groupA)
        n_B = len(groupB)
        n_AB = n_A * n_B
        contact_mapping = {(A, B): i * n_B + j for i, A in enumerate(groupA) for j, B in enumerate(groupB)}

        # This tensor will allow us to check whether specific contact requirements are met, and also "sink"
        # any extraneous contact info via the extra added idx
        contacts_info_tensor = torch.zeros(self.n_envs, n_AB + 1, device=self.device)

        # Populate info tensor
        for c in contacts:
            if -1 < c['env0'] < self.n_envs:
                contacts_info_tensor[c['env0'], contact_mapping.get((c['body0'], c['body1']), -1)] = 1
                contacts_info_tensor[c['env0'], contact_mapping.get((c['body1'], c['body0']), -1)] = 1

        # Use mode to compress info tensor
        if mode == "any":
            contacts_info_tensor = torch.sum(contacts_info_tensor[:, :-1])
        elif mode == "all":
            contacts_info_tensor = torch.prod(contacts_info_tensor[:, :-1])
        else:
            raise ValueError(f"Invalid mode specified. Options are 'any' or 'all'; got: {mode}")

        # Return tensor with (binary) contact info for each env
        return torch.where(contacts_info_tensor > 0,
                           torch.ones(self.n_envs, device=self.device),
                           torch.zeros(self.n_envs, device=self.device)).type(torch.bool)

    def compute_observations(self):
        """
        Computes observations for the current sim step

        Returns:
            tensor: (n_env, obs_dim) Observation tensor
        """
        # Refresh states
        self.sim_states.refresh(contact_forces=True)

        # Update internal states
        self._update_states(dt=self.dt)

        # Compute observations interally, concatenate, and then write it at once to the obs buf
        obs_buf = torch.cat(self._compute_observations(), dim=-1)

        # Add noise
        self.obs_buf = obs_buf * (1.0 + self.observation_noise * (-1.0 + 2.0 * torch.randn_like(obs_buf)))

        # Return obs
        return self.obs_buf

    def _compute_observations(self):
        """
        Computes observations for the current sim step

        This is the private method that should be extended by subclasses

        Returns:
            list of tensor: (n_env, any_dim) Observation tensor(s) from different sources
        """
        raise NotImplementedError

    def compute_rewards(self, actions):
        """
        Computes the rewards for the current step. Updates self.rew_buf accordingly

        Args:
            actions (tensor): (n_env, n_actions) Actions executed in sim
        """
        self.rew_buf[:] = self._compute_rewards(actions=actions)

    def _compute_rewards(self, actions):
        """
        Computes the rewards for the current step. Should be implemented by subclass

        Args:
            actions (tensor): (n_env, n_actions) Actions executed in sim

        Returns:
            tensor: (n_envs,) Computed rewards for all envs
        """
        raise NotImplementedError

    def compute_resets(self):
        """
        Computes whether a reset should occur or not. Updates self.reset_buf accordingly
        """
        reset_buf = self._compute_resets()

        # We filter out any resets that had just immediately occurred (i.e.: resets after a single timestep)
        self.reset_buf[:] = torch.where(self._last_reset_buf > 0, torch.zeros_like(reset_buf), reset_buf)

        # Update last reset buffer as this reset buffer
        self._last_reset_buf[:] = self.reset_buf.clone()

    def _compute_resets(self):
        """
        Computes whether a reset should occur or not. Should be implemented by subclass

        Returns:
            tensor: (n_envs,) Computed binary env reset flags for all envs
        """
        raise NotImplementedError

    @property
    def action_modes(self):
        """
        Action modes that this task uses. Should be a subset of
            (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_VEL, gymapi.DOF_MODE_EFFORT)

        Returns:
            set: Mode(s) used for this task's action space
        """
        raise NotImplementedError

    @property
    def action_dim(self):
        """
        Action dimension for this env

        Returns:
            int: Action dim
        """
        raise NotImplementedError

    @property
    def obs_dim(self):
        """
        Observation dimension for this env

        Returns:
        int: Observation dim
        """
        return self.obs_buf.shape[-1]

    @property
    def states_dim(self):
        """
        States dimension for this env (equivalent to ground truth state)

        Returns:
        int: State dim
        """
        return 0

    @property
    def force_sim_step_during_reset(self):
        """
        If set, forces a simulation step during reset
        (usually if states need to be manually set with using GPU pipeline)

        Returns:
            bool: True if forcing sim step during reset
        """
        return False
