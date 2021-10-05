# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from isaacgym import gymapi
import torch
from .base_controller import Controller


class JointTorqueController(Controller):
    """
    Joint Torque Controller.

    This controller expects D-DOF commands either in delta form (dq1, dq2, ..., dqD), or absolute form
    (q1, q2, ..., qD), as specified by the @use_delta argument.

    Args:
        input_min (int, float, or array): Minimum values below which received commands will be clipped
        input_max (int, float, or array): Maximum values above which received commands will be clipped
        output_min (int, float, or array): Lower end of range that received commands will be mapped to
        output_max (int, float, or array): Upper end of range that received commands will be mapped to
        control_min (int, float, or array): Minimum control values below which outputted controls will be clipped
        control_max (int, float, or array): Maximum control values above which outputted controls will be clipped
        control_noise (float): Amount of noise to apply. Should be in [0, 1)
        control_dim (int): Outputted control dimension -- should be number of joints from base to eef body frame
        device (str): Which device to send all tensors to by default
        use_delta (bool): Whether to expect received commands to be delta or absolute joint positions
        normalize_control (bool): Whether or not to normalize outputted controls to (-1, 1) range
    """
    def __init__(
        self,
        input_min,
        input_max,
        output_min,
        output_max,
        control_min,
        control_max,
        control_noise,
        control_dim,
        device,
        use_delta=True,
        normalize_control=True,
        **kwargs,                   # hacky way to sink extraneous args
    ):
        # Run super init first
        super().__init__(
            command_dim=control_dim,
            input_min=input_min,
            input_max=input_max,
            output_min=output_min,
            output_max=output_max,
            control_min=control_min,
            control_max=control_max,
            control_noise=control_noise,
            control_dim=control_dim,
            device=device,
            normalize_control=normalize_control,
        )

        # Store internal vars
        self.use_delta = use_delta

        # Initialize internal vars
        self.n_envs = None
        self.goal_torque = None

    def update_goal(self, control_dict, command, env_ids=None, train=False):
        """
        Updates the internal goal (absolute joint torques) based on the inputted joint command

        NOTE: received joints from @control_dict can be greater than control_dim; we assume the first control_dim
            indexes correspond to the relevant elements to be used for joint torque goal setting

        Args:
            control_dict (dict): Dictionary of keyword-mapped tensors including relevant control
                information (eef state, q states, etc.)

                Expected keys:
                    eef_state: shape of (N, 13), the (lin_pos, quat_ori, lin_vel, ang_vel) state of the eef body

            command (tensor): D-DOF joint torque command -- should be (dq1, dq2, ..., dqD), or absolute form
                (q1, q2, ..., qD) if self.use_delta is False.

            env_ids (None or tensor): If specified, should be (integer) IDs corresponding to the
                specific env instances of this robot that should be reset

            train (bool): If True, will assume env_ids is None and will NOT index specific goals so we avoid inplace
                operations and so that we can backprop later
        """
        # Scale the commands appropriately
        cmd = self.scale_command(command)

        # Set n_envs, goal_pos, and goal_ori if we haven't done so already
        if self.n_envs is None:
            self.n_envs = command.shape[0]
            self.goal_torque = torch.zeros(self.n_envs, self.control_dim, device=self.device)

        # If we're training, make sure env_ids is None
        if train:
            assert env_ids is None or len(env_ids) == self.n_envs, "When in training mode, env_ids must be None or len of n_envs!"
            # Directly set goals
            self.goal_torque = self.goal_torque + cmd if self.use_delta else cmd
        else:
            # If env_ids is None, we update all the envs
            if env_ids is None:
                env_ids = torch.arange(start=0, end=self.n_envs, device=self.device, dtype=torch.uint32)

            # Update goal
            self.goal_torque[env_ids] = self.goal_torque[env_ids] + cmd[env_ids] if self.use_delta else cmd[env_ids]

    def compute_control(self, control_dict):
        """
        Computes low-level joint torque controls.

        Since we are directly using joint-torque control, this simply is equivalent to returning the
        internal goal state

        Args:
            control_dict (dict): Dictionary of state tensors including relevant info for controller computation

                Expected keys:
                    eef_state: shape of (N, 13), the (lin_pos, quat_ori, lin_vel, ang_vel) state of the eef body

        Returns:
            tensor: Processed low-level joint position control actions
        """
        # Post-process internal goal (clipping + normalization)
        u = self.postprocess_control(self.goal_torque)

        # Return the control joint positions
        return u

    def reset(self, control_dict, env_ids=None):
        """
        Reset the internal vars associated with this controller

        Args:
            control_dict (dict): Dictionary of state tensors including relevant info for controller computation

                Expected keys:
                    eef_state: shape of (N, 13), the (lin_pos, quat_ori, lin_vel, ang_vel) state of the eef body

            env_ids (None or tensor): If specified, should be (integer) IDs corresponding to the
                specific env instances of this policy that should be reset
        """
        # Clear n_envs, goal if we're now controlling a new set of envs
        n_cmds = control_dict["eef_state"].shape[0]
        if self.n_envs != n_cmds:
            self.n_envs = None
            self.goal_torque = None
        # Reset corresponding envs to current positions
        cmd = torch.zeros(n_cmds, self.command_dim, device=self.device)
        self.update_goal(
            control_dict=control_dict,
            command=cmd,
            env_ids=env_ids
        )

    def get_flattened_goals(self):
        """
        Returns the current goal command in a serialized 2D form

        Returns:
            torch.tensor: (N, -1) current goals in this controller
        """
        return self.goal_torque

    @property
    def goal_dim(self):
        # This is the same as the control dimension
        return self.control_dim

    @property
    def control_type(self):
        # This controller outputs joint positions
        return gymapi.DOF_MODE_EFFORT

    @property
    def differentiable(self):
        # We can backprop through all computations
        return True
