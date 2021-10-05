# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
from isaacgym import gymapi
from oscar.utils.torch_utils import tensor_clamp
from collections import Iterable


class Controller:
    """
    Base controller from which all controllers extend from. This class includes basic APIs child controllers
    should adhere to.

    In general, the controller pipeline is as follows:

    received cmd --> clipped --> scaled --> processed through controller --> clipped --> normalized (optional)

    Args:
        command_dim (int): input dimension (i.e.: dimension of received commands)
        input_min (int, float, or array): Minimum values below which received commands will be clipped
        input_max (int, float, or array): Maximum values above which received commands will be clipped
        output_min (int, float, or array): Lower end of range that received commands will be mapped to
        output_max (int, float, or array): Upper end of range that received commands will be mapped to
        control_min (int, float, or array): Minimum control values below which outputted controls will be clipped
        control_max (int, float, or array): Maximum control values above which outputted controls will be clipped
        control_noise (float): Amount of noise to apply. Should be in [0, 1)
        control_dim (int): Outputted control dimension
        device (str): Which device to send all tensors to by default
        normalize_control (bool): Whether or not to normalize outputted controls to (-1, 1) range
    """
    def __init__(
        self,
        command_dim,
        input_min,
        input_max,
        output_min,
        output_max,
        control_min,
        control_max,
        control_dim,
        control_noise,
        device,
        normalize_control=True,
    ):
        # Store device
        self.device = device

        # Store dimensions
        self.command_dim = command_dim
        self.control_dim = control_dim

        # Store limits
        self.input_min = self.nums2tensorarray(nums=input_min, dim=self.command_dim)
        self.input_max = self.nums2tensorarray(nums=input_max, dim=self.command_dim)
        self.output_min = self.nums2tensorarray(nums=output_min, dim=self.command_dim)
        self.output_max = self.nums2tensorarray(nums=output_max, dim=self.command_dim)
        self.control_min = self.nums2tensorarray(nums=control_min, dim=self.control_dim)
        self.control_max = self.nums2tensorarray(nums=control_max, dim=self.control_dim)
        self.control_noise = control_noise
        self.normalize_control = normalize_control

        # Initialize other internal variables
        self.command_scale = None
        self.command_output_transform = None
        self.command_input_transform = None
        self.control_normalization_scale = None
        self.control_input_transform = None

    def scale_command(self, command):
        """
        Clips @command to be within self.input_min and self.input_max, and then re-scale the values to be within
        the range self.output_min and self.output_max
        Args:
            command (tensor): Command to scale
        Returns:
            tensor: Re-scaled command
        """
        # Only calculate command scale once if we havne't done so already
        if self.command_scale is None:
            self.command_scale = abs(self.output_max - self.output_min) / abs(self.input_max - self.input_min)
            self.command_output_transform = (self.output_max + self.output_min) / 2.0
            self.command_input_transform = (self.input_max + self.input_min) / 2.0
        command = tensor_clamp(command, self.input_min, self.input_max)
        transformed_command = (command - self.command_input_transform) * self.command_scale + self.command_output_transform

        return transformed_command

    def postprocess_control(self, control):
        """
        Clips @control to be within range [self.control_min, self.control_max], and optionally normalizes the commands
        to be within range [-1, 1] if self.normalize_control is True. Assumes final dim of @control is the relevant
        control dimension

        Args:
            control (tensor): Raw control computed from controller

        Returns:
            tensor: Clipped and potentially normalized control
        """
        # Add noise
        control = control + (self.control_max - self.control_min) * \
                  self.control_noise * (-1.0 + 2.0 * torch.randn_like(control))

        # Clamp control signal
        pp_control = tensor_clamp(control, self.control_min, self.control_max)

        # Also normalize if requested
        if self.normalize_control:
            # Only calculate control scale once if we havne't done so already
            if self.control_normalization_scale is None:
                self.control_normalization_scale = 2.0 / abs(self.control_max - self.control_min)
                self.control_input_transform = (self.control_max + self.control_min) / 2.0
            pp_control = (pp_control - self.control_input_transform) * self.control_normalization_scale

        return pp_control

    def nums2tensorarray(self, nums, dim):
        """
        Converts input @nums into torch tensor of length @dim. If @nums is a single number, broadcasts input to
        corresponding dimension size @dim before converting into torch tensor

        Args:
            nums (float or array): Numbers to map to tensor
            dim (int): Size of array to broadcast input to

        Returns:
            torch.Tensor: Mapped input numbers
        """
        # Make sure the inputted nums isn't a string
        assert not isinstance(nums, str), "Only numeric types are supported for this operation!"

        out = torch.tensor(nums, device=self.device) if isinstance(nums, Iterable) else torch.ones(dim, device=self.device) * nums

        return out

    def update_goal(self, control_dict, command, env_ids=None, train=False):
        """
        Updates the internal goal based on the inputted command

        Args:
            control_dict (dict): Dictionary of keyword-mapped tensors including relevant control
                information (eef state, q states, etc.)

            command (tensor): Command (specific to controller)

            env_ids (None or tensor): If specified, should be (integer) IDs corresponding to the
                specific env instances of this robot that should be reset

            train (bool): If True, will assume env_ids is None and will NOT index specific goals so we avoid inplace
                operations and so that we can backprop later
        """
        raise NotImplementedError

    def compute_control(self, control_dict):
        """
        Computes low-level controls using internal goal.

        Args:
            control_dict (dict): Dictionary of state tensors including relevant info for controller computation

        Returns:
            tensor: Processed low-level actions
        """
        raise NotImplementedError

    def reset(self, control_dict, env_ids=None):
        """
        Reset the internal vars associated with this controller

        Args:
            control_dict (dict): Dictionary of state tensors including relevant info for controller computation

                Expected keys:
                    eef_state: shape of (N, 13), the (lin_pos, quat_ori, lin_vel, ang_vel) state of the eef body
                    q: shape of (N, N_dof), current joint positions
                    j_eef: shape of (N, 6, N_dof), current jacobian matrix for end effector frame

                Note that N_dof can be greater than control_dim; we assume the first control_dim indexes correspond to
                    the relevant elements to be used for the ik computations

            env_ids (None or tensor): If specified, should be (integer) IDs corresponding to the
                specific env instances of this policy that should be reset
        """
        raise NotImplementedError

    def get_flattened_goals(self):
        """
        Returns the current goal command in a serialized 2D form

        Returns:
            torch.tensor: (N, -1) current goals in this controller
        """
        raise NotImplementedError

    @property
    def goal_dim(self):
        """
        Dimension of the (flattened) goal state for this controller

        Returns:
            int: Flattened goal dimension
        """
        raise NotImplementedError

    @property
    def control_type(self):
        """
        Defines the low-level control type this controller outputs. Should be one of gymapi.DOF_MODE_XXXX

        Returns:
            int: control type outputted by this controller
        """
        raise NotImplementedError

    @property
    def differentiable(self):
        """
        Whether this controller is differentiable (i.e.: backpropable via pytorch) or not

        Returns:
            bool: True if we can take gradients through the full compute_control() method
        """
        raise NotImplementedError
