# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from isaacgym import gymapi
import torch
from oscar.utils.torch_utils import quat_mul, quat2mat, orientation_error, axisangle2quat
from .base_controller import Controller


class IKController(Controller):
    """
    Inverse Kinematics Controller. Leverages position-based end effector control using IK damped least squares.

    NOTE: Currently only executes a single iteration of DLS.

    This controller expects 6DOF delta commands (dx, dy, dz, dax, day, daz), where the delta orientation
    commands are in axis-angle form, and outputs low-level joint position commands.

    Parameters (in this case, only damping) can either be set during initialization or provided from an external source;
    if the latter, the control_dict should include "damping" as one of its keys

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
        damping (int, float, or array): Damping to apply when solving damped least squares. Low values result in more
            accurate values but less robustness to singularities, and vise versa for large values
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
        damping=0.05,
        normalize_control=True,
        **kwargs,                   # hacky way to sink extraneous args
    ):
        # Run super init first
        super().__init__(
            command_dim=6,
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

        # Store gains
        self.damping = self.nums2tensorarray(nums=damping, dim=6)

        # Initialize internal vars
        self.n_envs = None
        self.goal_pos = None
        self.goal_ori_mat = None

    def update_goal(self, control_dict, command, env_ids=None, train=False):
        """
        Updates the internal goal (ee pos and ee ori mat) based on the inputted delta command

        Args:
            control_dict (dict): Dictionary of keyword-mapped tensors including relevant control
                information (eef state, q states, etc.)

                Expected keys:
                    eef_state: shape of (N, 13), the (lin_pos, quat_ori, lin_vel, ang_vel) state of the eef body

            command (tensor): 6DOF EEF command -- should be (dx, dy, dz, dax, day, daz), where the delta orientation
                commands are in axis angle form

            env_ids (None or tensor): If specified, should be (integer) IDs corresponding to the
                specific env instances of this robot that should be reset

            train (bool): If True, will assume env_ids is None and will NOT index specific goals so we avoid inplace
                operations and so that we can backprop later
        """
        # Get useful state info
        ee_pos = control_dict["eef_state"][:, :3]
        ee_quat = control_dict["eef_state"][:, 3:7]

        # Scale the commands appropriately
        dpose = self.scale_command(command)

        # Set n_envs, goal_pos, and goal_ori if we haven't done so already or if we need to update values
        if self.n_envs is None or command.shape[0] != self.n_envs:
            self.n_envs = command.shape[0]
            self.goal_pos = torch.zeros(self.n_envs, 3, device=self.device)
            self.goal_ori_mat = torch.zeros(self.n_envs, 3, 3, device=self.device)

        # If we're training, make sure env_ids is None
        if train:
            assert env_ids is None or len(env_ids) == self.n_envs, "When in training mode, env_ids must be None or len of n_envs!"
            # Directly set goals
            self.goal_pos = ee_pos + dpose[:, :3]
            self.goal_ori_mat = quat2mat(quat_mul(axisangle2quat(dpose[:, 3:6]), ee_quat))
        else:
            # If env_ids is None, we update all the envs
            if env_ids is None:
                # DON'T use individual indexes since this breaks backpropping
                env_ids = torch.arange(start=0, end=self.n_envs, device=self.device, dtype=torch.uint32)

            # Update specific goals
            self.goal_pos[env_ids] = ee_pos[env_ids] + dpose[env_ids, :3]
            self.goal_ori_mat[env_ids] = quat2mat(quat_mul(axisangle2quat(dpose[env_ids, 3:6]), ee_quat[env_ids]))

    def compute_control(self, control_dict):
        """
        Computes low-level joint position controls using internal eef goal pos / ori.

        Args:
            control_dict (dict): Dictionary of state tensors including relevant info for controller computation

                Expected keys:
                    eef_state: shape of (N, 13), the (lin_pos, quat_ori, lin_vel, ang_vel) state of the eef body
                    q: shape of (N, N_dof), current joint positions
                    j_eef: shape of (N, 6, N_dof), current jacobian matrix for end effector frame

                Note that N_dof can be greater than control_dim; we assume the first control_dim indexes correspond to
                    the relevant elements to be used for the ik computations

        Returns:
            tensor: Processed low-level joint position control actions
        """
        #  Extract relevant values from the control dict
        q = control_dict["q"][:, :self.control_dim].to(self.device)
        j_eef = control_dict["j_eef"].to(self.device)
        ee_pos = control_dict["eef_state"][:, :3].to(self.device)
        ee_quat = control_dict["eef_state"][:, 3:7].to(self.device)
        ee_vel = control_dict["eef_state"][:, 7:].to(self.device)

        # Possibly grab damping as well, otherwise use internal damping
        damping = self.nums2tensorarray(nums=control_dict["damping"], dim=6) if \
            "damping" in control_dict else self.damping

        # Solve IK problem
        # See https://www.researchgate.net/publication/273166356_Inverse_Kinematics_a_review_of_existing_techniques_and_introduction_of_a_new_fast_iterative_solver#pf1d
        # for a helpful overview

        # Calculate error
        pos_err = self.goal_pos - ee_pos
        ori_err = orientation_error(self.goal_ori_mat, quat2mat(ee_quat))
        err = torch.cat([pos_err, ori_err], dim=-1).unsqueeze(-1)

        # Solve DLS
        j_eef_T = torch.transpose(j_eef, 1, 2)
        lmda = (torch.eye(6).to(self.device) * (damping ** 2)).unsqueeze(0)
        u_dq = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmda)) @ err
        u_q = q + u_dq.squeeze(-1)

        # Post-process torques (clipping + normalization)
        u_q = self.postprocess_control(u_q)

        # Return the control joint positions
        return u_q

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
        # Clear n_envs, goal pos, and goal ori if we're now controlling a new set of envs
        n_cmds = control_dict["eef_state"].shape[0]
        if self.n_envs != n_cmds:
            self.n_envs = None
            self.goal_pos = None
            self.goal_ori = None
        # Reset corresponding envs to current positions
        self.update_goal(
            control_dict=control_dict,
            command=torch.zeros(n_cmds, 6),
            env_ids=env_ids
        )

    def get_flattened_goals(self):
        """
        Returns the current goal command in a serialized 2D form

        Returns:
            torch.tensor: (N, -1) current goals in this controller
        """
        return torch.cat([self.goal_pos, self.goal_ori_mat.view(-1, 9)], dim=-1)

    @property
    def goal_dim(self):
        # This is 3 from pos goals + 9 from the ee ori goal
        return 12

    @property
    def control_type(self):
        # This controller outputs joint positions
        return gymapi.DOF_MODE_POS

    @property
    def differentiable(self):
        # We can backprop through all IK computations
        return True
