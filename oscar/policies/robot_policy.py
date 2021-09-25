from isaacgym import gymapi
import torch
from oscar.controllers import *
from .policy import Policy
from oscar.models.control_models import DynamicsModel
import torch
from collections import OrderedDict

# Keep list of possible controllers for sanity checking
CONTROLLER_MODE_MAPPING = {
    "osc": gymapi.DOF_MODE_EFFORT,
    "oscar": gymapi.DOF_MODE_EFFORT,
    "ik": gymapi.DOF_MODE_POS,
    "ik_diff": gymapi.DOF_MODE_POS,
    "joint_vel": gymapi.DOF_MODE_POS,
    "joint_pos": gymapi.DOF_MODE_POS,
    "joint_pos_tor": gymapi.DOF_MODE_EFFORT,
    "joint_tor": gymapi.DOF_MODE_EFFORT,
    "joint_tor_delan": gymapi.DOF_MODE_EFFORT,
}
VALID_CONTROLLERS = set(CONTROLLER_MODE_MAPPING.keys())


class RobotArmPolicy(Policy):
    """
    Policy for a robot arm. Includes a controller to directly actuate the robot arm.

    Args:
        agent_config (dict): agent config that includes relevant agent-specific information
        obs_dim (int): Size of observation space
        extrinsics_dim (int): Size of extrinsics dynamic parameter space
        n_envs (int): Number of environments active in sim
        device (str): Device to map tensors to
        normalize_actions (bool): Whether to normalize outputted actions to be in [-1, 1]
        control_freq (int): Expected control frequency for this policy, in Hz
        control_steps_per_policy_step (int): How many control steps are executed in between a single policy step
        controller_config (None or dict): Configuration to use for controller. If None,
            a default one will be used. Otherwise, will update the default with any keys specified in this dict.
    """
    def __init__(
        self,
        agent_config,
        obs_dim,
        extrinsics_dim,
        n_envs,
        device,
        normalize_actions=True,
        control_freq=20,
        control_steps_per_policy_step=1,
        controller_config=None,
    ):
        # Run super init
        super().__init__(
            agent_config=agent_config,
            obs_dim=obs_dim,
            n_envs=n_envs,
            device=device,
            normalize_actions=normalize_actions,
        )

        # Grab relevant info from robot
        self.dof_upper_limits = torch.tensor(self.agent_config["dof_upper_limits"], device=self.device)
        self.dof_lower_limits = torch.tensor(self.agent_config["dof_lower_limits"], device=self.device)
        self.dof_default = torch.tensor(self.agent_config["dof_default"], device=self.device)
        self.velocity_limits = torch.tensor(self.agent_config["dof_max_velocities"], device=self.device)
        self.effort_limits = torch.tensor(self.agent_config["dof_max_efforts"], device=self.device)
        self.dof_arm = self.agent_config["dof_arm"]
        self.eef_action_dim = self.agent_config["eef_action_dim"]

        # Store information
        self.extrinsics_dim = extrinsics_dim
        self.differentiate_controller = None                                # Inferred from controller name
        self._learned_models = OrderedDict()                                # Filled during load_controller
        self.control_freq = control_freq
        self.control_steps_per_policy_step = control_steps_per_policy_step
        self.control_step = torch.zeros(self.n_envs, device=self.device, requires_grad=False, dtype=torch.int32)
        self.controller_config = controller_config

        # Internal vars
        self._eef_command = None                            # Filled during get_control() steps

        # Initialize controller
        self.controller = self.load_controller()

    def load_controller(self):
        """
        Loads a controller into this policy
        """
        # Setup configuration
        if self.controller_config is None:
            # We use the default config
            self.controller_config = self.default_controller_config
        else:
            # We want to modify the default control config with the current control config
            controller_config = self.default_controller_config
            controller_config.update(self.controller_config)
            self.controller_config = controller_config

        # Make sure controller type is valid
        assert self.controller_type.lower() in VALID_CONTROLLERS, \
            f"Invalid controller specified. Valid options: {VALID_CONTROLLERS}\ngot: {self.controller_type.lower()}"
        # Define controller class based on type
        if "osc" in self.controller_type.lower():
            controller_cls = OSCController
        elif "ik" in self.controller_type.lower():
            controller_cls = IKController
        elif "joint_pos" in self.controller_type.lower():
            controller_cls = JointPositionController
        elif "joint_tor" in self.controller_type.lower():
            controller_cls = JointTorqueController
        elif "joint_vel" in self.controller_type.lower():
            controller_cls = JointVelocityController
        else:
            raise ValueError(f"No valid controller with name {self.controller_type.lower()}; "
                             f"valid options must include osc, ik, or joint_pos")
        # Get whether we should differentiate the controller
        self.differentiate_controller = "diff" in self.controller_type.lower()

        # Create the controller
        controller = controller_cls(**self.controller_config)

        # Add learned models based on name
        if "oscar" in self.controller_type.lower():
            delan_args = self.controller_config["delan"]
            delan_args["n_dof"] = self.dof_arm
            self._learned_models["delan"] = DynamicsModel(
                delan_args=delan_args,
                max_velocities=self.velocity_limits,
                max_efforts=self.controller_config["control_max"],
                normalized_efforts=self.normalize_actions,
                obs_dim=self.obs_dim,
                extrinsics_dim=self.extrinsics_dim,
                history_dim=self.agent_config["history_length"],
                device=self.device,
            )

        # Return the controller
        return controller

    def get_control(self, control_dict, command=None):
        """
        Controls this robot for a single timestep in sim. This method should deploy the outputted controller
        actions in sim.

        Args:
            control_dict (dict): Keyword-mapped relevant information necessary for controller
                computation

            command (None or tensor): If specified, will update the controller's internal goal before
                executing a controller step
        """
        # Make sure command is of the correct dimension if we received a new command
        inputs = {}
        if command is not None:
            assert command.shape[-1] == self.input_dim, \
                f"Agent action dim is incorrect, expected {self.input_dim}, got {command.shape[0]}"
            # Separate eef action
            self._eef_command = command[..., self.input_dim - self.eef_action_dim:]
            arm_command = command[..., :self.input_dim - self.eef_action_dim]

            # Compose inputs to send to model forward pass
            inputs.update({
                "command": arm_command
            })

            # Add any learned components to the control dict
            for name, model in self.learned_models.items():
                if model.is_before_controller:
                    control_dict.update(model(control_dict=control_dict, **inputs))

            # Update arm command if it exists in the control dict
            arm_command = control_dict.get("command", arm_command)
            inputs["command"] = arm_command

            self.controller.update_goal(
                control_dict=control_dict,
                command=arm_command,
                env_ids=None if self.is_train else (self.control_step == 0).nonzero(as_tuple=True)[0],
                train=self.is_train,
            )

        # Calculate arm control
        arm_action = self.controller.compute_control(control_dict=control_dict)

        # Add compensation torques if using delan
        if "delan" in self.learned_models:
            arm_action = arm_action + control_dict["torques_coriolis"] + control_dict["torques_gravity"]

        # Add arm control to inputs dict
        inputs.update({
            "control": arm_action
        })
        for name, model in self.learned_models.items():
            if not model.is_before_controller:
                control_dict.update(model(control_dict=control_dict, **inputs))

        # Increment all controller steps
        self.control_step = (self.control_step + 1) % self.control_steps_per_policy_step

        # Return action
        return torch.cat([arm_action, self._eef_command], dim=-1)

    def reset(self, obs_dict, env_ids=None):
        """
        Resets this policy

        Args:
            obs_dict (dict): Keyword-mapped relevant information necessary for action computation.

                Expected keys:
                    control_dict (dict): Dictionary of state tensors including relevant info for controller computation

            env_ids (None or tensor): If specified, should be (integer) IDs corresponding to the
                specific env instances of this policy that should be reset
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.n_envs, device=self.device, dtype=torch.uint32)

        # Reset control steps
        self.control_step[env_ids] = 0

        # Only reset controllers associated with specific envs
        self.controller.reset(control_dict=obs_dict["control_dict"], env_ids=env_ids)

    @property
    def default_controller_config(self):
        """
        Default control configuration to use for this policy

        Returns:
            dict: Keyword-mapped values for the default controller configuration. Should, at the minimum, include the
                following keys:

                type (str): controller type, current options are {osc, ik, joint_pos}
                input_min (int, float, or array): Minimum values below which received commands will be clipped
                input_max (int, float, or array): Maximum values above which received commands will be clipped
                output_min (int, float, or array): Lower end of range that received commands will be mapped to
                output_max (int, float, or array): Upper end of range that received commands will be mapped to
                control_min (int, float, or array): Minimum control values below which outputted controls will be clipped
                control_max (int, float, or array): Maximum control values above which outputted controls will be clipped
                control_dim (int): Outputted control dimension
                device (str): Which device to send all tensors to by default
                normalize_control (bool): Whether or not to normalize outputted controls to (-1, 1) range
                actor_loss_scale (float): How much to scale actor loss rate by (taken in conjunction with all learned models'
                    values for this)
        """
        # Setup default limits based on current controller config
        controller_type = "joint_pos" if self.controller_config is None else self.controller_config["type"]

        # Define and return defaults based on type
        shared_dict = {
            "type": controller_type,
            "input_min": -1.,
            "input_max": 1.,
            "control_dim": self.dof_arm,
            "control_freq": self.control_freq,
            "device": self.device,
            "normalize_control": self.normalize_actions,
            "actor_loss_scale": 1.0,
            "use_motion_predictor": False,
            "use_motion_compensator": False,
        }
        controller_dict = None
        if "osc" in controller_type.lower():
            controller_dict = {
                "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "control_min": -self.effort_limits[:self.dof_arm],
                "control_max": self.effort_limits[:self.dof_arm],
                "kp": 150.0,
                "damping_ratio": 1.0,
                "kp_null": 10.0,
                "rest_qpos": self.dof_default[:self.dof_arm],
                "decouple_pos_ori": False,
            }
        elif "ik" in controller_type.lower():
            controller_dict = {
                "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "control_min": self.dof_lower_limits[:self.dof_arm],
                "control_max": self.dof_upper_limits[:self.dof_arm],
                "damping": 0.05,
            }
        elif "joint_vel" in controller_type.lower():
            controller_dict = {
                "output_min": [-1.0] * self.dof_arm,
                "output_max": [1.0] * self.dof_arm,
                "control_min": self.dof_lower_limits[:self.dof_arm],
                "control_max": self.dof_upper_limits[:self.dof_arm],
            }
        elif "joint_pos" in controller_type.lower():
            if "tor" in controller_type.lower():
                controller_dict = {
                    "input_min": self.dof_lower_limits[:self.dof_arm],
                    "input_max": self.dof_upper_limits[:self.dof_arm],
                    "output_min": self.dof_lower_limits[:self.dof_arm],
                    "output_max": self.dof_upper_limits[:self.dof_arm],
                    "control_min": -self.effort_limits[:self.dof_arm],
                    "control_max": self.effort_limits[:self.dof_arm],
                    "use_delta": False,
                    "use_torques": True,
                }
            else:
                controller_dict = {
                    "input_min": self.dof_lower_limits[:self.dof_arm],
                    "input_max": self.dof_upper_limits[:self.dof_arm],
                    "output_min": self.dof_lower_limits[:self.dof_arm],
                    "output_max": self.dof_upper_limits[:self.dof_arm],
                    "control_min": self.dof_lower_limits[:self.dof_arm],
                    "control_max": self.dof_upper_limits[:self.dof_arm],
                    "use_delta": False,
                    "use_torques": False,
                }
        elif "joint_tor" in controller_type.lower():
            controller_dict = {
                "output_min": -self.effort_limits[:self.dof_arm],
                "output_max": self.effort_limits[:self.dof_arm],
                "control_min": -self.effort_limits[:self.dof_arm],
                "control_max": self.effort_limits[:self.dof_arm],
                "use_delta": False,
            }
        else:
            raise ValueError(f"No valid controller with name {controller_type.lower()};"
                             f"valid options must include osc, ik, or joint_pos")

        # Combine dicts and return the default config
        shared_dict.update(controller_dict)

        return shared_dict

    def train(self):
        """
        Sets internal mode to train
        """
        # Make sure all learned models are set appropriately
        for model in self.learned_models.values():
            model.eval()

        # Run super
        super().train()

    def eval(self):
        """
        Sets internal mode to evaluation
        """
        # Make sure all learned models are set appropriately
        for model in self.learned_models.values():
            model.eval()
        # Run super
        super().eval()

    @property
    def goal_dim(self):
        """
        Goal dimension used internally by this controller.

        Returns:
            int: Goal dimension associated with this controller
        """
        return self.controller.goal_dim

    @property
    def input_dim(self):
        """
        Defines input dimension for this policy controller.

        Returns:
            int: Input action dimension
        """
        # Input dimension is controller input dimension + eef dimension
        return self.controller.command_dim + self.eef_action_dim

    @property
    def output_dim(self):
        """
        Defines output dimension for this policy controller.

        Returns:
            int: Output action dimension
        """
        # Input dimension is controller output dimension + eef dimension
        return self.controller.control_dim + self.eef_action_dim

    @property
    def controller_type(self):
        """
        Defines the controller type for this policy

        Returns:
            str: Name of the controller being used
        """
        return self.controller_config["type"]

    @property
    def learned_models(self):
        """
        Grabs all relevant learned components for this controller.

        Returns:
            dict: Keyword-mapped learned components for this policy, where each mapped value is a ControlModel
        """
        return self._learned_models

    @property
    def actor_loss_scale(self):
        """
        Scales the actor loss depending on the type of robot policy model being used.

        Returns:
            float: Scaling factor to multiply actor loss by in RL pipeline
        """
        # Initialize scale
        scale = self.controller_config["actor_loss_scale"]
        # Multiply all scaling factors from all control models
        for model in self.learned_models.values():
            scale *= model.actor_loss_scale

        # Return this value
        return scale
