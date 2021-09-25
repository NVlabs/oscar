from isaacgym import gymapi
import torch
from oscar.controllers import *


class Policy:
    """
    Base class for all policies. Can optionally include a controller

    Args:
        agent_config (dict): agent tconfig that includes relevant agent-specific information
        obs_dim (int): Size of observation space
        n_envs (int): Number of environements active in sim
        device (str): Device to map tensors to
        normalize_actions (bool): Whether to normalize outputted actions to be in [-1, 1]
    """
    def __init__(
        self,
        agent_config,
        obs_dim,
        n_envs,
        device,
        normalize_actions=True,
    ):
        # Store information
        self.agent_config = agent_config
        self.obs_dim = obs_dim
        self.n_envs = n_envs
        self.device = device
        self.normalize_actions = normalize_actions

        # Keep track of whether we're training
        self.is_train = False

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
        raise NotImplementedError

    def train(self):
        """
        Sets internal mode to train
        """
        self.is_train = True

    def eval(self):
        """
        Sets internal mode to evaluation
        """
        self.is_train = False

    @property
    def input_dim(self):
        """
        Defines input dimension for this policy controller.

        Returns:
            int: Input action dimension
        """
        raise NotImplementedError

    @property
    def output_dim(self):
        """
        Defines output dimension for this policy controller.

        Returns:
            int: Output action dimension
        """
        raise NotImplementedError
