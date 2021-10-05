# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""
Set of custom control-related models
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from oscar.utils.torch_utils import orientation_error_from_quat, axisangle2quat
from oscar.controllers.osc import OSCController
from oscar.utils.torch_custom import NonStrictModule, maybe_no_grad
from oscar.models.delan import DeepLagrangianNetwork
from rl_games.algos_torch import model_builder
from rl_games.common import object_factory
from rl_games.algos_torch.network_builder import A2CBuilder
from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.common import tr_helpers
import oscar.utils.macros as macros
import gym


def load_checkpoint(model, filename):
    print("=> loading checkpoint for model {}: '{}'".format(type(model).__name__, filename))
    state = torch.load(filename)
    return state


# Define base model class
class ControlModel(NonStrictModule):
    """
    Basic interface for implementing a custom control model

    Args:
        obs_dim (int): Size of observation space for the current environment
        device (str): Device to map all (instantiated) tensors to using this loss
    """
    def __init__(self, obs_dim, device):
        # Run super init first
        super().__init__()

        # Store device
        self.obs_dim = obs_dim
        self.device = device

        # Flag to keep track of whether we loaded a state dict or not
        self._loaded_checkpoint = False

    def __call__(self, control_dict, **kwargs):
        """
        Calculates relevant keyword-mapped model output(s). Should be implemented by subclass.

        Args:
            control_dict (dict): Keyword-mapped values relevant for controller computations
            kwargs (any): Any additional relevant keyword values for computing this loss

        Returns:
            dict: Keyword-mapped outputs from this model
        """
        raise NotImplementedError

    def prepare_for_train_loop(self):
        """
        Conduct any necessary steps to prepare for a training loop (e.g.: prepping a dataset)
        """
        pass

    def pre_env_step(self, control_dict, train=False, **kwargs):
        """
        Calculates any relevant values before an environment step. Should be implemented by subclass.
        train (bool): Whether we're currently training or in eval mode

        Args:
            control_dict (dict): Keyword-mapped values relevant for controller computations
            kwargs (any): Any additional relevant keyword values for computing values (e.g.: rewards, dones, etc.)

        Returns:
            dict: Keyword-mapped outputs from this model
        """
        return {}

    def post_env_step(self, control_dict, train=False, **kwargs):
        """
        Calculates any relevant values after an environment step. Should be implemented by subclass.
        train (bool): Whether we're currently training or in eval mode

        Args:
            control_dict (dict): Keyword-mapped values relevant for controller computations
            kwargs (any): Any additional relevant keyword values for computing values (e.g.: rewards, dones, etc.)

        Returns:
            dict: Keyword-mapped outputs from this model
        """
        return {}

    @autocast(enabled=macros.MIXED_PRECISION)
    def loss(self, control_dict, **kwargs):
        """
        Calculates loss for this model. Should be implemented by subclass.

        Args:
            control_dict (dict): Keyword-mapped values relevant for controller computations
            kwargs (any): Any additional relevant keyword values for computing this loss

        Returns:
            tensor: scalar loss value
        """
        raise NotImplementedError

    def gradient_step(self, loss, retain_graph=False):
        """
        Computes a gradient step for this model (calc gradients + optimizer step). Should be implemented by subclass.

        Args:
            loss (tensor): Scalar loss to backprop with
            retain_graph (bool): If set, will retain computation graph when taking gradient step
        """
        raise NotImplementedError

    def load_state_dict(self, state_dict, strict=True):
        # Override superclass method so that we only load a state dict once
        if not self._loaded_checkpoint:
            super().load_state_dict(state_dict=state_dict, strict=strict)
            # set flag so we don't load it again in the future
            self._loaded_checkpoint = True

    @property
    def actor_loss_scale(self):
        """
        Scales the loss to be applied to the actor in an RL pipeline. This may be useful to prevent instabilities when
        training end-to-end both a dynamics model and an RL agent.

        Returns:
            float: Scaling factor to multiply actor loss by in RL pipeline
        """
        return 1.0

    @property
    def train_with_actor_loss(self):
        """
        Should return True if this model is to be trained with the actor loss. (Separate actor-critic method should
        take care of actually verifying that the loss backprops into this model's learned parameters). False by default.

        Returns:
             bool: Whether to train this model with actor loss from an external Actor-Critic method
        """
        return False

    @property
    def trainable_parameters(self):
        """
        Returns list of trainable (i.e.: unfrozen) parameters that belong to this network.
        By default, list(this is self.model.parameters())

        Returns:
            list: List of trainable parameters
        """
        return list(self.parameters())

    @property
    def is_before_controller(self):
        """
        Whether this control model should have its forward pass executed before or after the controller step

        Returns:
            bool: Whether this model should hvae its forward pass deployed before or after the controller step
        """
        raise NotImplementedError


class DynamicsModel(ControlModel):
    """
    Class for modeling mass matrix. Utilizes DeLaN method for learning mass matrix

    Args:
        delan_args (dict): Relevant arguments to pass to DeLaN class constructor. Expected arguments are as follows:

            - n_dof (int): Degrees of freedom to model in mass matrix
            - n_width (int): Size of each hidden layer in network
            - n_depth (int): Number of hidden layers in network
            - embed_q (bool): If True, will map q to cos(q) and sin(q) before passing it to the learned networks
            - use_ground_truth_mm (bool): If True, will return ground truth mass matrix during forward call
            - use_extrinsics (bool): If True, will map extrinsic dynamics params to latent space and pass as input
            - extrinsics_with_residual (bool): If True, will use extrinsics with residual
            - bootstrap_extrinsics (bool): If True, will bootstrap extrinsics from history of torque / q states
            - steps_per_extrinsics_update (int): How many steps to run before updating the extrinsics network values
            - extrinsics_net_kwargs (dict): Keyword arguments to pass to extrinsics net constructor
                - mlp_hidden_dims (list of int): Extrinsics embedding network hidden layer sizes if not bootstrapping extrinsics,
                    or the layers for initially embedding state-control vectors
                - cnn_input_channels (list of int): Only relevant for bootstrapping extrinsics. Conv1d input channel sizes
                - cnn_output_channels (list of int): Only relevant for bootstrapping extrinsics. Conv1d output channel sizes
                - cnn_kernels (list of int): Only relevant for bootstrapping extrinsics. Conv1d kernel sizes
                - cnn_strides (list of int): Only relevant for bootstrapping extrinsics. Conv1d stride sizes
            - extrinsics_latent_dim: (int): Extrinsics embedding network output dimension
            - use_compensation_torques (bool): If True, will output compensation torques to add to policy
            - diagonal_epsilon (int): Constant to add to mass matrix diagonal
            - activation (str): Activation function to use. Options are {SoftPlus, ReLu, Cos, Linear}
            - diag_activation (str): Activation function to use for diagonal MM block.
                Options are {SoftPlus, ReLu, Cos, Linear}
            - b_init (float): Initialization scale for bias network elements
            - w_init (str): Initialization scheme for network weights. Options are {xavier_normal, orthogonal, sparse}
            - g_hidden (float): Std to use when initializing normal distribution of hidden layer network weights
            - g_output (float): Std to use when initializing normal distribution of output layer network weights
            - lr (float): Learning rate to use per optimization step
            - weight_decay (float): Weight decay to use during training
            - max_grad_norm (float): Maximium allowable gradient for training this model
            - loss_threshold (float): Loss value above which actor losses will be attenuated (for more stable training)
            - train_with_actor_loss (bool): If True, will train this model with the actor loss in addition to the
                DeLaN-specific loss
            - train_with_forward_loss (bool): If True, will add forward qdd regression loss
            - pretrained_model (None or str): If set, will load a pretrained model from the specified fpath
                (should be .pth file that corresponds to rlg_games structure)
            - learn_residual (bool): If True, will include auxiliary learned residual model(s) that
                will leverage prior mass matrices to dynamic model components
            - max_residual_magnitude (float): Maximum percentage of MM to modify base MM via residual.
                If not using exponential residual, then this value should be in (0.0 - 1.0)
                If using exponential residual, then this value should be in (0.0 - X), sets limits for (exp(-X), exp(X))
            - use_exponential_residual (bool): If True, will pass the residual through an exp() block [occurs after
                tanh / clipping] and directly multiply the base model with this value (NOT additive)
            - use_tanh_residual_output (bool): If True, will enforce residual magnitude limit via Tanh activation.
                Otherwise, will use hard clipping
            - n_width_residual (int): Size of each hidden layer in residual network
            - n_depth_residual (int): Number of hidden layers in residual network
            - b_init_residual (float): Initialization scale for bias residual network elements
            - b_diag_init_residual (float): Initialization scale for bias diagonal MM residual network elements
            - freeze_base (bool): If True, will keep delan (base model) in eval mode, so no training steps occur
            - freeze_residual (bool or list of bool): If True, will keep delan (residual model(s)) in eval mode,
                so no training steps occur. Number: Number of residuals is inferred from this array

        max_velocities (tensor): Maximum velocities to be used during the loss computation
        max_efforts (tensor): Maximum efforts to be used during the loss computation
        normalized_efforts (bool): If True, will interpret received controls as being normalized to (-1, 1) when
            calculating delan loss
        obs_dim (int): Size of observation space for the current environment
        extrinsics_dim (int): Size of extrinsics dynamic parameter space for the current environment
        history_dim (int): Size of history buffer of q, qd, etc. values
        device (str): Device to map all (instantiated) tensors to using this loss

    """
    def __init__(self, delan_args, max_velocities, max_efforts, normalized_efforts, obs_dim, extrinsics_dim, history_dim, device):
        # Run super
        super().__init__(obs_dim=obs_dim, device=device)

        # Create model and other relevant values
        self.model = DeepLagrangianNetwork(
            obs_dim=obs_dim,
            extrinsics_dim=extrinsics_dim,
            history_dim=history_dim,
            device=self.device,
            **delan_args
        )

        # Load pre-trained if we have one
        if delan_args["pretrained_model"] is not None:
            ckpt = load_checkpoint(self, delan_args["pretrained_model"])
            self.load_state_dict(ckpt["delan"])

        # Determine if we're fully freezing weights or not
        self._trainable_params = []
        all_residuals_frozen = sum(self.model.freeze_residual) == len(self.model.freeze_residual)
        self.freeze = self.model.freeze_base and (all_residuals_frozen or not self.model._learn_residual) and (self.model.freeze_extrinsics or not self.model._use_extrinsics)
        if self.freeze:
            # Immediately set model in eval mode, and no need to create optimizer
            self.eval()
            self.optimizer = None
        else:
            # Compose learnable params
            if not self.model.freeze_base:
                for net in self.model.base_nets:
                    self._trainable_params += list(net.parameters())
            if not self.model.freeze_extrinsics:
                for net in self.model.extrinsics_nets:
                    self._trainable_params += list(net.parameters())
            if self.model._learn_residual:
                for i, freeze_residual in enumerate(self.model.freeze_residual):
                    if not freeze_residual:
                        for net in self.model.residual_nets[i]:
                            self._trainable_params += list(net.parameters())
            # We only create the optimizer if we're not freezing the model
            self.optimizer = torch.optim.Adam(
                self._trainable_params,
                lr=delan_args["lr"],
                weight_decay=delan_args["weight_decay"],
            )
        self.max_velocities = max_velocities
        self.max_efforts = max_efforts
        self.max_grad_norm = delan_args["max_grad_norm"]
        self.delan_args = delan_args
        self.normalized_efforts = normalized_efforts
        self.dof = self.model.n_dof
        self.train_with_forward_loss = delan_args["train_with_forward_loss"]
        self.steps_per_extrinsics_update = delan_args["steps_per_extrinsics_update"]
        self.extrinsics_update_ctr = 0
        self.extrinsics_dict_cache = None
        self.use_compensation_torques = delan_args["use_compensation_torques"]
        self.use_ground_truth_mm = delan_args["use_ground_truth_mm"]
        self._cache = {}

        # Internal vars
        self._current_mean_loss = torch.tensor(1.0e+10, device=self.device)

        # Send model to appropriate device
        self.model.to(self.device)

    def __call__(self, control_dict, **kwargs):
        """
        Calculate mass matrix from DeLaN model

        Args:
            control_dict (dict): Keyword-mapped values relevant for controller computations
            kwargs (any): Any additional relevant keyword values for computing this loss

        Returns:
            dict:
                mm: mass matrix value
        """
        # Compute mass matrix from delan
        q = control_dict["q"][:, :self.dof]
        qd = control_dict["qd"][:, :self.dof]
        qdd = control_dict["qdd"][:, :self.dof]
        obs = control_dict["obs"] if "obs" in control_dict else None
        j_eef = control_dict["j_eef"][:, :, :self.dof]
        extrinsics = control_dict["extrinsics"] if "extrinsics" in control_dict else None

        # Compose history dict if we're at the appropriate step
        if self.extrinsics_update_ctr % self.steps_per_extrinsics_update == 0:
            self.extrinsics_dict_cache = self._compose_history_dict(control_dict)

        _, mm, c, g, _, _, _, _ = self.model._dyn_model(q, qd, qdd, obs, j_eef, extrinsics, self.extrinsics_dict_cache)

        # Optionally use ground truth mass matrix
        if self.use_ground_truth_mm:
            mm = control_dict["mm"]

        # Update counter
        self.extrinsics_update_ctr = (self.extrinsics_update_ctr + 1) % self.steps_per_extrinsics_update

        # zero-out torque values if we're not using compensation torques
        if not self.use_compensation_torques:
            c = torch.zeros_like(c)
            g = torch.zeros_like(g)
        # Otherwise, normalize if requested
        elif self.normalized_efforts:
                c = c / self.max_efforts
                g = g / self.max_efforts

        ret = {
            "mm": mm,
            "torques_coriolis": c,
            "torques_gravity": g
        }

        self._cache.update({k: v.detach() for k, v in ret.items()})

        return ret

    def ff_torques(self, control_dict, q=None, qd=None, qdd=None, **kwargs):
        """
        Calculate the predicted feedforward torques from this model. Setpoints @q, @qd, and / or @qdd can be
        optionally specified; if left empty, will be inferred from @control_dict

        Note: Normalizes efforst if self.normalized_efforts is True.

        Args:
            control_dict (dict): Keyword-mapped values relevant for controller computations
            q (None or tensor): If specified, should be the desired q state
            qd (None or tensor): If specified, should be the desired q velocity
            qdd (None or tensor): If specified, should be the desired q acceleration
            kwargs (any): Any additional relevant keyword values for computing this loss

        Returns:
            tensor: Feedforward torques from this model
        """
        if q is None:
            q = control_dict["q"][:, :self.dof]
        if qd is None:
            qd = control_dict["qd"][:, :self.dof]
        if qdd is None:
            qdd = control_dict["qdd"][:, :self.dof]
        obs = control_dict["obs"] if "obs" in control_dict else None
        j_eef = control_dict["j_eef"][:, :, :self.dof]
        extrinsics = control_dict["extrinsics"] if "extrinsics" in control_dict else None

        # Compose history dict
        history_dict = self._compose_history_dict(control_dict)

        # Calculate torques
        ff_torques = self.model.inv_dyn(q, qd, qdd, obs, j_eef, extrinsics, history_dict)

        # Normalize if requested
        if self.normalized_efforts:
            ff_torques = ff_torques / self.max_efforts

        return ff_torques

    @autocast(enabled=macros.MIXED_PRECISION)
    def loss(self, control_dict, **kwargs):
        with maybe_no_grad(no_grad=self.freeze):
            # Grab relevant values from kwargs
            tau = kwargs["control"][:, :self.dof]

            # De-normalize efforts if necessary
            if self.normalized_efforts:
                tau = tau * self.max_efforts

            # Delan inv dyn loss
            q = control_dict["q"][:, :self.dof]
            qd = control_dict["qd"][:, :self.dof]
            qdd = control_dict["qdd"][:, :self.dof]
            obs = control_dict["obs"]
            j_eef = control_dict["j_eef"]
            extrinsics = control_dict["extrinsics"]

            # Compose history dict
            history_dict = self._compose_history_dict(control_dict)

            # Include DeLaN loss for learning inverse dynamics
            # Compute the Rigid Body Dynamics Model:
            tau_hat, dEdt_hat, tau_fric_pred, tau_force_pred, qdd_hat = self.model(q, qd, qdd, tau, obs, j_eef, extrinsics, history_dict, get_qdd_pred=self.train_with_forward_loss)
            # Compute the loss of the Euler-Lagrange Differential Equation:
            err_inv = torch.sum((tau_hat - tau) ** 2, dim=1)
            # Compute the loss of the Power Conservation:
            dEdt = torch.matmul(qd.view(-1, self.dof, 1).transpose(dim0=1, dim1=2), (tau + tau_fric_pred + tau_force_pred).view(-1, self.dof, 1)).view(-1)
            err_dEdt = (dEdt_hat - dEdt) ** 2

            # Compose loss
            delan_loss = err_inv + err_dEdt

            if self.train_with_forward_loss:
                delan_loss = delan_loss + torch.sum((qdd_hat - qdd) ** 2, dim=1)

            # Zero out values where qd values are bigger than limit
            delan_loss[torch.sum(torch.abs(qd) > self.max_velocities.unsqueeze(0), dim=-1) > 0] = 0.0

            # Store mean of this loss value for calculating actor loss scale
            self._current_mean_loss = torch.mean(delan_loss).detach()

        # Return loss
        return delan_loss

    def get_cache(self):
        """
        Grabs cached values for this control model

        Returns:
            dict: Keyword-mapped tensor generated by this control model from the most recent (__call__) method
        """
        return {k: v.clone() for k, v in self._cache.items()}

    def _compose_history_dict(self, control_dict):
        """
        Creates history dict from corresponding values in @control_dict.

        Args:
            control_dict (dict): Keyword-mapped values relevant for controller computations

        Returns:
            dict: Specific values' history (each should be (n_envs, hist_len, D) tensor
        """
        history_dict = {}
        for k, v in control_dict.items():
            if "history" in k:
                history_dict[k] = v[..., :self.dof]
                # Optionally normalize the control
                if "control" in k:
                    history_dict[k] /= self.max_efforts.unsqueeze(0).unsqueeze(0)

        return history_dict

    def gradient_step(self, loss, retain_graph=False):
        # We only take a gradient step if we're not freezing the weights
        if not self.freeze:
            # Zero out gradients
            self.optimizer.zero_grad()
            # Run backprop and clip gradients
            loss.backward(retain_graph=retain_graph)
            nn.utils.clip_grad_norm_(self._trainable_params, max_norm=self.max_grad_norm)
            # Take a gradient step
            self.optimizer.step()

    def train(self, mode=True):
        # We always pass False if we're freezing (all) the weights, otherwise run super method as normal
        return super().train(mode=False) if self.freeze else super().train(mode=mode)

    @property
    def actor_loss_scale(self):
        # This is the loss threshold divided by the current delan loss
        # Intuition: If delan loss is large, then our dynamics model is changing and we cannot rely on a given set of
        #   actions producing the same results (and, more importantly, the same reward). So we want to reduce the actor
        #   loss so we don't take as large a gradient step in this case.
        return torch.clip(self.delan_args["loss_threshold"] / self._current_mean_loss, 0.1, 1.0)

    @property
    def train_with_actor_loss(self):
        # Return based on inputted setting
        return self.delan_args["train_with_actor_loss"]

    @property
    def trainable_parameters(self):
        return self._trainable_params

    @property
    def is_before_controller(self):
        return True
