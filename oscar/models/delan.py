import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from oscar.utils.torch_custom import *


class LowTri:
    def __init__(self, m):

        # Calculate lower triangular matrix indices using numpy
        self._m = m
        self._idx = np.tril_indices(self._m)

    def __call__(self, l):
        batch_size = l.shape[0]
        self._L = torch.zeros(batch_size, self._m, self._m).type_as(l)

        # Assign values to matrix:
        self._L[:batch_size, self._idx[0], self._idx[1]] = l[:]
        return self._L[:batch_size]


class Permute(nn.Module):
    """
    Class for permuting tensors. Permutes dimensiona coording to @dims
    """
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, dims=self.dims)


class SoftplusDer(nn.Module):
    def __init__(self, beta=1.):
        super(SoftplusDer, self).__init__()
        self._beta = beta

    def forward(self, x):
        cx = torch.clamp(x, -20., 20.)
        exp_x = torch.exp(self._beta * cx)
        out = exp_x / (exp_x + 1.0)

        if torch.isnan(out).any():
            print("SoftPlus Forward output is NaN.")
        return out


class ReLUDer(nn.Module):
    def __init__(self):
        super(ReLUDer, self).__init__()

    def forward(self, x):
        return torch.ceil(torch.clamp(x, 0, 1))


class ELUDer(nn.Module):
    def __init__(self):
        super(ELUDer, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), 0, 1)


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()

    def forward(self, x):
        return x


class LinearDer(nn.Module):
    def __init__(self):
        super(LinearDer, self).__init__()

    def forward(self, x):
        return torch.clamp(x, 1, 1)


class Cos(nn.Module):
    def __init__(self):
        super(Cos, self).__init__()

    def forward(self, x):
        return torch.cos(x)


class CosDer(nn.Module):
    def __init__(self):
        super(CosDer, self).__init__()

    def forward(self, x):
        return -torch.sin(x)


class Tanh(nn.Module):
    def __init__(self, limit=1.0):
        super(Tanh, self).__init__()
        self.limit = limit

    def forward(self, x):
        return self.limit * torch.tanh(x)


class TanhDer(nn.Module):
    def __init__(self, limit=1.0):
        super(TanhDer, self).__init__()
        self.limit = limit

    def forward(self, x):
        x = torch.tanh(x)
        return self.limit * (1 - x * x)


class TanhExp(nn.Module):
    def __init__(self, limit=1.0):
        super(TanhExp, self).__init__()
        self.limit = limit

    def forward(self, x):
        return torch.exp(self.limit * torch.tanh(x))


class TanhExpDer(nn.Module):
    def __init__(self, limit=1.0):
        super(TanhExpDer, self).__init__()
        self.limit = limit

    def forward(self, x):
        x = torch.tanh(x)
        return torch.exp(self.limit * torch.tanh(x)) * (self.limit * (1 - x * x))


class Clip(nn.Module):
    def __init__(self, limit=1.0):
        super(Clip, self).__init__()
        self.limit = limit

    def forward(self, x):
        return torch.clip(x, -self.limit, self.limit)


class ClipDer(nn.Module):
    def __init__(self, limit=1.0):
        super(ClipDer, self).__init__()
        self.limit = limit

    def forward(self, x):
        return torch.where(torch.abs(x) < self.limit, torch.ones_like(x), torch.zeros_like(x))


class ClipExp(nn.Module):
    def __init__(self, limit=1.0):
        super(ClipExp, self).__init__()
        self.limit = limit

    def forward(self, x):
        return torch.exp(torch.clip(x, -self.limit, self.limit))


class ClipExpDer(nn.Module):
    def __init__(self, limit=1.0):
        super(ClipExpDer, self).__init__()
        self.limit = limit

    def forward(self, x):
        return torch.where(torch.abs(x) < self.limit, torch.exp(x), torch.zeros_like(x))


class LagrangianLayer(nn.Module):

    def __init__(self, input_size, n_dof, activation="ReLu", **activation_kwargs):
        super(LagrangianLayer, self).__init__()

        # Create layer weights and biases:
        self.n_dof = n_dof
        self.weight = nn.Parameter(torch.Tensor(n_dof, input_size))
        self.bias = nn.Parameter(torch.Tensor(n_dof))

        # Initialize activation function and its derivative:
        if activation == "ReLu":
            self.g = nn.ReLU()
            self.g_prime = ReLUDer()

        elif activation == "ELU":
            self.g = nn.ELU()
            self.g_prime = ELUDer()

        elif activation == "Exp":
            self.g = torch.exp
            self.g_prime = torch.exp

        elif activation == "SoftPlus":
            self.softplus_beta = 1.0
            self.g = nn.Softplus(beta=self.softplus_beta)
            self.g_prime = SoftplusDer(beta=self.softplus_beta)

        elif activation == "Cos":
            self.g = Cos()
            self.g_prime = CosDer()

        elif activation == "Linear":
            self.g = Linear()
            self.g_prime = LinearDer()

        elif activation == "Tanh":
            self.g = Tanh(**activation_kwargs)
            self.g_prime = TanhDer(**activation_kwargs)

        elif activation == "TanhExp":
            self.g = TanhExp(**activation_kwargs)
            self.g_prime = TanhExpDer(**activation_kwargs)

        elif activation == "Clip":
            self.g = Clip(**activation_kwargs)
            self.g_prime = ClipDer(**activation_kwargs)

        elif activation == "ClipExp":
            self.g = ClipExp(**activation_kwargs)
            self.g_prime = ClipExpDer(**activation_kwargs)

        else:
            raise ValueError("Activation Type must be in ['Linear', 'ReLu', 'ELU', 'Exp', 'SoftPlus', 'Cos', 'Tanh', 'TanhExp', 'Clip', 'ClipExp'] but is {0}".format(self.activation))

    def forward(self, q, der_prev):
        # Apply Affine Transformation:
        a = F.linear(q, self.weight, self.bias)
        out = self.g(a)
        der = torch.matmul(self.g_prime(a).view(-1, self.n_dof, 1) * self.weight, der_prev)
        return out, der


class DeepLagrangianNetwork(nn.Module):

    def __init__(self, n_dof, obs_dim, extrinsics_dim, history_dim, device, **kwargs):
        super(DeepLagrangianNetwork, self).__init__()

        self.device = device

        # Read optional arguments:
        self.n_width = kwargs.get("n_width", 128)
        self.n_hidden = kwargs.get("n_depth", 1)
        self._b0 = kwargs.get("b_init", 0.1)
        self._b0_diag = kwargs.get("b_diag_init", 0.1)

        self._w_init = kwargs.get("w_init", "xavier_normal")
        self._g_hidden = kwargs.get("g_hidden", np.sqrt(2.))
        self._g_output = kwargs.get("g_hidden", 0.125)
        self._p_sparse = kwargs.get("p_sparse", 0.2)
        self._epsilon = kwargs.get("diagonal_epsilon", 1.e-5)

        # Custom args
        self.obs_dim = obs_dim
        self.extrinsics_dim = extrinsics_dim
        self._history_dim = history_dim
        self.freeze_base = kwargs["freeze_base"]
        self.freeze_extrinsics = kwargs["freeze_extrinsics"]
        self.freeze_residual = [kwargs["freeze_residual"]] if \
            isinstance(kwargs["freeze_residual"], bool) else kwargs["freeze_residual"]
        self._embed_q = kwargs.get("embed_q", False)     # Determines whether we map q to cos(q) and sin(q)
        self._use_extrinsics = kwargs.get("use_extrinsics", False)      # Determines whether we map extrinsic dynamics params to latent space and pass as input
        self._extrinsics_with_residual = kwargs.get("extrinsics_with_residual")
        self._bootstrap_extrinsics = kwargs.get("bootstrap_extrinsics")
        self._extrinsics_net_kwargs = kwargs["extrinsics_net_kwargs"]
        self._extrinsics_latent_dim = kwargs.get("extrinsics_latent_dim", 8)
        self._diag_act = kwargs.get("diag_activation", "ReLu")
        self._learn_residual = kwargs.get("learn_residual", False)
        self.n_residuals = len(self.freeze_residual) if self._learn_residual else None
        self.n_width_residual = kwargs.get("n_width_residual", 128)
        self.n_hidden_residual = kwargs.get("n_depth_residual", 1)
        self._b0_residual = kwargs.get("b_init_residual", 0.0)
        self._b0_diag_residual = kwargs.get("b_diag_init_residual", 0.0)
        self._max_residual_magnitude = kwargs.get("max_residual_magnitude", 0.1)        # Should be in (0, 1)
        self._use_tanh_residual_output = kwargs.get("use_tanh_residual_output", True)   # If False, will use raw clipping instead
        self._use_exponential_residual = kwargs.get("use_exponential_residual", False)      # Whether residual is exponentially scaled or not

        # # Define last obs
        # self.last_obs = torch.zeros(1, self.obs_dim, device=self.device)

        # Compute In- / Output Sizes:
        self.n_dof = n_dof
        self.m = int((n_dof ** 2 + n_dof) / 2)

        # Construct Weight Initialization:
        if self._w_init == "xavier_normal":

            # Construct initialization function:
            def init_hidden(layer):

                # Set the Hidden Gain:
                if self._g_hidden <= 0.0: hidden_gain = torch.nn.init.calculate_gain('relu')
                else: hidden_gain = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.xavier_normal_(layer.weight, hidden_gain)

            def init_output(layer):
                # Set Output Gain:
                if self._g_output <= 0.0: output_gain = torch.nn.init.calculate_gain('linear')
                else: output_gain = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.xavier_normal_(layer.weight, output_gain)

            # Construct initialization function for residual:
            def init_hidden_residual(layer):

                # Set the Hidden Gain:
                if self._g_hidden <= 0.0: hidden_gain = torch.nn.init.calculate_gain('relu')
                else: hidden_gain = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0_residual)
                torch.nn.init.xavier_normal_(layer.weight, hidden_gain)

            def init_output_residual(layer):
                # Set Output Gain:
                if self._g_output <= 0.0: output_gain = torch.nn.init.calculate_gain('linear')
                else: output_gain = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0_residual)
                torch.nn.init.xavier_normal_(layer.weight, output_gain)

        elif self._w_init == "orthogonal":

            # Construct initialization function:
            def init_hidden(layer):
                # Set the Hidden Gain:
                if self._g_hidden <= 0.0: hidden_gain = torch.nn.init.calculate_gain('relu')
                else: hidden_gain = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.orthogonal_(layer.weight, hidden_gain)

            def init_output(layer):
                # Set Output Gain:
                if self._g_output <= 0.0: output_gain = torch.nn.init.calculate_gain('linear')
                else: output_gain = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.orthogonal_(layer.weight, output_gain)

            # Construct initialization function for residual:
            def init_hidden_residual(layer):
                # Set the Hidden Gain:
                if self._g_hidden <= 0.0: hidden_gain = torch.nn.init.calculate_gain('relu')
                else: hidden_gain = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0_residual)
                torch.nn.init.orthogonal_(layer.weight, hidden_gain)

            def init_output_residual(layer):
                # Set Output Gain:
                if self._g_output <= 0.0: output_gain = torch.nn.init.calculate_gain('linear')
                else: output_gain = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0_residual)
                torch.nn.init.orthogonal_(layer.weight, output_gain)

        elif self._w_init == "sparse":
            assert self._p_sparse < 1. and self._p_sparse >= 0.0

            # Construct initialization function:
            def init_hidden(layer):
                p_non_zero = self._p_sparse
                hidden_std = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.sparse_(layer.weight, p_non_zero, hidden_std)

            def init_output(layer):
                p_non_zero = self._p_sparse
                output_std = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.sparse_(layer.weight, p_non_zero, output_std)

            # Construct initialization function for residual:
            def init_hidden_residual(layer):
                p_non_zero = self._p_sparse
                hidden_std = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0_residual)
                torch.nn.init.sparse_(layer.weight, p_non_zero, hidden_std)

            def init_output_residual(layer):
                p_non_zero = self._p_sparse
                output_std = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0_residual)
                torch.nn.init.sparse_(layer.weight, p_non_zero, output_std)

        else:
            raise ValueError("Weight Initialization Type must be in ['xavier_normal', 'orthogonal', 'sparse'] but is {0}".format(self._w_init))

        # Compute non-zero elements of L:
        l_output_size = int((self.n_dof ** 2 + self.n_dof) / 2)
        l_lower_size = l_output_size - self.n_dof

        # Calculate the indices of the diagonal elements of L:
        idx_diag = np.arange(self.n_dof) + 1
        idx_diag = idx_diag * (idx_diag + 1) / 2 - 1

        # Calculate the indices of the off-diagonal elements of L:
        idx_tril = np.extract([x not in idx_diag for x in np.arange(l_output_size)], np.arange(l_output_size))

        # Indexing for concatenation of l_o  and l_d
        cat_idx = np.hstack((idx_diag, idx_tril))
        order = np.argsort(cat_idx)
        self._idx = np.arange(cat_idx.size)[order]

        # create it once and only apply repeat, this may decrease memory allocation
        self._eye = torch.eye(self.n_dof).view(1, self.n_dof, self.n_dof)
        self.low_tri = LowTri(self.n_dof)

        # Create Network:
        self.layers = nn.ModuleList()
        non_linearity = kwargs.get("activation", "ReLu")

        # Create Input Layer:
        input_dim = self.n_dof * 2 if self._embed_q else self.n_dof
        if self._use_extrinsics and not self._extrinsics_with_residual:
            input_dim += self._extrinsics_latent_dim
        self.layers.append(LagrangianLayer(input_dim, self.n_width, activation=non_linearity))
        init_hidden(self.layers[-1])

        # Create Hidden Layer:
        for _ in range(1, self.n_hidden):
            self.layers.append(LagrangianLayer(self.n_width, self.n_width, activation=non_linearity))
            init_hidden(self.layers[-1])

        # Create output Layer:
        self.net_g = LagrangianLayer(self.n_width, 1, activation="Linear")
        init_output(self.net_g)

        self.net_lo = LagrangianLayer(self.n_width, l_lower_size, activation="Linear")
        init_hidden(self.net_lo)

        # The diagonal must be non-negative. Therefore, the non-linearity must be a positive one
        assert self._diag_act in {"ReLu", "Exp", "SoftPlus"}
        self.net_ld = LagrangianLayer(self.n_width, self.n_dof, activation=self._diag_act)
        init_hidden(self.net_ld)
        torch.nn.init.constant_(self.net_ld.bias, self._b0_diag)

        # Create Extrinsics Embedding Network if requested:
        if self._use_extrinsics:
            self.layers_extrinsics = nn.ModuleList()

            # Create Input + Hidden Layers:
            if self._bootstrap_extrinsics:
                # If we're using cnn, use MLP proceeded by 1D Conv Net
                if self._extrinsics_net_kwargs["use_cnn"]:
                    input_dim_ext = self.n_dof * 4 if self._embed_q else 3  # q (maybe sin + cos), qd, controls
                    for hidden_dim in self._extrinsics_net_kwargs["mlp_hidden_dims"]:
                        self.layers_extrinsics.append(nn.Linear(input_dim_ext, hidden_dim))
                        init_hidden(self.layers_extrinsics[-1])
                        self.layers_extrinsics.append(nn.ReLU())
                        input_dim_ext = hidden_dim
                    input_channel_dim = self._extrinsics_net_kwargs["cnn_input_channels"][0]
                    self.layers_extrinsics.append(nn.Linear(input_dim_ext, input_channel_dim))
                    self.layers_extrinsics.append(Permute(dims=(0, 2, 1)))
                    l_dim = self._history_dim
                    c_dim = None
                    for in_c, out_c, kernel, stride in zip(
                        self._extrinsics_net_kwargs["cnn_input_channels"],
                        self._extrinsics_net_kwargs["cnn_output_channels"],
                        self._extrinsics_net_kwargs["cnn_kernels"],
                        self._extrinsics_net_kwargs["cnn_strides"],
                    ):
                        self.layers_extrinsics.append(nn.Conv1d(in_c, out_c, kernel_size=kernel, stride=stride))
                        init_hidden(self.layers_extrinsics[-1])
                        self.layers_extrinsics.append(nn.ReLU())
                        l_dim = int((l_dim - (kernel - 1) - 1) / stride + 1)
                        c_dim = out_c
                    # Finally add a flattening layer
                    self.layers_extrinsics.append(nn.Flatten())
                    # Calculate output dimension
                    input_dim_ext = l_dim * c_dim

                # Otherwise, just use MLP
                else:
                    input_dim_ext = self._history_dim * self.n_dof * 4 if self._embed_q else 3  # q (maybe sin + cos), qd, controls
                    self.layers_extrinsics.append(nn.Flatten())
                    for hidden_dim in self._extrinsics_net_kwargs["mlp_hidden_dims"]:
                        self.layers_extrinsics.append(nn.Linear(input_dim_ext, hidden_dim))
                        init_hidden(self.layers_extrinsics[-1])
                        self.layers_extrinsics.append(nn.ReLU())
                        input_dim_ext = hidden_dim

            else:       # Not bootstrapping extrinsics, just use flat network
                input_dim_ext = self.extrinsics_dim
                for hidden_dim in self._extrinsics_net_kwargs["mlp_hidden_dims"]:
                    self.layers_extrinsics.append(nn.Linear(input_dim_ext, hidden_dim))
                    init_hidden(self.layers_extrinsics[-1])
                    self.layers_extrinsics.append(nn.ReLU())
                    input_dim_ext = hidden_dim

            # Create output Layer:
            self.net_extrinsics_latent = nn.Linear(input_dim_ext, self._extrinsics_latent_dim)
            init_output(self.net_extrinsics_latent)

        # Create Residual Network if requested:
        if self._learn_residual:
            # Define arrays of residual nets to learn
            self.layers_residuals = nn.ModuleList()
            self.net_g_residuals = nn.ModuleList()
            self.net_lo_residuals = nn.ModuleList()
            self.net_ld_residuals = nn.ModuleList()

            for i in range(self.n_residuals):
                non_linearity = kwargs.get("activation", "ReLu")
                if self._use_exponential_residual:
                    out_activation = "TanhExp" if self._use_tanh_residual_output else "ClipExp"
                else:
                    out_activation = "Tanh" if self._use_tanh_residual_output else "Clip"

                # Create Network:
                layers_residual = nn.ModuleList()

                # Create Input Layer:
                input_dim = self.n_dof * 2 if self._embed_q else self.n_dof
                input_dim += l_lower_size + self.n_dof + 1
                if self._use_extrinsics and self._extrinsics_with_residual:
                    input_dim += self._extrinsics_latent_dim
                layers_residual.append(LagrangianLayer(input_dim, self.n_width_residual, activation=non_linearity))
                init_hidden_residual(layers_residual[-1])

                # Create Hidden Layer:
                for _ in range(1, self.n_hidden_residual):
                    layers_residual.append(LagrangianLayer(self.n_width_residual, self.n_width_residual, activation=non_linearity))
                    init_hidden_residual(layers_residual[-1])

                # Add these nets to the residual layers
                self.layers_residuals.append(layers_residual)

                # Create output Layer:
                net_g_residual = LagrangianLayer(self.n_width_residual, 1, activation=out_activation, limit=self._max_residual_magnitude)
                init_output_residual(net_g_residual)
                self.net_g_residuals.append(net_g_residual)

                net_lo_residual = LagrangianLayer(self.n_width_residual, l_lower_size, activation=out_activation, limit=self._max_residual_magnitude)
                init_hidden_residual(net_lo_residual)
                self.net_lo_residuals.append(net_lo_residual)

                # The diagonal can be negative, but cannot be greater in magnitude than the base diagonal value
                net_ld_residual = LagrangianLayer(self.n_width_residual, self.n_dof, activation=out_activation, limit=self._max_residual_magnitude)
                init_hidden_residual(net_ld_residual)
                torch.nn.init.constant_(net_ld_residual.bias, self._b0_diag_residual)
                self.net_ld_residuals.append(net_ld_residual)

        # Store reference to base and residual networks for easy access
        self.base_nets = [self.layers, self.net_g, self.net_lo, self.net_ld]
        self.extrinsics_nets = []
        if self._use_extrinsics:
            self.extrinsics_nets += [self.layers_extrinsics, self.net_extrinsics_latent]
        self.residual_nets = [[self.layers_residuals[i], self.net_g_residuals[i],
                              self.net_lo_residuals[i], self.net_ld_residuals[i]] for i in range(self.n_residuals)] if \
            self._learn_residual else []

    def forward(self, q, qd, qdd, tau=None, obs=None, j_eef=None, extrinsics=None, history_dict=None, get_qdd_pred=False):
        out = self._dyn_model(q, qd, qdd, obs, j_eef, extrinsics, history_dict)
        tau_pred, H, c, g = out[0], out[1], out[2], out[3]
        dEdt = out[6] + out[7]

        qdd_pred = None
        if get_qdd_pred:
            # Compute Acceleration, e.g., forward model:
            invH = torch.inverse(H)
            qdd_pred = torch.matmul(invH, (tau - c - g).view(-1, self.n_dof, 1)).view(-1, self.n_dof)

        return tau_pred, dEdt, qdd_pred

    def _dyn_model(self, q, qd, qdd, obs=None, j_eef=None, extrinsics=None, history_dict=None):
        # Embed q if necessary
        if self._embed_q:
            cq, sq = torch.cos(q), torch.sin(q)
            q = torch.cat([cq, sq], dim=-1)
            # Calculate derivative dcos(q)/dq , dsin(q)/dq
            der_input = torch.cat([-sq.unsqueeze(-1) * self._eye, cq.unsqueeze(-1) * self._eye], dim=1).type_as(q)
        # Otherwise, just create derivative
        else:
            # Create initial derivative of dq/dq = 1
            der_input = self._eye.repeat(q.shape[0], 1, 1).type_as(q)

        qd_3d = qd.view(-1, self.n_dof, 1)
        qd_4d = qd.view(-1, 1, self.n_dof, 1)

        # First compute extrinsics embedding if used
        if self._use_extrinsics:
            if self._bootstrap_extrinsics:
                q_history = [torch.cos(history_dict["q_history"]), torch.sin(history_dict["q_history"])] if self._embed_q else [history_dict["q_history"]]
                extrinsics = torch.cat(
                    [
                        *q_history,
                        history_dict["qd_history"],
                        history_dict["arm_control_history"],
                    ],
                    dim=-1)
            for layer in self.layers_extrinsics:
                extrinsics = layer(extrinsics)

            # Pass extrinsics through final layer
            extrinsics = self.net_extrinsics_latent(extrinsics)
            der_ext = torch.zeros(q.shape[0], self._extrinsics_latent_dim, self.n_dof, device=self.device).type_as(q)

            # Append values to normal inputs if we're including them with the base model
            if not self._extrinsics_with_residual:
                q = torch.cat([q, extrinsics], dim=-1)
                der_input = torch.cat([der_input, der_ext], dim=1)

        der = der_input

        # Compute shared network between l & g:
        y, der = self.layers[0](q, der)

        for i in range(1, len(self.layers)):
            y, der = self.layers[i](y, der)

        # Compute the network heads including the corresponding derivative:
        l_lower, der_l_lower = self.net_lo(y, der)
        l_diag, der_l_diag = self.net_ld(y, der)

        # Compute the potential energy and the gravitational force:
        V, der_V = self.net_g(y, der)
        V = V.squeeze()
        g = der_V.squeeze()

        # Assemble l and der_l
        l_diag = l_diag
        l = torch.cat((l_diag, l_lower), 1)[:, self._idx]
        der_l = torch.cat((der_l_diag, der_l_lower), 1)[:, self._idx, :]

        # Compute H:
        L = self.low_tri(l)
        LT = L.transpose(dim0=1, dim1=2)
        H = torch.matmul(L, LT) + self._epsilon * torch.eye(self.n_dof).type_as(L)

        # Calculate dH/dt
        Ldt = self.low_tri(torch.matmul(der_l, qd_3d).view(-1, self.m))
        Hdt = torch.matmul(L, Ldt.transpose(dim0=1, dim1=2)) + torch.matmul(Ldt, LT)

        # Calculate dH/dq:
        Ldq = self.low_tri(der_l.transpose(2, 1).reshape(-1, self.m)).reshape(-1, self.n_dof, self.n_dof, self.n_dof)
        Hdq = torch.matmul(Ldq, LT.view(-1, 1, self.n_dof, self.n_dof)) + torch.matmul(L.view(-1, 1, self.n_dof, self.n_dof), Ldq.transpose(2, 3))

        # If we're using residual, add those in now
        if self._learn_residual:
            # Loop through all residuals and run sequentially
            for layers_residual, net_g_residual, net_lo_residual, net_ld_residual in zip(
                self.layers_residuals, self.net_g_residuals, self.net_lo_residuals, self.net_ld_residuals
            ):
                inputs = [q, extrinsics] if (self._extrinsics_with_residual and self._use_extrinsics) else [q]
                der_ext = torch.zeros(q.shape[0], self._extrinsics_latent_dim, self.n_dof, device=self.device).type_as(q)
                der_inputs = [der_input, der_ext] if (self._extrinsics_with_residual and self._use_extrinsics) else [der_input]
                # Create initial derivative to pass to residual network (dq/dq, dl_lower/dq, dl_diag/dq, g)
                der_residual = torch.cat([
                    *der_inputs,
                    der_l_lower,
                    der_l_diag,
                    der_V,
                ], dim=1)

                # Compose initial input to pass to residual network
                y_residual = torch.cat([
                    *inputs,
                    l_lower,
                    l_diag,
                    V.unsqueeze(-1),
                ], dim=-1)

                # Compute shared network between l & g
                for layer in layers_residual:
                    y_residual, der_residual = layer(y_residual, der_residual)

                # Compute the network heads including the corresponding derivative:
                l_lower_residual, der_l_lower_residual = net_lo_residual(y_residual, der_residual)
                l_diag_residual, der_l_diag_residual = net_ld_residual(y_residual, der_residual)

                # Compute the potential energy and the gravitational force:
                V_residual, der_V_residual = net_g_residual(y_residual, der_residual)
                V_residual = V_residual.squeeze()
                g_residual = der_V_residual.squeeze()

                # Assemble l and der_l
                l_lower_residual_cat = torch.cat((torch.zeros_like(l_diag_residual), l_lower_residual), 1)[:, self._idx]
                l_diag_residual_cat = torch.cat((l_diag_residual, torch.zeros_like(l_lower_residual)), 1)[:, self._idx]
                der_l_lower_residual_cat = torch.cat((torch.zeros_like(der_l_diag_residual), der_l_lower_residual), 1)[:, self._idx, :]
                der_l_diag_residual_cat = torch.cat((der_l_diag_residual, torch.zeros_like(der_l_lower_residual)), 1)[:, self._idx, :]

                # Update H
                L_lower_residual = self.low_tri(l_lower_residual_cat)
                L_diag_residual = self.low_tri(l_diag_residual_cat)
                LT_lower_residual = L_lower_residual.transpose(dim0=1, dim1=2)
                H_res = L_lower_residual + LT_lower_residual + L_diag_residual
                H_base = H      # need reference for calculating other values
                H = H_base * H_res

                # Update dH/dt -- we add in dH / dt * H_res + H_base * dH_res / dt
                # dH_res / dt = dL_res_lower / dt + dL_res_lower ^ T / dt + dL_res_diag / dt
                L_lower_residual_dt = self.low_tri(torch.matmul(der_l_lower_residual_cat, qd_3d).view(-1, self.m))
                L_diag_residual_dt = self.low_tri(torch.matmul(der_l_diag_residual_cat, qd_3d).view(-1, self.m))
                Hdt_res = L_lower_residual_dt + L_lower_residual_dt.transpose(dim0=1, dim1=2) + L_diag_residual_dt
                Hdt_base = Hdt     # save reference to original value
                Hdt = Hdt_base * H_res + H_base * Hdt_res

                # Update dH/dq -- we add in dH / dq * H_res + H_base * dH_res / dq
                # dH_res / dq = dL_res_lower / dq + dL_res_lower ^ T / dq + dL_res_diag / dq
                L_lower_residual_dq = self.low_tri(der_l_lower_residual_cat.transpose(2, 1).reshape(-1, self.m)).\
                    reshape(-1, self.n_dof, self.n_dof, self.n_dof)
                L_diag_residual_dq = self.low_tri(der_l_diag_residual_cat.transpose(2, 1).reshape(-1, self.m)).\
                    reshape(-1, self.n_dof, self.n_dof, self.n_dof)
                Hdq_res = L_lower_residual_dq + L_lower_residual_dq.transpose(dim0=1, dim1=2) + L_diag_residual_dq
                Hdq_base = Hdq      # save reference to original value
                Hdq = Hdq_base * H_res.unsqueeze(-1) + H_base.unsqueeze(-1) * Hdq_res

                # Update V
                V_base = V          # keep reference to original value for computing g
                V = V_base * V_residual

                # Update g -- we add in g * V_res + V_base * g_res
                g_base = g
                g = g_base * V_residual.unsqueeze(-1) + V_base.unsqueeze(-1) * g_residual

                # Add original values if we're not using exponential residual
                if not self._use_exponential_residual:
                    H = H + H_base
                    Hdt = Hdt + Hdt_base
                    Hdq = Hdq + Hdq_base
                    V = V + V_base
                    g = g + g_base

        # Compute the Coriolis & Centrifugal forces:
        Hdt_qd = torch.matmul(Hdt, qd_3d).view(-1, self.n_dof)
        quad_dq = torch.matmul(qd_4d.transpose(dim0=2, dim1=3), torch.matmul(Hdq, qd_4d)).view(-1, self.n_dof)
        c = Hdt_qd - 1. / 2. * quad_dq

        # Compute the Torque using the inverse model:
        H_qdd = torch.matmul(H, qdd.view(-1, self.n_dof, 1)).view(-1, self.n_dof)
        tau_pred = H_qdd + c + g

        # Compute kinetic energy T
        H_qd = torch.matmul(H, qd_3d).view(-1, self.n_dof)
        T = 1. / 2. * torch.matmul(qd_4d.transpose(dim0=2, dim1=3), H_qd.view(-1, 1, self.n_dof, 1)).view(-1)

        # Compute dT/dt:
        qd_H_qdd = torch.matmul(qd_4d.transpose(dim0=2, dim1=3), H_qdd.view(-1, 1, self.n_dof, 1)).view(-1)
        qd_Hdt_qd = torch.matmul(qd_4d.transpose(dim0=2, dim1=3), Hdt_qd.view(-1, 1, self.n_dof, 1)).view(-1)
        dTdt = qd_H_qdd + 0.5 * qd_Hdt_qd

        # Compute dV/dt
        dVdt = torch.matmul(qd_4d.transpose(dim0=2, dim1=3), g.view(-1, 1, self.n_dof, 1)).view(-1)
        return tau_pred, H, c, g, T, V, dTdt, dVdt

    def inv_dyn(self, q, qd, qdd, obs=None, j_eef=None, extrinsics=None, history_dict=None):
        out = self._dyn_model(q, qd, qdd, obs, j_eef, extrinsics, history_dict)
        tau_pred = out[0]
        return tau_pred

    def for_dyn(self, q, qd, tau, obs=None, j_eef=None, extrinsics=None, history_dict=None):
        out = self._dyn_model(q, qd, torch.zeros_like(q), obs, j_eef, extrinsics, history_dict)
        H, c, g = out[1], out[2], out[3]

        # Compute Acceleration, e.g., forward model:
        invH = torch.inverse(H)
        qdd_pred = torch.matmul(invH, (tau - c - g).view(-1, self.n_dof, 1)).view(-1, self.n_dof)
        return qdd_pred

    def energy(self, q, qd, obs=None, j_eef=None, extrinsics=None, history_dict=None):
        out = self._dyn_model(q, qd, torch.zeros_like(q), obs, j_eef, extrinsics, history_dict)
        E = out[4] + out[5]
        return E

    def energy_dot(self, q, qd, qdd, obs=None, j_eef=None, extrinsics=None, history_dict=None):
        out = self._dyn_model(q, qd, qdd, obs, j_eef, extrinsics, history_dict)
        dEdt = out[6] + out[7]
        return dEdt

    def to(self, device):
        # Run super method first
        super(DeepLagrangianNetwork, self).to(device=device)
        # Map eye module
        self._eye = self._eye.to(device=device)

    def cuda(self, device=None):

        # Move the Network to the GPU:
        super(DeepLagrangianNetwork, self).cuda(device=device)

        # Move the eye matrix to the GPU:
        self._eye = self._eye.cuda()
        return self

    def cpu(self):

        # Move the Network to the CPU:
        super(DeepLagrangianNetwork, self).cpu()

        # Move the eye matrix to the CPU:
        self._eye = self._eye.cpu()
        return self

    def train(self, mode=True):
        # Run super method first
        super().train(mode=mode)

        # We force specific components to be in eval mode if we're freezing the weights
        if self.freeze_base:
            for net in self.base_nets:
                net.eval()
        if self._use_extrinsics:
            if self.freeze_extrinsics:
                for net in self.extrinsics_nets:
                    net.eval()
        if self._learn_residual:
            for i, freeze_residual in enumerate(self.freeze_residual):
                if freeze_residual:
                    for net in self.residual_nets[i]:
                        net.eval()
