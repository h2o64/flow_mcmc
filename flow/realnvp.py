# RealNVP Normalizing flow
# - Density estimation using Real NVP (Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio)

# Libraries
import numpy as np
import torch
import torch.nn as nn
from pyro.distributions.transforms import AffineCoupling
from torch.distributions import MultivariateNormal

def make_checker_mask(dim, dim_phys, parity):
    if dim_phys == 1:
        checker = torch.ones((dim,), dtype=torch.uint8) - parity
        checker[::2] = parity
    elif dim_phys == 2:
        dim_grid = int(np.sqrt(dim))
        checker = torch.ones((dim_grid, dim_grid), dtype=torch.uint8) - parity
        checker[::2] = parity
        checker[::2, ::2] = parity
        checker[1::2, 1::2] = parity
    else:
        raise RuntimeError('Mask shape not understood')
    return checker.float()

class MLP(nn.Module):
    def __init__(self, layerdims, activation=torch.relu, init_scale=None):
        super().__init__()
        self.layerdims = layerdims
        self.activation = activation
        linears = [
            nn.Linear(layerdims[i], layerdims[i + 1])
            for i in range(len(layerdims) - 1)
        ]

        if init_scale is not None:
            for l, layer in enumerate(linears):
                torch.nn.init.normal_(
                    layer.weight,
                    std=init_scale / np.sqrt(layerdims[l]),
                )
                torch.nn.init.zeros_(layer.bias)

        self.linears = nn.ModuleList(linears)

    def forward(self, x):
        layers = list(enumerate(self.linears))
        for _, l in layers[:-1]:
            x = self.activation(l(x))
        y = layers[-1][1](x)
        return y


class ResidualAffineCoupling(nn.Module):
    """Residual Affine Coupling layer
    Implements coupling layers with a rescaling
    Args:
        s (nn.Module): scale network
        t (nn.Module): translation network
        mask (binary tensor): binary array of same
        dt (float): rescaling factor for s and t
    """

    def __init__(self, s=None, t=None, mask=None, dt=1, equivariant=False):
        super().__init__()

        self.mask = mask
        self.scale_net = s
        self.trans_net = t
        self.dt = dt
        self.equivariant = equivariant

    def forward(self, x, log_det_jac=None, inverse=False):
        if log_det_jac is None:
            log_det_jac = 0

        s = self.mask * self.scale_net(x * (1 - self.mask))
        s = torch.tanh(s)
        t = self.mask * self.trans_net(x * (1 - self.mask))

        s = self.dt * s
        t = self.dt * t

        if self.equivariant:
            s = torch.abs(s)

        if inverse:
            # if torch.isnan(torch.exp(-s)).any():
            #     raise RuntimeError("Scale factor has NaN entries")
            log_det_jac -= s.view(s.size(0), -1).sum(-1)

            x = x * torch.exp(-s) - t

        else:
            log_det_jac += s.view(s.size(0), -1).sum(-1)
            x = (x + t) * torch.exp(s)
            # if torch.isnan(torch.exp(s)).any():
            #     raise RuntimeError("Scale factor has NaN entries")

        return x, log_det_jac

class MLConv2d(nn.Module):
    def __init__(self, hidden_sizes=[10], kernel_size=5, in_channels=1,
                 out_channels=1, init_scale=1):
        super(MLConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        sizes = [in_channels] + hidden_sizes + [out_channels]
        assert kernel_size % 2 == 1, 'kernel size must be odd for PyTorch >= 1.5.0'
        
        padding_size = (kernel_size // 2)
        layers = []

        for i in range(len(sizes) - 1):
            layers.append(
                torch.nn.Conv2d(sizes[i], sizes[i+1], kernel_size, 
                                padding=padding_size, stride=1, 
                                padding_mode='circular'))
            if i != len(sizes) - 2:
                layers.append(torch.nn.LeakyReLU())
        self.net = nn.ModuleList(layers)

        for param in list(self.net.parameters()): 
            # print(torch.abs(torch.mean(param)))
            param.data = param.data * init_scale


    def forward(self, x):
        reshape = False
        if len(x.shape) < 4:
            reshape = True
            dim = int(np.sqrt(x.shape[-1] / self.in_channels))
            x = x.view(x.shape[0], self.in_channels, dim, dim)

        layers = list(enumerate(self.net))
        for _, l in layers:
            x = l(x)

        if reshape:
            return x.view(x.shape[0], int(x.shape[1] * dim * dim))
        else:
            return x

class RealNVP(nn.Module):
    """ Minimal Real NVP architecture
    Args:
        dims (int,):
        n_realnvp_blocks (int): each with 2 layers
        residual
        block_depth (int): number of pair of integration step per block
    """

    def __init__(self, dim, channels, n_realnvp_blocks, block_depth,
                 init_weight_scale=1,
                 prior_arg={'type': 'standn'},
                 mask_type='half',  # 'half' or 'inter'
                 residual=True,
                 equivariant=False,
                 hidden_dim=10,
                 hidden_depth=3,
                 hidden_activation=torch.relu,
                 dim_phys=1,
                 models_type='mlp',
                 device='cpu'):
        super(RealNVP, self).__init__()

        self.device = device
        self.models_type = models_type
        self.dim = dim
        self.dim_phys = dim_phys
        self.channels = channels
        self.n_blocks = n_realnvp_blocks
        self.block_depth = block_depth
        self.couplings_per_block = 2  # necessary for one update
        self.n_layers_in_coupling = hidden_depth
        self.hidden_dim_in_coupling = hidden_dim
        self.hidden_activation = hidden_activation
        self.init_scale_in_coupling = init_weight_scale
        self.residual = residual
        self.equivariant = equivariant

        mask = torch.ones(dim, device=self.device)
        if mask_type == 'half':
            mask[:int(dim / 2)] = 0
        elif mask_type == 'inter':
            mask = make_checker_mask(dim, dim_phys, 0)
            mask = mask.to(device)
        else:
            raise RuntimeError('Mask type is either half or inter')
        
        self.mask = mask.view(1, dim)
        self.coupling_layers = self.initialize()

        self.prior_arg = prior_arg

        if prior_arg['type'] == 'standn':
            self.prior_prec = torch.eye(dim).to(device)
            self.prior_log_det = 0
            self.prior_distrib = MultivariateNormal(
                torch.zeros((dim,), device=self.device), self.prior_prec)

        elif prior_arg['type'] == 'uncoupled':
            self.prior_prec = prior_arg['a'] * torch.eye(dim).to(device)
            self.prior_log_det = - torch.logdet(self.prior_prec)
            self.prior_distrib = MultivariateNormal(
                torch.zeros((dim,), device=self.device),
                precision_matrix=self.prior_prec)

        elif prior_arg['type'] == 'coupled':
            self.beta_prior = prior_arg['beta']
            self.coef = prior_arg['alpha'] * dim
            prec = torch.eye(dim) * (3 * self.coef + 1 / self.coef)
            prec -= self.coef * torch.triu(torch.triu(torch.ones_like(prec),
                                                      diagonal=-1).T, diagonal=-1)
            prec = prior_arg['beta'] * prec
            self.prior_prec = prec.to(self.device)
            self.prior_log_det = - torch.logdet(prec)
            self.prior_distrib = MultivariateNormal(
                torch.zeros((dim,), device=self.device),
                precision_matrix=self.prior_prec)
        
        elif prior_arg['type'] == 'white':
            cov = prior_arg['cov']
            self.prior_prec = torch.inverse(cov).to(device)
            self.prior_prec = 0.5 * (self.prior_prec + self.prior_prec.T)
            self.prior_mean = prior_arg['mean'].to(device)
            self.prior_log_det = - torch.logdet(self.prior_prec)
            self.prior_distrib = MultivariateNormal(
                prior_arg['mean'],
                precision_matrix=self.prior_prec
                )

        elif prior_arg['type'] == 'coupled_pbc':
            self.beta_prior = prior_arg['beta']
            dim_phys = prior_arg['dim_phys']
            dim_grid = prior_arg['dim_grid']
            
            eps = 0.1
            quadratic_coef = 4 + eps
            sub_prec = (1 + quadratic_coef) * torch.eye(dim_grid)
            sub_prec -= torch.triu(torch.triu(torch.ones_like(sub_prec),
                                                      diagonal=-1).T, diagonal=-1)
            sub_prec[0, -1] = - 1  # pbc
            sub_prec[-1, 0] = - 1  # pbc

            if dim_phys == 1:
                prec = prior_arg['beta'] * sub_prec

            elif dim_phys == 2:
                # interation along one axis
                prec = torch.block_diag(*(sub_prec for d in range(dim_grid)))
                # interation along second axis
                diags = torch.triu(torch.triu(torch.ones_like(prec),
                                                      diagonal=-dim_grid).T, diagonal=-dim_grid)
                diags -= torch.triu(torch.triu(torch.ones_like(prec),
                                                      diagonal=-dim_grid+1).T, diagonal=-dim_grid+1)
                prec -= diags
                prec[:dim_grid, -dim_grid:] = - torch.eye(dim_grid)  # pbc
                prec[-dim_grid:, :dim_grid] = - torch.eye(dim_grid)  # pbc
                prec = prior_arg['beta'] * prec

            self.prior_prec = prec.to(self.device)
            self.prior_log_det = - torch.logdet(prec)
            self.prior_distrib = MultivariateNormal(
                torch.zeros((dim,), device=self.device),
                precision_matrix=self.prior_prec)

        else:
            raise NotImplementedError("Invalid prior arg type")

    def forward(self, x, return_per_block=False):
        log_det_jac = torch.zeros(x.shape[0], device=self.device)

        if return_per_block:
            xs = [x]
            log_det_jacs = [log_det_jac.clone()]

        for block in range(self.n_blocks):
            couplings = self.coupling_layers[block]

            for dt in range(self.block_depth):
                for coupling_layer in couplings:
                    x, log_det_jac = coupling_layer(x, log_det_jac)
                    # if torch.isnan(x).any():
                    #     print('layer', dt)
                    #     raise RuntimeError('Layer became Nan')

                if return_per_block:
                    xs.append(x)
                    log_det_jacs.append(log_det_jac.clone())

        if return_per_block:
            return xs, log_det_jacs
        else:
            return x, log_det_jac

    def inverse(self, x, return_per_block=False):
        log_det_jac = torch.zeros(x.shape[0], device=self.device)

        if return_per_block:
            xs = [x]
            log_det_jacs = [log_det_jac]

        for block in range(self.n_blocks):
            couplings = self.coupling_layers[::-1][block]

            for dt in range(self.block_depth):
                for coupling_layer in couplings[::-1]:
                    x, log_det_jac = coupling_layer(
                        x, log_det_jac, inverse=True)

                if return_per_block:
                    xs.append(x)
                    log_det_jacs.append(log_det_jac.clone())

        if return_per_block:
            return xs, log_det_jacs
        else:
            return x, log_det_jac

    def initialize(self):
        dim = self.dim
        coupling_layers = []

        for block in range(self.n_blocks):
            layer_dims = [self.hidden_dim_in_coupling] * \
                (self.n_layers_in_coupling - 2)
            layer_dims = [dim] + layer_dims + [dim]

            couplings = self.build_coupling_block(layer_dims)

            coupling_layers.append(nn.ModuleList(couplings))

        return nn.ModuleList(coupling_layers)

    def build_coupling_block(self, layer_dims=None, nets=None, reverse=False):
        count = 0
        coupling_layers = []
        for count in range(self.couplings_per_block):

            if self.models_type == 'mlp':
                s = MLP(layer_dims, init_scale=self.init_scale_in_coupling,
                        activation=self.hidden_activation)
                t = MLP(layer_dims, init_scale=self.init_scale_in_coupling,
                        activation=self.hidden_activation)

            elif self.models_type == 'conv':
                layer_dims = layer_dims[1:-1]  # only hidden layers
                if self.dim_phys == 2:
                    s = MLConv2d(hidden_sizes=layer_dims, kernel_size=5, 
                            in_channels=self.channels, out_channels=self.channels, 
                            init_scale=self.init_scale_in_coupling)
                    t = MLConv2d(hidden_sizes=layer_dims, kernel_size=5, 
                            in_channels=self.channels, out_channels=self.channels, 
                            init_scale=self.init_scale_in_coupling)
                elif self.dim_phys == 1:
                    s = MLConv1d(hidden_sizes=layer_dims, kernel_size=5, 
                            in_channels=self.channels, out_channels=self.channels, 
                            init_scale=self.init_scale_in_coupling)
                    t = MLConv1d(hidden_sizes=layer_dims, kernel_size=5, 
                            in_channels=self.channels, out_channels=self.channels, 
                            init_scale=self.init_scale_in_coupling)
            else:
                raise RuntimeError('type of networks not understood')

            s = s.to(self.device)
            t = t.to(self.device)

            if count % 2 == 0:
                mask = 1 - self.mask
            else:
                mask = self.mask

            if self.residual:
                dt = self.n_blocks * self.couplings_per_block * self.block_depth
                dt = 2 / dt
                # dt = 0.1
                coupling_layers.append(ResidualAffineCoupling(
                  s, t, mask, dt=dt, equivariant=self.equivariant))
            else:
                coupling_layers.append(AffineCoupling(s, t, mask))

        return coupling_layers

    def nll(self, x, from_z=False):
        """
        adding from_z option for 'reparametrization trick'
        """
        if from_z:
            z = x
            x, log_det_jac = self.forward(z)
            log_det_jac = - log_det_jac
        else:
            z, log_det_jac = self.backward(x)
        if self.prior_arg['type'] == 'white':
            z = z - self.prior_mean

        prior_ll = - 0.5 * torch.einsum('ki,ij,kj->k', z, self.prior_prec, z)
        prior_ll -= 0.5 * (self.dim * np.log(2 * np.pi) + self.prior_log_det)

        return - (prior_ll + log_det_jac)

    def sample(self, n):
        if self.prior_arg['type'] == 'standn':
            z = torch.randn(n, self.dim, device=self.device)
        else:
            z = self.prior_distrib.rsample(torch.Size([n, ])).to(self.device)

        return self.forward(z)[0]

    def U(self, x):
        """
        alias
        """ 
        return self.nll(x)

    def get_weight_scale(self):
        return sum([torch.sum(torch.square(param)) for name, param in self.named_parameters()
            if 'scale_net' in name and 'weight' in name])