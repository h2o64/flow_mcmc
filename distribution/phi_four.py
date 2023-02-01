# Phi Four distribution
# Taken from https://github.com/marylou-gabrie/adapt-flow-ergo/blob/main/adaflonaco/phifour_utils.py

# Libraries
import torch
import math
import torch.nn.functional as F

class PhiFour(torch.distributions.Distribution):

    arg_constraints = {
        'a': torch.distributions.constraints.positive,
        'b': torch.distributions.constraints.real,
        'dim_grid' : torch.distributions.constraints.positive_integer
    }
    support = torch.distributions.constraints.real_vector
    has_rsample = False

    def __init__(self, a, b, dim_grid, dim_phys=1,
                 bc=('dirichlet', 0),
                 tilt=None,
                 validate_args=None):
        """
        Class to handle operations around PhiFour model
        Args:
            a: coupling term coef
            b: local field coef
            dim_grid: grid size in one dimension
            dim_phys: number of dimensions of the physical grid
            tilt: None or {"val":0.7, "lambda":0.1} - for biasing distribution
        """
        self.a = a
        self.b = b
        self.dim_grid = dim_grid
        self.dim_phys = dim_phys
        self.sum_dims = tuple(i + 1 for i in range(dim_phys))

        self.bc = bc
        self.tilt = tilt
        super(PhiFour, self).__init__(validate_args=validate_args)

    def init_field(self, n_or_values):
        if isinstance(n_or_values, int):
            x = torch.rand((n_or_values,) + (self.dim_grid,) * self.dim_phys)
            x = x * 2 - 1
        else:
            x = n_or_values
        return x
    
    def reshape_to_dimphys(self, x):
        if self.dim_phys == 2:
            x_ = x.reshape(-1, self.dim_grid, self.dim_grid)
        else:
            x_ = x
        return x_

    def V(self, x):
        x = self.reshape_to_dimphys(x)
        coef = self.a * self.dim_grid
        V = ((1 - x ** 2) ** 2 / 4 + self.b * x).sum(self.sum_dims) / coef
        if self.tilt is not None: 
            tilt = (self.tilt['val'] - x.mean(self.sum_dims)) ** 2 
            tilt = self.tilt["lambda"] * tilt / (4 * self.dim_grid)
            V += tilt
        return V

    def U(self, x):
        # Does not include the temperature! need to be explicitely added in Gibbs factor
        assert self.dim_phys < 3
        x = self.reshape_to_dimphys(x)

        if self.bc[0] == 'dirichlet':
            x_ = F.pad(input=x, pad=(1,) * (2 * self.dim_phys), mode='constant',
                      value=self.bc[1])
        elif self.bc[0] == 'pbc':
            #adding "channel dimension" for circular torch padding 
            x_ = x.unsqueeze(0) 
            #only pad one side, not to double count gradients at the edges
            x_ = F.pad(input=x_, pad=(1,0,) * (self.dim_phys), mode='circular')
            x_.squeeze_(0) 
        else:
            raise NotImplementedError("Only dirichlet and periodic BC"         
                                      "implemeted for now")

        if self.dim_phys == 2:
            grad_x = ((x_[:, 1:, :-1] - x_[:, :-1, :-1]) ** 2 / 2)
            grad_y = ((x_[:, :-1, 1:] - x_[:, :-1, :-1]) ** 2 / 2)
            grad_term = (grad_x + grad_y).sum(self.sum_dims)
        else:
            grad_term = ((x_[:, 1:] - x_[:, :-1]) ** 2 / 2).sum(self.sum_dims)
        
        coef = self.a * self.dim_grid
        return grad_term * coef + self.V(x)

    def log_prob(self, x):
        return -self.U(x)

# Physics informed base
class PhiFourBase(torch.distributions.Distribution):

    arg_constraints = {}
    support = torch.distributions.constraints.real_vector
    has_rsample = True

    def __init__(self, dim, device, a=0.1, b=0.0, alpha=0.1, beta=1, prior_type='coupled', mean=None, cov=None, dim_phys=1, validate_args=None):
        # Build the prior
        self.dim = dim
        self.device = device
        self.prior_type = prior_type
        if prior_type == 'uncoupled':
            self.prior_prec = a * torch.eye(dim).to(device)
            self.prior_log_det = - torch.logdet(self.prior_prec)
            self.prior_distrib = torch.distributions.MultivariateNormal(
                torch.zeros((dim,), device=self.device),
                precision_matrix=self.prior_prec)
        elif prior_type == 'coupled':
            self.beta_prior = beta
            self.coef = alpha * dim
            prec = torch.eye(dim) * (3 * self.coef + 1 / self.coef)
            prec -= self.coef * torch.triu(torch.triu(torch.ones_like(prec),
                                                      diagonal=-1).T, diagonal=-1)
            prec = beta * prec
            self.prior_prec = prec.to(self.device)
            self.prior_log_det = - torch.logdet(prec)
            self.prior_distrib = torch.distributions.MultivariateNormal(
                torch.zeros((dim,), device=self.device),
                precision_matrix=self.prior_prec)
        elif prior_type == 'white':
            cov = cov
            self.prior_prec = torch.inverse(cov).to(device)
            self.prior_prec = 0.5 * (self.prior_prec + self.prior_prec.T)
            self.prior_mean = mean.to(device)
            self.prior_log_det = - torch.logdet(self.prior_prec)
            self.prior_distrib = torch.distributions.MultivariateNormal(
                mean,
                precision_matrix=self.prior_prec
                )
        elif prior_type == 'coupled_pbc':
            self.beta_prior = beta
            dim_grid = dim / dim_phys
            eps = 0.1
            quadratic_coef = 4 + eps
            sub_prec = (1 + quadratic_coef) * torch.eye(dim_grid)
            sub_prec -= torch.triu(torch.triu(torch.ones_like(sub_prec),
                                                      diagonal=-1).T, diagonal=-1)
            sub_prec[0, -1] = - 1  # pbc
            sub_prec[-1, 0] = - 1  # pbc

            if dim_phys == 1:
                prec = beta * sub_prec

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
                prec = beta * prec

            self.prior_prec = prec.to(self.device)
            self.prior_log_det = - torch.logdet(prec)
            self.prior_distrib = torch.distributions.MultivariateNormal(
                torch.zeros((dim,), device=self.device),
                precision_matrix=self.prior_prec)

        else:
            raise NotImplementedError("Invalid prior arg type")
        super(PhiFourBase, self).__init__(validate_args=validate_args)

    def log_prob(self, value):
        if self.prior_type == 'white':
            value = value - self.prior_mean
        prior_ll = - 0.5 * torch.einsum('ki,ij,kj->k', value, self.prior_prec, value)
        prior_ll -= 0.5 * (self.dim * math.log(2 * math.pi) + self.prior_log_det)
        return prior_ll

    def rsample(self, sample_shape=torch.Size()):
        return self.prior_distrib.rsample(sample_shape=sample_shape).to(self.device)