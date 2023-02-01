# Gaussian mixture model

# Description:
# This is a mixture of 4 standard normal gaussian modes located in 
# (-a,-a,-a, ..., -a, -a, -a)
# (+a,+a,+a, ..., +a, +a, +a)
# (-a,-a,-a, ..., +a, +a, +a)
# (+a,+a,+a, ..., -a, -a, -a)
# with a = 2 sqrt(2 * log(dim))
# The proposal is a centered gaussian with scale = 3sqrt(2) + 1
# The value of "a" ensures that modes are very well separated
# The values of "scale" ensure that the proposal covers all the modes

# Libraires
import torch
import math

# Make the target
def make_target(dim, device, alpha=0.5910, radius=1.0):
    # Make the gaussian mixture
    half_dim = int(dim / 2)
    pi = torch.FloatTensor([1/4]*4).to(device)
    means = torch.ones((4, dim)).to(device)
    means[0,:half_dim] *= -1.0
    means[2] *= -1.0
    means[3,half_dim:] *= -1.0
    means *= alpha * math.sqrt(2.0 * math.log(dim))
    covs = torch.stack([pow(radius,2) * torch.eye(dim)] * 4).to(device)
    mix = torch.distributions.Categorical(pi)
    comp = torch.distributions.MultivariateNormal(
        loc=means,
        covariance_matrix=covs
    )
    target = torch.distributions.MixtureSameFamily(mix, comp)
    return target

# Make the proposal
def make_proposal(target):
    proposal = torch.distributions.MultivariateNormal(
        loc=target.mean,
        covariance_matrix=torch.diag(torch.square(target.stddev))
    )
    return proposal
