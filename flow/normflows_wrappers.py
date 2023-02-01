# Wrappers for VincentStimper/normalizing-flows

# Libraries
import torch
from functools import reduce

# Wrap the base
class BaseDistributionWrapper:

    def __init__(self, base):
        self.base = base

    def sample(self, sample_shape):
        size = reduce(lambda x, y : x * y, sample_shape)
        return self.base.forward(num_samples=size)[0].reshape((*sample_shape, -1))

    def log_prob(self, value):
        return self.base.log_prob(value)

# Wrap the flow under our API
class WrappedNormFlowModel:

    def __init__(self, flow):
        self.flow = flow

    def forward(self, z):
        log_jac = torch.zeros(z.shape[0], device=z.device)
        x = z
        for flow in self.flow.flows:
            x, log_det = flow(x)
            log_jac -= log_det
        return x, log_jac

    def inverse(self, x):
        log_jac = torch.zeros(x.shape[0], device=x.device)
        z = x
        for i in range(len(self.flow.flows) - 1, -1, -1):
            z, log_det = self.flow.flows[i].inverse(z)
            log_jac += log_det
        return z, log_jac