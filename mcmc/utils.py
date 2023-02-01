# Various tools

# Libraries
import torch
import math

# Heuristic for adaptative step size
def heuristics_step_size(stepsize, mean_acceptance, target_acceptance=0.75, factor=1.03, tol=0.01):
	if mean_acceptance - target_acceptance > tol:
		return stepsize * factor
	if target_acceptance - mean_acceptance > tol:
		return stepsize / factor
	return min(stepsize, 1.0)

# Sample according to multivariate normal with diagonal matrix
def sample_multivariate_normal_diag(batch_size, mean, variance):
	dim = mean.shape[-1]
	z = torch.randn((batch_size, dim), device=mean.device)
	return math.sqrt(variance) * z + mean

# Evaluate the log density of multivariate normal with diagonal matrix
# WARNING: Single sample along a batch size
# Remove a multiplicative factor : - 0.5 * dim * torch.log(2.0 * torch.pi * variance)
def log_prob_multivariate_normal_diag(samples, mean, variance):
	return (-0.5 * torch.sum(torch.square(samples - mean), dim=-1) / variance)

# Evaluate the log density of centered standard multivariate normal
log_sqrt_2_pi = math.log(math.sqrt(2.0 * math.pi))
def log_prob_multivariate_normal(samples):
	return -0.5 * torch.sum(torch.square(samples), dim=-1)

# Evaluate the log density of centered multivariate normal with diagonal matrix (different coeffficients)
def log_prob_multivariate_normal_diag_diff(samples, variances):
	return -0.5 * torch.sum(torch.square(samples) / variances, dim=-1)

# Compute IMH/IS weights
def compute_imh_is_weights(flow, target, proposal, batch_size, samples_target=None):
    # Samples from the target
    if samples_target is None:
    	samples_target = target.sample(sample_shape=(batch_size,))
    # Samples from the flow
    samples_flow, _ = flow.forward(proposal.sample(sample_shape=(batch_size,)))
    # Get the log_probs
    prob_target_of_target = target.log_prob(samples_target)
    prob_target_of_flow = target.log_prob(samples_flow)
    samples_target_z, log_jac_samples_target = flow.inverse(samples_target)
    prob_flow_of_target = proposal.log_prob(samples_target_z) + log_jac_samples_target
    samples_flow_z, log_jac_samples_flow = flow.inverse(samples_flow)
    prob_flow_of_flow = proposal.log_prob(samples_flow_z) + log_jac_samples_flow
    # Compute IMH weights
    imh_log_weights = (prob_target_of_flow - prob_flow_of_flow) - (prob_target_of_target - prob_flow_of_target)
    imh_log_weights = torch.minimum(torch.zeros_like(imh_log_weights), imh_log_weights)
    # Compute IS weights
    is_log_weights = prob_target_of_flow - prob_flow_of_flow
    is_log_weights = is_log_weights - is_log_weights.logsumexp(dim=0)
    is_weights = torch.exp(is_log_weights)
    # Compute participation ratio
    participation_ratio = 1.0 / torch.sum(torch.square(is_weights))
    return torch.exp(imh_log_weights), participation_ratio / batch_size

# DEBUG
if __name__ == "__main__":

	# Sample from multivaritate normal
	x = sample_multivariate_normal_diag(
		batch_size=64,
		mean=torch.randn((64, 16)),
		variance=torch.tensor(1e-2)
	)
	print("{} == (64, 16)".format(x.shape))

	# Log probability of multivariate normal
	log_prob = log_prob_multivariate_normal_diag(
		samples=torch.randn((64,16)),
		mean=torch.randn((64,16)),
		variance=torch.tensor(1e-2)
	)
	print("{} == (64,)".format(log_prob.shape))
