# Classic Importance Resampling

# Libraries
import torch
from .mcmc import MCMC

# Classic Importance Resampling
class IS(MCMC):

	def __init__(self, proposal, N, flow=None):
		# Save the parameters
		self.N = N
		self.proposal = proposal
		self.flow = flow
		# Call parent constructor
		super(IS, self).__init__()

	def sample_proposal(self, batch_size):
		x = self.proposal.sample(sample_shape=(batch_size,))
		if self.flow is not None:
			x = self.flow.forward(x)[0]
		return x

	def eval_proposal(self, samples):
		if self.flow is not None:
			samples_z, log_jac = self.flow.inverse(samples)
		else:
			samples_z, log_jac = samples, 0.0
		return self.proposal.log_prob(samples_z) + log_jac

	def sample(self, x_s_t_0, n_steps, target, temp=1.0, warmup_steps=0, verbose=False):
		# Compute the particles and their weights
		particles = self.sample_proposal(self.N)
		proposal_log_prob = self.eval_proposal(particles)
		target_log_prob = target(particles) / temp
		log_weights = target_log_prob - proposal_log_prob
		log_weights = log_weights - log_weights.logsumexp(dim=0)
		weights = torch.exp(log_weights)
		# Save the weights
		self.diagnostics['weights'] = weights
		# Compute the ESS
		ess = 1.0 / torch.sum(torch.square(weights))
		ess = ess.mean().cpu()
		self.diagnostics['ess'] = ess
		return particles.view((self.N, 1, x_s_t_0.shape[-1]))

# DEBUG
if __name__ == "__main__":

	# Libraries
	import numpy as np
	import matplotlib.pyplot as plt

	# Torch device
	device = torch.device('cuda')

	# Target distribution
	target = torch.distributions.MultivariateNormal(
		loc=torch.zeros((2,)).to(device),
		covariance_matrix=torch.diag(torch.logspace(-0.5,0.5,2)).to(device)
	)

	# Create the sampler
	proposal = torch.distributions.MultivariateNormal(
		loc=torch.zeros((2,)).to(device),
		covariance_matrix=torch.eye(2).to(device)
	)
	sampler = IS(proposal=proposal, N=2048)
	samples = sampler.sample(
		x_s_t_0=target.sample(sample_shape=(32,)),
		n_steps=400,
		target=target.log_prob,
		verbose=True
	).detach().cpu()
	print('ESS = ', sampler.get_diagnostics('ess').mean())

	# Make a grid
	x = np.linspace(-3, 3, 128)
	y = np.linspace(-4, 4, 128)
	X, Y = np.meshgrid(x, y)
	Z = torch.Tensor(torch.stack([torch.Tensor(X),torch.Tensor(Y)], dim=2)).to(device)

	# Compute the grid of the target
	log_prob = target.log_prob(Z.view((-1,2))).view((128,128)).detach().cpu()

	# Display everything
	plt.figure(figsize=(10,10))
	plt.contour(X, Y, torch.exp(log_prob), 20, linestyles=':', colors='k')
	plt.scatter(samples[:,0,0], samples[:,0,1], alpha=0.75, color='red')
	plt.show()
