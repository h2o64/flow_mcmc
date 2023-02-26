# Iterated Sampling Importance Resampling

# Libraries
import torch
from .mcmc import MCMC
from tqdm import trange

# Iterated Sampling Importance Resampling
class iSIR(MCMC):

	def __init__(self, proposal, N, flow=None, compute_ess=False):
		# Save the parameters
		self.N = N
		self.proposal = proposal
		self.flow = flow
		self.compute_ess = compute_ess
		# Call parent constructor
		super(iSIR, self).__init__()

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

	def one_step(self, x_s_t, target, temp, target_x_s_t=None, prop_x_s_t=None):
		batch_size = x_s_t.shape[0]
		# Disable gradients
		with torch.no_grad():
			# Generate particles
			particles = self.sample_proposal(batch_size * self.N).view((batch_size, self.N, -1))
			# Insert the previous point
			particles[torch.arange(batch_size),0] = x_s_t
			particles = particles.view((batch_size * self.N, -1))
			# Compute the importance weights
			log_prob_proposal = self.eval_proposal(particles)
			log_prob_target = target(particles) / temp
			log_weights = log_prob_target - log_prob_proposal
			log_weights = log_weights.view((batch_size, self.N))
			particles = particles.view((batch_size, self.N, -1))
			# Normalize weights
			log_weights = log_weights - log_weights.logsumexp(dim=-1, keepdim=True)
			weights = torch.exp(log_weights)
			weights[weights != weights] = 0.0
			weights[weights.sum(1) == 0.0] = 1.0
			# Compute the ESS
			if self.compute_ess:
				ess = 1.0 / torch.sum(torch.square(weights), dim=-1)
				ess = ess.mean().cpu()
			else:
				ess = 0.0
			# Select the particles
			indices = torch.multinomial(weights, 1).squeeze()
			x_s_t = particles[torch.arange(batch_size), indices]
			# Compute the acceptance
			mean_acc = (indices != 0).float().mean().cpu()
		# Return the data
		return x_s_t, target_x_s_t, prop_x_s_t, mean_acc, ess

	def sample(self, x_s_t_0, n_steps, target, temp=1.0, warmup_steps=0, verbose=False):
		x_s_t = x_s_t_0.clone()
		x_s_ts = torch.zeros((n_steps, *x_s_t.shape), device=x_s_t.device)
		global_accepts = torch.zeros((n_steps,)).cpu()
		steps_ess = torch.zeros((n_steps,)).cpu()
		r = trange(n_steps) if verbose else range(n_steps)
		# Warmup steps
		if warmup_steps > 0:
			for _ in range(warmup_steps):
				x_s_t, _, _, _, _ = self.one_step(x_s_t=x_s_t, target=target, temp=temp)
		# Sampling steps
		for ell in r:
			x_s_ts[ell], _, _, global_accepts[ell], steps_ess[ell] = self.one_step(
				x_s_t=x_s_t,
				target=target,
				temp=temp
			)
			x_s_t = x_s_ts[ell]
		self.diagnostics['global_acceptance'] = global_accepts
		if self.compute_ess:
			self.diagnostics['ess'] = steps_ess
		return x_s_ts

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
	sampler = iSIR(proposal=proposal, N=64)
	samples = sampler.sample(
		x_s_t_0=target.sample(sample_shape=(32,)),
		n_steps=400,
		target=target.log_prob,
		verbose=True
	).detach().cpu()
	print('Acceptance = ', sampler.get_diagnostics('global_acceptance').mean())
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
