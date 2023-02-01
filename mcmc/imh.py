# Independent Metropolis Hastings algorithm

# Libraries
import torch
from .mcmc import MCMC
from tqdm import trange

# Independent Metropolis Hastings algorithm
class IndependentMetropolisHastings(MCMC):

	def __init__(self, proposal, flow=None):
		# Save the parameters
		self.proposal = proposal
		self.flow = flow
		# Call parent constructor
		super(IndependentMetropolisHastings, self).__init__()

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

	def one_step(self, x_s_t, target_x_s_t, prop_x_s_t, target, temp):
		batch_size = x_s_t.shape[0]
		# Disable gradients
		with torch.no_grad():
			# Sample the proposal
			x_prop = self.sample_proposal(batch_size)
			target_x_prop = target(x_prop)
			prop_x_prop = self.eval_proposal(x_prop)
			# Compute the Metropolis Hasting ratio
			joint_prop = (target_x_prop / temp) - prop_x_prop
			joint_orig = (target_x_s_t / temp) - prop_x_s_t
			# Acceptance step
			mask = torch.log(torch.rand_like(joint_orig, device=x_s_t.device)) < joint_prop - joint_orig
			x_s_t.data[mask] = x_prop[mask]
			target_x_s_t.data[mask] = target_x_prop[mask]
			prop_x_s_t.data[mask] = prop_x_prop[mask]
			# Compute mean acceptance propobability
			mean_acc = torch.mean(mask.float()).cpu()
		# Return the data
		return x_s_t, target_x_s_t, prop_x_s_t, mean_acc, 0.0

	def sample(self, x_s_t_0, n_steps, target, temp=1.0, warmup_steps=0, verbose=False):
		x_s_t = x_s_t_0.clone()
		target_x_s_t = target(x_s_t)
		prop_x_s_t = self.eval_proposal(x_s_t)
		x_s_ts = torch.zeros((n_steps, *x_s_t.shape), device=x_s_t.device)
		global_accepts = torch.zeros((n_steps,)).cpu()
		r = trange(n_steps) if verbose else range(n_steps)
		# Warmup steps
		if warmup_steps > 0:
			for _ in range(warmup_steps):
				x_s_t, target_x_s_t, prop_x_s_t, _, _ = self.one_step(x_s_t=x_s_t, target_x_s_t=target_x_s_t,
					prop_x_s_t=prop_x_s_t, target=target, temp=temp)
		# Sampling
		for ell in r:
			x_s_ts[ell], target_x_s_t, prop_x_s_t, global_accepts[ell], _ = self.one_step(
				x_s_t=x_s_t,
				target_x_s_t=target_x_s_t,
				prop_x_s_t=prop_x_s_t,
				target=target,
				temp=temp
			)
		self.diagnostics['global_acceptance'] = global_accepts
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
	sampler = IndependentMetropolisHastings(proposal=proposal)
	samples = sampler.sample(
		x_s_t_0=target.sample(sample_shape=(32,)),
		n_steps=400,
		target=target.log_prob,
		verbose=True
	).detach().cpu()
	print('Acceptance = ', sampler.get_diagnostics('global_acceptance').mean())

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
