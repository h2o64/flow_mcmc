# Random Walk Metropolis Hastings

# Libraries
import torch
from .mcmc import MCMC
from .utils import heuristics_step_size
from tqdm import trange

# Random Walk Metropolis Hastings
class RWMH(MCMC):

	def __init__(self, step_size, target_acceptance=None):
		# Step size adaptation
		if target_acceptance is None:
			self.adapt_stepsize = lambda x : self.step_size
		else:
			self.adapt_stepsize = lambda x : heuristics_step_size(self.step_size, x, target_acceptance=target_acceptance,
				factor=1.05, tol=0.03)
		# Step size
		self.step_size = step_size
		# Call parent constructor
		super(RWMH, self).__init__()

	def one_step(self, x_s_t, target, temp):
		# Disable gradient computation
		with torch.no_grad():
			# Make a proposal
			x_prop = x_s_t + self.step_size * torch.randn_like(x_s_t, device=x_s_t.device)
			# Compute the Metropolis-Hastings ratio
			log_acc = (target(x_prop) - target(x_s_t)) / temp
			mask = torch.log(torch.rand((x_s_t.shape[0],), device=x_s_t.device)) < log_acc
			x_s_t.data[mask] = x_prop[mask]
			# Compute mean acceptance propobability
			if x_s_t.shape[0] > 1:
				mean_acc = torch.mean(mask.float()).cpu()
			else:
				mean_acc = min(float(torch.exp(log_acc).cpu()), 1.0)
			self.step_size = self.adapt_stepsize(mean_acc)
		# Return the data
		return x_s_t, mean_acc

	def sample(self, x_s_t_0, n_steps, target, temp=1.0, warmup_steps=0, verbose=False):
		x_s_t = x_s_t_0.clone()
		x_s_ts = torch.zeros((n_steps, *x_s_t.shape), device=x_s_t.device)
		local_accepts = torch.zeros((n_steps,)).cpu()
		r = trange(n_steps) if verbose else range(n_steps)
		# Warmup steps
		if warmup_steps > 0:
			for _ in range(warmup_steps):
				x_s_t, _ = self.one_step(x_s_t=x_s_t, target=target, temp=temp)
		# Sampling
		for ell in r:
			x_s_ts[ell], local_accepts[ell] = self.one_step(
				x_s_t=x_s_t,
				target=target,
				temp=temp
			)
		self.diagnostics['local_acceptance'] = local_accepts
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
	sampler = RWMH(step_size=1e-2, target_acceptance=0.65)
	samples = sampler.sample(
		x_s_t_0=target.sample(sample_shape=(64,)),
		n_steps=400,
		target=target.log_prob,
		verbose=True
	).detach().cpu()
	print('Acceptance = ', sampler.get_diagnostics('local_acceptance').mean())

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
