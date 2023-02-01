# Elliptical Slice Sampler

# Libraries
import torch
from .mcmc import MCMC
from tqdm import trange

# Elliptical Slice Sampler
class ESS(MCMC):

	def __init__(self, covariance_matrix=None):
		self.use_chol_factor = False
		if covariance_matrix is not None:
			self.covariance_matrix_col = torch.linalg.cholesky(covariance_matrix)
			self.use_chol_factor = True
		# Call parent constructor
		super(ESS, self).__init__()

	def one_step(self, x_s_t, target, temp):
		# Choose an ellipse
		nu = torch.randn_like(x_s_t)
		if self.use_chol_factor:
			nu = torch.matmul(self.covariance_matrix_col, nu.T).T
		# Log-likelihood threshold
		u = torch.rand(x_s_t.shape[0], device=x_s_t.device)
		log_y = (target(x_s_t) / temp) + torch.log(u)
		# Define a bracket
		theta = - 2.0 * torch.pi * torch.rand((x_s_t.shape[0],), device=x_s_t.device) + 2.0 * torch.pi
		theta_min, theta_max = theta.clone() - 2.0 * torch.pi, theta.clone()
		# Try to accept
		x_prop = x_s_t * torch.cos(theta)[:,None] + nu * torch.sin(theta)[:,None]
		n_tries = torch.ones((x_s_t.shape[0],), device=x_s_t.device)
		mask = (target(x_prop) / temp) <= log_y
		while torch.any(mask):
			# Shrink the bracket
			theta_min = torch.where(mask & (theta < 0), theta, theta_min)
			theta_max = torch.where(mask & (theta >= 0), theta, theta_max)
			r = torch.rand((mask.sum(),), device=x_s_t.device)
			theta[mask] = (theta_min[mask] - theta_max[mask]) * r + theta_max[mask]
			# Try a new point
			x_prop[mask] = x_s_t[mask] * torch.cos(theta[mask])[:,None] + nu[mask] * torch.sin(theta[mask])[:,None]
			# Increment the number of tries
			n_tries[mask] += 1
			# Update the mask
			mask[mask.clone()] = (target(x_prop[mask]) / temp) <= log_y[mask]
		return x_prop, n_tries.mean()

	def sample(self, x_s_t_0, n_steps, target, temp=1.0, warmup_steps=0, verbose=False):
		x_s_t = x_s_t_0.clone()
		x_s_ts = torch.zeros((n_steps, *x_s_t.shape), device=x_s_t.device)
		num_tries = torch.zeros((n_steps,)).cpu()
		r = trange(n_steps) if verbose else range(n_steps)
		# Warmup steps
		if warmup_steps > 0:
			for _ in range(warmup_steps):
				x_s_t, _ = self.one_step(x_s_t=x_s_t, target=target, temp=temp)
		# Sampling steps
		for ell in r:
			x_s_ts[ell], num_tries[ell] = self.one_step(
				x_s_t=x_s_t,
				target=target,
				temp=temp
			)
		self.diagnostics['num_tries'] = num_tries
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
	sampler = ESS()
	samples = sampler.sample(
		x_s_t_0=target.sample(sample_shape=(64,)),
		n_steps=400,
		target=target.log_prob,
		verbose=True
	).detach().cpu()
	print('Number of tries = {} +/- {}'.format(sampler.get_diagnostics('num_tries').mean(), sampler.get_diagnostics('num_tries').std()))

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