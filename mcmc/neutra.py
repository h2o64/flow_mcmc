# NeuTra-MCMC trick

# Bibliography
# - NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using Neural Transport (Matthew Hoffman, Pavel Sountsov, Joshua V. Dillon, Ian Langmore, Dustin Tran, Srinivas Vasudevan)

# Libraries
import torch
from .mcmc import MCMC

# NeuTra-lize an MCMC sampler
class NeuTra(MCMC):

	def __init__(self, inner_sampler, flow, save_latent=False, use_reshape=False):
		# Save the inner sampler
		self.inner_sampler = inner_sampler
		# Save the flow
		self.flow = flow
		# Save the latent samples or to reshape samples
		self.save_latent = save_latent
		self.use_reshape = use_reshape
		# Call the parent constructor
		super(NeuTra, self).__init__()

	def sample(self, x_s_t_0, n_steps, target, temp=1.0, warmup_steps=0, verbose=False):

		# Create a new target in the latent space
		def latent_target(z):
			x, log_jac = self.flow.forward(z)
			return (target(x) / temp) + log_jac

		# Sample in the latent space
		samples_latent = self.inner_sampler.sample(
			x_s_t_0=self.flow.inverse(x_s_t_0)[0].detach(),
			n_steps=n_steps,
			target=latent_target,
			temp=temp,
			warmup_steps=warmup_steps,
			verbose=verbose
		)
		for k in self.inner_sampler.diagnostics.keys():
			self.diagnostics[k] = self.inner_sampler.diagnostics[k]
		if self.save_latent:
			self.diagnostics['latent_samples'] = samples_latent

		# Push the samples back in the data space
		samples_shape = samples_latent.shape
		if self.use_reshape:
			samples = self.flow.forward(samples_latent.reshape(-1, samples_shape[-1]))[0].reshape(samples_shape)
		else:
			samples = self.flow.forward(samples_latent.view(-1, samples_shape[-1]))[0].view(samples_shape)
		return samples

# DEBUG
if __name__ == "__main__":

	# Libraries
	import numpy as np
	import matplotlib.pyplot as plt
	from mala import MALA

	# Torch device
	device = torch.device('cuda')

	# Target distribution
	target = torch.distributions.MultivariateNormal(
		loc=torch.zeros((2,)).to(device),
		covariance_matrix=torch.diag(torch.logspace(-0.5,0.5,2)).to(device)
	)

	# Build a perfect flow
	class PerfectFlow:

		def __init__(self, covariance_matrix):
			self.a = torch.linalg.cholesky(covariance_matrix)
			self.a_inv = torch.linalg.inv(self.a)
			self.log_jac = torch.linalg.slogdet(self.a).logabsdet

		def forward(self, z):
		    x = torch.matmul(self.a, z[...,None]).squeeze(-1)
		    log_jac = self.log_jac * torch.ones(z.shape[:-1], device=z.device)
		    return x, log_jac

		def inverse(self, x):
		    z = torch.matmul(self.a_inv, x[...,None]).squeeze(-1)
		    log_jac = -self.log_jac * torch.ones(x.shape[:-1], device=x.device)
		    return z, log_jac

		def __call__(self, z):
		    return self.forward(z)

	# Create the sampler
	flow = PerfectFlow(covariance_matrix=target.covariance_matrix)
	inner_sampler = MALA(step_size=1e-2, target_acceptance=0.65)
	sampler = NeuTra(inner_sampler=inner_sampler, flow=flow)
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
