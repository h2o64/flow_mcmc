# Global local algorithms

# Bibliography 
# - "Adaptive Monte Carlo augmented with normalizing flows" (Marylou Gabrié, Grant M. Rotskoff, Eric Vanden-Eijnden)
# - "Local-Global MCMC kernels: the best of both worlds" (Sergey Samsonov, Evgeny Lagutin, Marylou Gabrié, Alain Durmus, Alexey Naumov, Eric Moulines)

# Libraries
import torch
from .mcmc import MCMC
from .isir import iSIR
from .imh import IndependentMetropolisHastings
from tqdm import trange

# Global Local MCMC sampler
class GlobalLocal(MCMC):

	def __init__(self, global_mcmc, local_mcmc, n_local_mcmc):
		# Global and local kernels
		self.global_mcmc = global_mcmc
		self.local_mcmc = local_mcmc
		self.n_local_mcmc = n_local_mcmc
		# Call parent constructor
		super(GlobalLocal, self).__init__()

	def sample(self, x_s_t_0, n_steps, target, temp=1.0, verbose=False, warmup_steps=0):
		x_s_t = x_s_t_0.clone()
		if isinstance(self.global_mcmc, IndependentMetropolisHastings):
			target_x_s_t = target(x_s_t)
			prop_x_s_t = self.global_mcmc.eval_proposal(x_s_t)
		else:
			target_x_s_t = None
			prop_x_s_t = None
		x_s_ts = torch.zeros((n_steps, *x_s_t.shape), device=x_s_t.device)
		local_accepts = torch.zeros((n_steps,)).cpu()
		global_accepts = torch.zeros((n_steps,)).cpu()
		other_global_diagnostics = torch.zeros((n_steps,)).cpu()
		step_count, local_step_count, global_step_count = 0, 0, 0
		r = trange(n_steps) if verbose else range(n_steps)
		# Do a little warmup of the local steps if needed
		if warmup_steps > 0:
			x_s_t, target_x_s_t, prop_x_s_t, _, _ = self.global_mcmc.one_step(
				x_s_t=x_s_t,
				target_x_s_t=target_x_s_t,
				prop_x_s_t=prop_x_s_t,
				target=target,
				temp=temp
			)
			x_s_t = self.local_mcmc.sample(
				x_s_t_0=x_s_t,
				warmup_steps=int(warmup_steps * 0.90),
				n_steps=int(warmup_steps * 0.10),
				target=target,
				temp=temp,
				verbose=False
			).detach().clone()[-1]
			self.local_mcmc.reset_diagnostics()
		for ell in r:
			if step_count >= n_steps:
				break
			elif ell % 2 == 0:
				# Global step
				x_s_ts[step_count], target_x_s_t, prop_x_s_t, global_accepts[global_step_count], other_global_diagnostics[global_step_count] = self.global_mcmc.one_step(
					x_s_t=x_s_t,
					target_x_s_t=target_x_s_t,
					prop_x_s_t=prop_x_s_t,
					target=target,
					temp=temp
				)
				x_s_t = x_s_ts[step_count]
				step_count += 1
				global_step_count += 1
			else:
				# Local step
				n_steps_local_ = min(self.n_local_mcmc, n_steps - step_count)
				local_samples = self.local_mcmc.sample(
					x_s_t_0=x_s_t,
					n_steps=n_steps_local_,
					target=target,
					temp=temp,
					verbose=False
				)
				x_s_ts[step_count:step_count+local_samples.shape[0]] = local_samples
				x_s_t = local_samples[-1]
				step_count += local_samples.shape[0]
				local_step_count += 1
				local_acc = self.local_mcmc.get_diagnostics('local_acceptance')
				if isinstance(local_acc, float):
					local_accepts[local_step_count] = local_acc
				else:
					local_accepts[local_step_count] = local_acc.mean()
		self.diagnostics['local_acceptance'] = local_accepts[:local_step_count]
		self.diagnostics['global_acceptance'] = global_accepts[:global_step_count]
		self.diagnostics['other_global_diagnostics'] = other_global_diagnostics[:global_step_count]
		return x_s_ts[:step_count]

class Ex2MCMC(GlobalLocal):

	def __init__(self, proposal, N, local_mcmc, n_local_mcmc, flow=None):
		# Make the global kernel
		self.N = N
		global_mcmc = iSIR(N=N, proposal=proposal, flow=flow)
		# Make the global-local sampler
		super(Ex2MCMC, self).__init__(
			global_mcmc=global_mcmc,
			local_mcmc=local_mcmc,
			n_local_mcmc=n_local_mcmc
		)

	def sample(self, x_s_t_0, n_steps, target, temp=1.0, verbose=False, warmup_steps=0):
		samples = super(Ex2MCMC, self).sample(
			x_s_t_0=x_s_t_0,
			n_steps=n_steps,
			target=target,
			temp=temp,
			verbose=verbose,
			warmup_steps=warmup_steps
		)
		self.diagnostics['ess'] = self.diagnostics['other_global_diagnostics'].clone()
		del self.diagnostics['other_global_diagnostics']
		return samples

class AdaptiveMCMC(GlobalLocal):

	def __init__(self, proposal, local_mcmc, n_local_mcmc, flow=None):
		# Make the global kernel
		global_mcmc = IndependentMetropolisHastings(proposal=proposal, flow=flow)
		# Make the global-local sampler
		super(AdaptiveMCMC, self).__init__(
			global_mcmc=global_mcmc,
			local_mcmc=local_mcmc,
			n_local_mcmc=n_local_mcmc
		)

	def sample(self, x_s_t_0, n_steps, target, temp=1.0, verbose=False, warmup_steps=0):
		samples = super(AdaptiveMCMC, self).sample(
			x_s_t_0=x_s_t_0,
			n_steps=n_steps,
			target=target,
			temp=temp,
			verbose=verbose,
			warmup_steps=warmup_steps
		)
		del self.diagnostics['other_global_diagnostics']
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

	# Create the sampler
	proposal = torch.distributions.MultivariateNormal(
		loc=torch.zeros((2,)).to(device),
		covariance_matrix=torch.eye(2).to(device)
	)
	local_sampler = MALA(step_size=1e-2, target_acceptance=0.65)
	sampler_ex2 = Ex2MCMC(
		proposal=proposal,
		N=16,
		local_mcmc=local_sampler,
		n_local_mcmc=4
	)
	samples_ex2 = sampler_ex2.sample(
		x_s_t_0=target.sample(sample_shape=(32,)),
		n_steps=400,
		target=target.log_prob,
		verbose=True
	).detach().cpu()
	print('Acceptance local (Ex2MCMC) = ', sampler_ex2.get_diagnostics('local_acceptance').mean())
	print('Acceptance global (Ex2MCMC) = ', sampler_ex2.get_diagnostics('global_acceptance').mean())
	print('Effective Sample Size (Ex2MCMC) = ', sampler_ex2.get_diagnostics('ess').mean())
	local_sampler = MALA(step_size=1e-2, target_acceptance=0.65)
	sampler_adaptive = AdaptiveMCMC(
		proposal=proposal,
		local_mcmc=local_sampler,
		n_local_mcmc=4
	)
	samples_adaptive = sampler_adaptive.sample(
		x_s_t_0=target.sample(sample_shape=(32,)),
		n_steps=400,
		target=target.log_prob,
		verbose=True
	).detach().cpu()
	print('Acceptance local (AdaptiveMCMC) = ', sampler_adaptive.get_diagnostics('local_acceptance').mean())
	print('Acceptance global (AdaptiveMCMC) = ', sampler_adaptive.get_diagnostics('global_acceptance').mean())

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
	plt.scatter(samples_ex2[:,0,0], samples_ex2[:,0,1], alpha=0.75, color='red')
	plt.scatter(samples_adaptive[:,0,0], samples_adaptive[:,0,1], alpha=0.75, color='blue')
	plt.show()
