# Local samplers based on Pyro

# Libraries
import torch
from .mcmc import MCMC
from pyro.infer import NUTS as NUTS_pyro
from pyro.infer import HMC as HMC_pyro
from pyro.infer import MCMC as MCMC_pyro
from pyro.infer.mcmc.api import _UnarySampler
from pyro.ops.integrator import potential_grad

# Slightly customized NUTS/HMC
class CustomPyroHMC(HMC_pyro):
    def setup(self, warmup_steps, *args, **kwargs):
        self._warmup_steps = warmup_steps
        if self.model is not None:
            self._initialize_model_properties(args, kwargs)
        if self.initial_params:
            z = {k: v.detach() for k, v in self.initial_params.items()}
            z_grads, potential_energy = potential_grad(self.potential_fn, z)
        else:
            z_grads, potential_energy = {}, self.potential_fn(self.initial_params)
        self._cache(self.initial_params, potential_energy, z_grads)
        if self.initial_params and not kwargs['skip_adaptation_reset']:
            self._initialize_adapter()

class CustomPyroNUTS(NUTS_pyro):
    def setup(self, warmup_steps, *args, **kwargs):
        self._warmup_steps = warmup_steps
        if self.model is not None:
            self._initialize_model_properties(args, kwargs)
        if self.initial_params:
            z = {k: v.detach() for k, v in self.initial_params.items()}
            z_grads, potential_energy = potential_grad(self.potential_fn, z)
        else:
            z_grads, potential_energy = {}, self.potential_fn(self.initial_params)
        self._cache(self.initial_params, potential_energy, z_grads)
        if self.initial_params and not kwargs['skip_adaptation_reset']:
            self._initialize_adapter()

# Generic Pyro class
class PyroMCMC(MCMC):

	def __init__(self):
		# Make the Pyro kernel parameters
		self.kernel = None
		self.mcmc = None
		# Call parent constructor
		super(PyroMCMC, self).__init__()

	def make_kernel(self, target, temp):
		pass

	def make_mcmc(self, n_steps, x_s_t_0, warmup_steps, verbose):
		# Make the MCMC algorithm
		batch_size = x_s_t_0.shape[0]
		initial_params = { 'points' : x_s_t_0 }
		self.mcmc = MCMC_pyro(
	        kernel=self.kernel,
	        num_samples=n_steps,
	        initial_params=initial_params,
	        warmup_steps=warmup_steps,
	        disable_progbar=not verbose,
			disable_validation=True,
			num_chains=batch_size
	    )
	    # Manually force _UnarySampler
		self.mcmc.sampler = _UnarySampler(
			self.kernel,
			n_steps,
			warmup_steps,
			batch_size,
			not verbose,
			initial_params=initial_params,
			hook=None,
		)

	def sample(self, x_s_t_0, n_steps, target, temp=1.0, warmup_steps=0, verbose=False):
		# Check if the kernel exists
		if self.kernel is None:
			self.make_kernel(target, temp)
		# Check if MCMC exists
		if self.mcmc is None:
			self.make_mcmc(n_steps, x_s_t_0, warmup_steps, verbose)
			skip_adaptation_reset = False
		else:
			# Reset the number of steps
			if self.mcmc.warmup_steps != warmup_steps:
				self.mcmc.warmup_steps = warmup_steps
				self.mcmc.sampler.warmup_steps = warmup_steps
			if self.mcmc.num_samples != n_steps:
				self.mcmc.num_samples = n_steps
				self.mcmc.sampler.num_samples = n_steps
			# Reset the initial params
			initial_params = { 'points' : x_s_t_0 }
			self.mcmc._validate_kernel(initial_params)
			self.mcmc.sampler.initial_params = initial_params
			skip_adaptation_reset = True
		# Run the MCMC algorithm
		self.mcmc.run(skip_adaptation_reset=skip_adaptation_reset)
		# Collect the samples
		samples = self.mcmc.get_samples(group_by_chain=True)["points"]
		# Collect the acceptance rate
		batch_size = x_s_t_0.shape[0]
		self.diagnostics['local_acceptance'] = sum(self.mcmc._diagnostics[i]['acceptance rate'] for i in range(batch_size)) / batch_size
		return samples.transpose(0,1)

# No-U-Turn Sampler
class NUTS(PyroMCMC):

	def __init__(self, step_size, adapt_step_size=False, adapt_mass_matrix=True):
		# Make the Pyro kernel parameters
		self.step_size = step_size
		self.adapt_step_size = adapt_step_size
		self.adapt_mass_matrix = adapt_mass_matrix
		# Call parent constructor
		super(NUTS, self).__init__()

	def make_kernel(self, target, temp):
		self.kernel = CustomPyroNUTS(
			potential_fn=lambda x : -target(x['points']).sum() / temp,
			step_size=self.step_size,
			adapt_step_size=self.adapt_step_size,
			adapt_mass_matrix=self.adapt_mass_matrix,
			full_mass=False
		)

# Hamiltonian Monte Carlo Sampler
class HMC(PyroMCMC):

	def __init__(self, trajectory_length, step_size, adapt_step_size=False, adapt_mass_matrix=True):
		# Make the Pyro kernel parameters
		self.trajectory_length = trajectory_length
		self.step_size = step_size
		self.adapt_step_size = adapt_step_size
		self.adapt_mass_matrix = adapt_mass_matrix
		self.kernel = None
		# Call parent constructor
		super(HMC, self).__init__()

	def make_kernel(self, target, temp):
		self.kernel = CustomPyroHMC(
		        potential_fn=lambda x : -target(x['points']).sum() / temp,
		        trajectory_length=self.trajectory_length,
		        step_size=self.step_size,
		        adapt_step_size=self.adapt_step_size,
		        adapt_mass_matrix=self.adapt_mass_matrix,
		        full_mass=False
		)

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
	sampler_nuts = NUTS(
		step_size=1e-2,
		adapt_step_size=True
	)
	samples_nuts = sampler_nuts.sample(
		x_s_t_0=target.sample(sample_shape=(4,)),
		n_steps=400,
		target=target.log_prob,
		warmup_steps=100,
		verbose=True
	).detach().cpu()
	sampler_hmc = HMC(
		step_size=1e-2,
		trajectory_length=3,
		adapt_step_size=True
	)
	samples_hmc = sampler_hmc.sample(
		x_s_t_0=target.sample(sample_shape=(4,)),
		n_steps=400,
		target=target.log_prob,
		warmup_steps=100,
		verbose=True
	).detach().cpu()

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
	plt.scatter(samples_nuts[:,0,0], samples_nuts[:,0,1], alpha=0.75, color='red')
	plt.scatter(samples_hmc[:,0,0], samples_hmc[:,0,1], alpha=0.75, color='blue')
	plt.show()