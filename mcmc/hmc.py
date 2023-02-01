# Hamiltonian Monte Carlo
# Based on
# - DS-telecom-3 - Markov Chain Monte Carlo - Theory and practical applications (2021-2022)
# - https://colindcarroll.com/2019/04/11/hamiltonian-monte-carlo-from-scratch/
# - https://colindcarroll.com/2019/04/21/step-size-adaptation-in-hamiltonian-monte-carlo/

# Libraries
import torch
import math
from .mcmc import MCMC
from .integrators import leapfrog, leapfrog_twostage, leapfrog_threestage, verlet
from tqdm import trange

# Implementing dual-averaging step size adaptation
class DualAveragingStepSize:
    def __init__(
        self,
        initial_step_size,
        target_accept=0.8,
        gamma=0.05,
        t0=10.0,
        kappa=0.75,
    ):
        """Tune the step size to achieve a desired target acceptance.
        Uses stochastic approximation of Robbins and Monro (1951), described in
        Hoffman and Gelman (2013), section 3.2.1, and using those default values.
        Parameters
        ----------
        initial_step_size: float > 0
            Used to set a reasonable value for the stochastic step to drift towards
        target_accept: float in (0, 1)
            Will try to find a step size that accepts this percent of proposals
        gamma: float
            How quickly the stochastic step size reverts to a value mu
        t0: float > 0
            Larger values stabilize step size exploration early, while perhaps slowing
            convergence
        kappa: float in (0.5, 1]
            The smaller kappa is, the faster we forget earlier step size iterates
        """
        self.mu = math.log(10 * initial_step_size)
        self.target_accept = target_accept
        self.gamma = gamma
        self.t = t0
        self.kappa = kappa
        self.error_sum = 0
        self.log_averaged_step = 0

    def update(self, p_accept):
        """Propose a new step size.
        This method returns both a stochastic step size and a dual-averaged
        step size. While tuning, the HMC algorithm should use the stochastic
        step size and call `update` every loop. After tuning, HMC should use
        the dual-averaged step size for sampling.
        Parameters
        ----------
        p_accept: float
            The probability of the previous HMC proposal being accepted
        Returns
        -------
        float, float
            A stochastic step size, and a dual-averaged step size
        """
        self.error_sum += self.target_accept - p_accept
        log_step = self.mu - self.error_sum / (math.sqrt(self.t) * self.gamma)
        eta = self.t ** -self.kappa
        self.log_averaged_step = (
            eta * log_step + (1 - eta) * self.log_averaged_step
        )
        self.t += 1
        return math.exp(log_step), math.exp(self.log_averaged_step)

# Hamiltonian Monte Carlo
class HMC(MCMC):

	def __init__(self, step_size, trajectory_length, adapt_step_size=True, adapt_mass_matrix=True, early_find_step_size=True, integrator='verlet'):
		# Step size and trajectory length
		self.step_size = step_size
		self.trajectory_length = trajectory_length
		# Adapt the step-size
		self.early_find_step_size = early_find_step_size
		self.adapt_step_size = adapt_step_size
		self.step_size_tuning = None
		# Adapt the mass-matrix
		self.adapt_mass_matrix = adapt_mass_matrix
		self.variances = None
		# Momentum distribution
		self.momemtum_distribution = None
		# Define the integrator
		if integrator == 'verlet':
			self.integrator = verlet
		elif integrator == 1:
			self.integrator = leapfrog
		elif integrator == 2:
			self.integrator = leapfrog_twostage
		elif integrator == 3:
			self.integrator = leapfrog_threestage
		# Call parent constructor
		super(HMC, self).__init__()

	# Find a reasonable step size
	# Stolen from https://github.com/pyro-ppl/pyro/blob/master/pyro/infer/mcmc/hmc.py#L170
	def find_reasonable_step_size(self, z, z_energy, z_grad, potential_grad, direction_threshold=math.log(0.8)):
		# Get the energies
		r = self.momemtum_distribution.sample(z.shape[:-1])
		energy_current = z_energy - self.momemtum_distribution.log_prob(r).sum(dim=-1)
		# Perform leapfrog steps
		z_new, r_new, z_new_energy, z_new_grad = self.integrator(
			q=z.clone(),
			p=r.clone(),
			dVdq=z_grad,
			potential_grad=potential_grad,
			path_len=self.trajectory_length,
			step_size=self.step_size,
			inv_mass_matrix=self.variances if self.adapt_mass_matrix else 1.0
		)
		energy_new = z_new_energy - self.momemtum_distribution.log_prob(r_new).sum(dim=-1)
		# Set a direction
		delta_energy = (energy_new - energy_current).mean()
		direction = 1 if direction_threshold < -delta_energy else -1
		# Define the new step size and direction
		step_size_scale = 2**direction
		direction_new = direction
		# Keep scale step_size until accept_prob crosses its target
		while (direction_new == direction) and (abs(self.step_size) < 1e10):
			# Update the step size
			self.step_size *= step_size_scale
			# Get the energies
			r = self.momemtum_distribution.sample(z.shape[:-1])
			energy_current = z_energy - self.momemtum_distribution.log_prob(r).sum(dim=-1)
			# Perform leapfrog steps
			z_new, r_new, z_new_energy, z_new_grad = self.integrator(
				q=z.clone(),
				p=r.clone(),
				dVdq=z_grad,
				potential_grad=potential_grad,
				path_len=self.trajectory_length,
				step_size=self.step_size,
				inv_mass_matrix=self.variances if self.adapt_mass_matrix else 1.0
			)
			energy_new = z_new_energy - self.momemtum_distribution.log_prob(r_new).sum(dim=-1)
			# Set a direction
			delta_energy = (energy_new - energy_current).mean()
			direction = 1 if direction_threshold < -delta_energy else -1

	# Compute variances according to a Welford's scheme
	def update_variances(self, x, update_var=True):
		with torch.no_grad():
		# Create the variances if needed
			# Increment
			self.acc_count += 1
			# Update the mean
			delta = x - self.means
			self.means += delta / self.acc_count
			# Update the accumulator
			delta2 = x - self.means
			self.var_accumulator += delta * delta2
			# Update the variances
			if self.acc_count > 1 and update_var:
				self.variances = self.var_accumulator / (self.acc_count - 1)
				self.scales = torch.sqrt(self.variances)
				self.momemtum_distribution = torch.distributions.Normal(
					loc=self.momemtum_distribution.loc,
					scale=self.scales,
					validate_args=False
				)

	def one_step(self, x_s_t, potential_x_s_t, grad_x_s_t, potential, potential_grad, temp, warmup_step_id=None, warmup_steps_max=None):
		# Integrate over the current path
		p0 = self.momemtum_distribution.sample(x_s_t.shape[:-1])
		q_new, p_new, final_V, final_dVdq = self.integrator(
			q=x_s_t.clone(),
			p=p0.clone(),
			dVdq=grad_x_s_t,
			potential_grad=potential_grad,
			path_len=2.0 * torch.rand(1) * self.trajectory_length,
			step_size=self.step_size,
			inv_mass_matrix=self.variances if self.adapt_mass_matrix else 1.0
		)
		# Compute the MH ratio
		with torch.no_grad():
			joint_prop = self.momemtum_distribution.log_prob(p0).sum(dim=-1) - potential_x_s_t
			joint_orig = self.momemtum_distribution.log_prob(p_new).sum(dim=-1) - final_V
			energy_change = joint_orig - joint_prop
			# Acceptance step
			mask = torch.log(torch.rand_like(joint_prop, device=x_s_t.device)) < energy_change
			x_s_t.data[mask] = q_new[mask]
			potential_x_s_t.data[mask] = final_V[mask]
			grad_x_s_t.data[mask] = grad_x_s_t[mask]
		# Compute mean acceptance propobability
		if x_s_t.shape[0] > 1:
			mean_acc = torch.mean(mask.float()).cpu()
		else:
			mean_acc = torch.minimum(
				torch.exp(energy_change).cpu(),
				torch.ones_like(energy_change).cpu()
			)
			mean_acc[energy_change.isnan()] = 0.0
		# Adapt the step size
		if self.adapt_step_size and warmup_step_id is not None:
			if warmup_step_id < warmup_steps_max - 1:
				self.step_size, _ = self.step_size_tuning.update(mean_acc)
			elif warmup_step_id == warmup_steps_max - 1:
				_, self.step_size = self.step_size_tuning.update(mean_acc)
		# Adapt the mass matrix
		if self.adapt_mass_matrix and warmup_step_id is not None:
			for i in range(x_s_t.shape[0]):
				self.update_variances(x_s_t[i], update_var=(i == (x_s_t.shape[0] - 1)) and (i > warmup_steps_max/2))
		# Return the data
		return x_s_t, potential_x_s_t, grad_x_s_t, mean_acc

	def sample(self, x_s_t_0, n_steps, target, temp=1.0, warmup_steps=0, verbose=False):
		x_s_t = torch.autograd.Variable(x_s_t_0.clone(), requires_grad=True)
		x_s_ts = torch.zeros((n_steps, *x_s_t.shape), device=x_s_t.device)
		local_accepts = torch.zeros((n_steps,)).cpu()
		r = trange(n_steps) if verbose else range(n_steps)
		# Define the potential function
		def potential(x): return -target(x) / temp
		def potential_grad(x):
			en = -target(x) / temp
			grad = torch.autograd.grad(en.sum(), x)[0]
			return en, grad
		# Initial potential and grad
		potential_x_s_t, grad_x_s_t = potential_grad(x_s_t)
		potential_x_s_t, grad_x_s_t = potential_x_s_t.clone(), grad_x_s_t.clone()
		# Initialize the variance
		if self.variances is None:
			self.variances = torch.ones((x_s_t.shape[-1], ), device=x_s_t.device)
			self.scales = torch.sqrt(self.variances)
			self.means = torch.zeros((x_s_t.shape[-1],), device=x_s_t.device)
			self.var_accumulator = torch.zeros((x_s_t.shape[-1],), device=x_s_t.device)
			self.acc_count = 0
		# Initialize the momentum distribution
		if self.momemtum_distribution is None:
			self.momemtum_distribution = torch.distributions.Normal(
				loc=torch.zeros((x_s_t.shape[-1],), device=x_s_t.device),
				scale=self.scales,
				validate_args=False
			)
		# Find a reasonable step size
		if self.early_find_step_size and warmup_steps > 0:
			self.find_reasonable_step_size(x_s_t, potential_x_s_t, grad_x_s_t, potential_grad)
		# Find the step size
		if self.adapt_step_size and self.step_size_tuning is None:
			self.step_size_tuning = DualAveragingStepSize(initial_step_size=self.step_size)
		# Warmup steps
		if warmup_steps > 0:
			for warmup_step_id in range(warmup_steps):
				x_s_t, potential_x_s_t, grad_x_s_t, _ = self.one_step(
					x_s_t=x_s_t,
					potential_x_s_t=potential_x_s_t,
					grad_x_s_t=grad_x_s_t,
					potential=potential,
					potential_grad=potential_grad,
					temp=temp,
					warmup_step_id=warmup_step_id,
					warmup_steps_max=warmup_steps
				)
		# Sampling steps
		for ell in r:
			x_s_ts[ell], potential_x_s_t, grad_x_s_t, local_accepts[ell] = self.one_step(
					x_s_t=x_s_t,
					potential_x_s_t=potential_x_s_t,
					grad_x_s_t=grad_x_s_t,
					potential=potential,
					potential_grad=potential_grad,
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
		covariance_matrix=torch.diag(torch.logspace(-2,2,2)).to(device)
	)

	# Create the sampler
	sampler = HMC(step_size=1e-1, trajectory_length=10, adapt_mass_matrix=False)
	samples = sampler.sample(
		x_s_t_0=target.sample(sample_shape=(32,)),
		warmup_steps=100,
		n_steps=700,
		target=target.log_prob,
		verbose=True
	).detach().cpu()
	print('Acceptance = ', sampler.get_diagnostics('local_acceptance').mean())

	# Compute the KL divergence
	samples_shape = samples.shape
	samples = samples.view((-1,2))
	samples_mean = samples.mean(dim=0)
	samples_cov = samples.T.cov()
	samples = samples.view(samples_shape)
	samples_dist = torch.distributions.MultivariateNormal(
		loc=samples_mean.to(device),
		covariance_matrix=samples_cov.to(device)
	)
	print('KL = ', torch.distributions.kl.kl_divergence(target, samples_dist))

	# Make a grid
	x = np.linspace(-0.5, 0.5, 256)
	y = np.linspace(-50, 50, 256)
	X, Y = np.meshgrid(x, y)
	Z = torch.Tensor(torch.stack([torch.Tensor(X),torch.Tensor(Y)], dim=2)).to(device)

	# Compute the grid of the target
	log_prob = target.log_prob(Z.view((-1,2))).view((256,256)).detach().cpu()

	# Display everything
	plt.figure(figsize=(10,10))
	plt.contour(X, Y, torch.exp(log_prob), 20, linestyles=':', colors='k')
	plt.scatter(samples[:,0,0], samples[:,0,1], alpha=0.75, color='red')
	plt.show()
