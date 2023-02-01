# Learn flow while sampling

# Libraries
from tqdm import trange
from .mcmc import MCMC

# Learn using the sampler
class LearnMCMC(MCMC):

	def __init__(self, n_mcmc_steps, n_train_steps, opt, loss_type, inner_sampler, flow, opt_scheduler=None, alpha=0.8, use_weight_reg=True):
		# Training params
		self.n_mcmc_steps = n_mcmc_steps
		self.n_train_steps = n_train_steps
		self.opt = opt
		self.opt_scheduler = opt_scheduler
		self.loss_type = loss_type
		self.alpha = alpha
		self.use_weight_reg = use_weight_reg
		# Inner sampler
		self.inner_sampler = inner_sampler
		# Flow
		self.flow = flow
		# Call the parent constructor
		super(LearnMCMC, self).__init__()

	def train_flow(self, x_s_t_0, target, base, temp=1.0, warmup_steps=0, callback=None, verbose=False):
		# Set the range
		if verbose:
			r = trange(self.n_train_steps, unit='step')
		else:
			r = range(self.n_train_steps)
		# Do the training
		losses = []
		x_init = x_s_t_0.detach().clone().to(x_s_t_0.device)
		for step_id in r:
			# Reset the optimizer
			self.opt.zero_grad()
			# Compute the loss
			if self.loss_type == 'forward_kl' or self.loss_type == 'convex_kl':
				# Sample the target
				if step_id == 0:
					warmup_steps_ = warmup_steps
				else:
					warmup_steps_ = 0
				x = self.inner_sampler.sample(
					x_s_t_0=x_init,
					n_steps=self.n_mcmc_steps,
					target=target,
					temp=temp,
					warmup_steps=warmup_steps_
				)
				x_init = x[-1].detach().clone()
				x = x.view((-1, x.shape[-1]))
				# Compute the bare minimum for gradient
				z, log_jac = self.flow.inverse(x)
				loss = -(base.log_prob(z) + log_jac).mean()
				if self.loss_type == 'convex_kl':
					loss_f = loss
			if self.loss_type == 'backward_kl' or self.loss_type == 'convex_kl':
				# Sample the base
				z = base.sample((x_init.shape[0] * self.n_mcmc_steps,))
				# Compute the bare minimum for the gradient
				x, log_jac = self.flow.forward(z)
				loss = -(target(x) + log_jac).mean()
				if self.loss_type == 'convex_kl':
					loss_b = loss
			if self.loss_type == 'convex_kl':
				loss = self.alpha * loss_f + (1 - self.alpha) * loss_b
			# Add a regularization
			if self.use_weight_reg:
				loss += 5e-5 * self.flow.get_weight_scale()
			# Optimizer step
			loss.backward()
			self.opt.step()
			if self.opt_scheduler is not None:
				self.opt_scheduler.step()
			# Callback
			if callback is not None and self.loss_type != 'convex_kl':
				callback(self, step_id, x, z, log_jac, loss)
			# Verbose
			losses.append(loss.item())
			if verbose:
				r.set_description('loss = {}'.format(round(loss.item(), 4)), refresh=True)
		# Save the loss
		self.diagnostics['losses'] = losses

	def sample(self, x_s_t_0, n_steps, target, temp=1.0, warmup_steps=0, verbose=False):
		samples = self.inner_sampler.sample(
			x_s_t_0=x_s_t_0,
			n_steps=n_steps,
			target=target,
			temp=temp,
			warmup_steps=warmup_steps,
			verbose=verbose
		)
		self.diagnostics = self.diagnostics | self.inner_sampler.diagnostics
		return samples

# Debug
if __name__ == "__main__":

	# Import PhiFour
	import torch
	from distribution.phi_four import PhiFour, PhiFourBase

	# Get the Pytorch device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Make PhiFour
	dim = 128
	dist = PhiFour(
	    a=torch.tensor(0.1),
	    b=torch.tensor(0.0),
	    dim_grid=torch.tensor(dim)
	)
	beta = 20

	# Make the flow
	from flow.realnvp import RealNVP
	flow = RealNVP(
        dim=dim,
        channels=1,
        n_realnvp_blocks=6,
        block_depth=1,
        init_weight_scale=1e-6,
        hidden_dim=128,
        hidden_depth=2,
        residual=True,
        equivariant=True,
        device=device
    )

	# Make FlEx2MCMC
	from global_local import Ex2MCMC
	from mala import MALA

	# Make sampler
	proposal = PhiFourBase(
		dim=dim,
		device=device,
		prior_type='coupled',
		alpha=dist.a,
		beta=beta
	)
	inner_sampler = Ex2MCMC(
		N=64,
		proposal=proposal,
		local_mcmc=MALA(step_size=5e-5, target_acceptance=0.75),
		n_local_mcmc=10,
		flow=flow
	)

	# Train the flow
	opt = torch.optim.Adam(flow.parameters(), lr=1e-4)
	opt_scheduler = torch.optim.lr_scheduler.StepLR(
		optimizer=opt,
		step_size=100,
		gamma=0.99
	)
	batch_size = 4096
	batch_size_mcmc = 64
	sampler = LearnMCMC(
		n_mcmc_steps=int(batch_size/batch_size_mcmc),
		n_train_steps=5000,
		opt=opt,
		opt_scheduler=opt_scheduler,
		loss_type='convex_kl',
		inner_sampler=inner_sampler,
		flow=flow
	)

	# Launch training
	x_init = torch.ones((batch_size_mcmc, dim)).to(device)
	x_init[int(0.5*batch_size_mcmc):,:] = -1
	sampler.train_flow(
		x_s_t_0=x_init,
		target=dist.log_prob,
		base=proposal,
		temp=1.0/beta,
		verbose=True
	)

	# Make the init point
	x_init = flow.forward(proposal.sample((batch_size_mcmc,)))[0]

	# Ex2MCMC
	samples = sampler.sample(
		x_s_t_0=x_init,
		warmup_steps=250,
		n_steps=500,
		target=dist.log_prob,
		temp=1.0/beta,
		verbose=True
	)
	samples = samples.view((-1, dim)).detach().cpu().numpy()

	print('acceptance_globale = ', sampler.get_diagnostics('global_acceptance').mean())
	print('acceptance_locale = ', sampler.get_diagnostics('local_acceptance').mean())

	# Display the results
	import matplotlib.pyplot as plt
	for i in range(batch_size_mcmc):
		plt.plot(range(dim), samples[i], alpha=0.1, color='green', linewidth=3)
	plt.show()
