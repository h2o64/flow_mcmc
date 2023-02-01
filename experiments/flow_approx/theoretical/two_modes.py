# Libraries
import numpy as np
import torch
import math
from experiments.utils import create_mcmc

# KDE
# from sklearn.neighbors import KernelDensity
# from sklearn.model_selection import GridSearchCV
from KDEpy import FFTKDE, TreeKDE

# Make pi
def make_pi(sigma, a):
    locs = torch.ones((2,)).to(a.device) * a
    locs[0] *= -1
    return torch.distributions.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(torch.ones(2,).to(sigma.device)),
        component_distribution=torch.distributions.Normal(
            loc=locs,
            scale=torch.ones((2,)).to(sigma.device) * sigma
        )
    )

# Make Q
def make_q(sigma_q):
    return torch.distributions.Normal(
        loc=torch.zeros((1,)).to(sigma_q.device),
        scale=torch.ones((1,)).to(sigma_q.device) * sigma_q
    )

# Compute L from sigma
def compute_L(sigma, a):
    return (1.0/(2*math.sqrt(2*math.pi)*sigma)) * (1 + torch.exp(-2*torch.square(a)/torch.square(sigma)) - 2*torch.exp(-torch.square(a)/(2*torch.square(sigma))))

# Compute the TV
def compute_tv(x_samples, target, n_samples=2048):
	# Convert to numpy
	x_samples_np = x_samples.detach().cpu().numpy()
	# Sample from the target
	pi_samples = target.sample(x_samples_np.shape[:-1])
	# Get the min and max
	x_min = float(min(pi_samples.min(), x_samples_np.min()))
	x_max = float(max(pi_samples.max(), x_samples_np.max()))
	# Make a grid
	points = np.linspace(x_min, x_max, n_samples)
	# Perform a KDE on x_samples
	samples_prob = TreeKDE(bw="ISJ").fit(x_samples_np).evaluate(points)
	# Get the likelihood over the target
	true_prob = torch.exp(target.log_prob(torch.from_numpy(points).to(x_samples.device))).detach().cpu().numpy()
	# Compute the TV
	return 0.5 * np.abs(samples_prob - true_prob).mean() * (x_max - x_min)

# Compute the TV with a Monte-Carlo estimator
def compute_tv_mc(x_samples, target, n_samples=8192):
	# Convert to numpy
	x_samples_np = x_samples.detach().cpu().numpy()
	# Perform a KDE on x_samples
	samples, samples_prob = FFTKDE(bw="ISJ").fit(x_samples_np).evaluate(n_samples)
	samples_prob = np.log(samples_prob)
	# Evaluate pi
	pi_prob = target.log_prob(torch.from_numpy(samples).to(x_samples.device)).detach().cpu().numpy()
	# Compute the TV
	return 0.5 * np.average(np.abs(1 - np.exp(pi_prob - samples_prob)), weights=np.exp(samples_prob))

# Find the mixing time using dichotomia
def find_mixing_time(x, target, target_eps, n_start=32, n_eval=64):
	# Initialisation
	n_max = x.shape[0]
	cur_max = n_max
	cur_min = n_start
	cur_middle = int((cur_max + cur_min) / 2)
	for _ in range(n_eval):
		if cur_min >= cur_max: break
		cur = compute_tv_mc(x[:cur_middle].reshape((-1,1)), target)
		if cur < target_eps:
			cur_max = cur_middle-1
		else:
			cur_min = cur_middle+1
		cur_middle = int((cur_max + cur_min) / 2)
	return cur_middle

# Main function
def main(savepath, device, seed, sigma_size, n_mcmc_steps, warmup_steps, batch_size, mini_batch_size, target_eps):

	# Range for sigma
	sigma_range = torch.logspace(-2, 0, sigma_size).to(device)

	# Collect area
	a = torch.tensor(1).to(device)
	L_range = compute_L(sigma_range, a).detach().cpu()
	eps_imh = torch.ones((sigma_size, 2))
	n_imh = torch.ones((sigma_size, 2))
	acc_imh = torch.ones((sigma_size, 2))
	eps_mala = torch.ones((sigma_size, 2))
	n_mala = torch.ones((sigma_size, 2))
	acc_mala = torch.ones((sigma_size, 2))

	# Get the precision
	for i in range(sigma_size):

		# Show debug info
		print("============ sigma = {:.2e} =============".format(sigma_range[i]))

		# Make the target
		target = make_pi(sigma_range[i], torch.tensor(1).to(device))

		# Make the constant proposal
		proposal = make_q(sigma_q=target.stddev)

		# Build the initial MCMC state
		x_orig = target.component_distribution.sample(sample_shape=(batch_size,))[:,0]
		x_orig = x_orig.view((-1, 1))

		# IMH
		## Make the sampler
		sampler = create_mcmc(
			conf_={
				'name': 'imh',
				'neutralize': False,
				'params' : {}
			},
			dim=1,
			proposal=proposal,
			flow=None
		)
		## Collect the samples
		samples = sampler.sample(
		    x_s_t_0=x_orig.clone(),
		    n_steps=n_mcmc_steps,
		    target=target.log_prob,
		    warmup_steps=warmup_steps,
		    verbose=True
		).detach()
		## Compute the TV
		eps = np.array([compute_tv_mc(samples[:,i], target) for i in range(batch_size)])
		eps_imh[i,0] = float(np.mean(eps))
		eps_imh[i,1] = float(np.std(eps))
		print('eps_imh = {} +/- {}'.format(*eps_imh[i]))
		## Reshape the samples
		samples = samples.view((samples.shape[0], -1, mini_batch_size, 1))
		## Compute the mixing time
		ns = np.array([find_mixing_time(samples[:,i], target, target_eps=target_eps) for i in range(samples.shape[1])])
		n_imh[i,0] = float(np.mean(ns))
		n_imh[i,1] = float(np.std(ns))
		print('n_imh = {} +/- {}'.format(*n_imh[i]))
		## Get the acceptance
		acc = sampler.get_diagnostics('global_acceptance')
		acc_imh[i,0] = acc.mean()
		acc_imh[i,1] = acc.std()
		print('acc_imh = {} +/- {}'.format(*acc_imh[i]))

		# MALA
		## Make the sampler
		sampler = create_mcmc(
			conf_={
				'name': 'mala',
				'neutralize': False,
				'params': {
					'step_size': 0.01,
					'target_acceptance': 0.574
				}
			},
			dim=1,
			proposal=proposal,
			flow=None
		)
		## Collect the samples
		samples = sampler.sample(
		    x_s_t_0=x_orig.clone(),
		    n_steps=n_mcmc_steps,
		    target=target.log_prob,
		    warmup_steps=warmup_steps,
		    verbose=True
		).detach()
		## Compute the TV
		eps = np.array([compute_tv_mc(samples[:,i], target) for i in range(batch_size)])
		eps_mala[i,0] = float(np.mean(eps))
		eps_mala[i,1] = float(np.std(eps))
		print('eps_mala = {} +/- {}'.format(*eps_mala[i]))
		## Compute the mixing time
		ns = np.array([find_mixing_time(samples[:,i], target, target_eps=target_eps) for i in range(batch_size)])
		n_mala[i,0] = float(np.mean(ns))
		n_mala[i,1] = float(np.std(ns))
		print('n_mala = {} +/- {}'.format(*n_mala[i]))
		## Get the acceptance
		acc = sampler.get_diagnostics('local_acceptance')
		acc_mala[i,0] = acc.mean()
		acc_mala[i,1] = acc.std()
		print('acc_mala = {} +/- {}'.format(*acc_mala[i]))

	# Save everything
	torch.save(L_range, '{}/L_range.pt'.format(savepath))
	torch.save(eps_imh, '{}/eps_imh.pt'.format(savepath))
	torch.save(eps_mala, '{}/eps_mala.pt'.format(savepath))
	torch.save(acc_imh, '{}/acc_imh.pt'.format(savepath))
	torch.save(acc_mala, '{}/acc_mala.pt'.format(savepath))
	torch.save(n_imh, '{}/n_imh.pt'.format(savepath))
	torch.save(n_mala, '{}/n_mala.pt'.format(savepath))

if __name__ == "__main__":
	# Libraries
	import argparse
	# Parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--savepath', type=str)
	parser.add_argument('--seed', type=int)
	parser.add_argument('--sigma_size', type=int, default=64)
	parser.add_argument('--n_mcmc_steps', type=int, default=8192)
	parser.add_argument('--warmup_steps', type=int, default=16)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--mini_batch_size', type=int, default=32)
	parser.add_argument('--target_eps', type=float, default=8e-2)
	args = parser.parse_args()
	# Get the Pytorch device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# Get batch sizes
	if not args.batch_size % args.mini_batch_size == 0:
		print('mini_batch_size ({}) must divide batch_size {}'.format(args.mini_batch_size, args.batch_size))
		exit(1)
	# Run the experiment
	main(
		savepath=args.savepath,
		device=device,
		seed=args.seed,
		sigma_size=args.sigma_size,
		n_mcmc_steps=args.n_mcmc_steps,
		warmup_steps=args.warmup_steps,
		batch_size=args.batch_size,
		mini_batch_size=args.mini_batch_size,
		target_eps=args.target_eps
	)