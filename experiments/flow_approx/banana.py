# Banana
# Details :
# - Banana with RealNVP flows

# Libraries
import torch
import time
import math
from experiments.utils import set_seed
from experiments.utils import compute_random_projections, compute_total_variation
from models.banana import Banana
from flow.realnvp import RealNVP
import pickle

# MCMC
from mcmc.ess import ESS
from mcmc.mala import MALA
from mcmc.isir import iSIR
from mcmc.classic_is import IS
from mcmc.global_local import Ex2MCMC
from mcmc.neutra import NeuTra
from mcmc.learn_flow import LearnMCMC

# Get hyperparameters for the flow
def get_hyperparams_flow(dim):
	# Adapt the number of steps to the problem
	min_dim, min_it = 16, 1000
	max_dim, max_it = 512, 7000
	a = (max_it - min_it) / (max_dim - min_dim)
	b = min_it - min_dim * a
	n_steps = int(a * dim + b)
	n_steps = int(n_steps)
	print('n_steps = ', n_steps)
	# Adapt the patience to the problem
	min_dim, min_patience = 16, 75
	max_dim, max_patience = 512, 300
	a = (max_patience - min_patience) / (max_dim - min_dim)
	b = min_patience - min_dim * a
	patience = int(a * dim + b)
	print('patience = ', patience)
	# Adapt the learning rate to the problem
	min_dim, min_log_lr = 16, math.log(1e-2)
	max_dim, max_log_lr = 512, math.log(1e-5)
	a = (max_log_lr - min_log_lr) / (max_dim - min_dim)
	b = min_log_lr - min_dim * a
	lr = math.exp(a * dim + b)
	print('lr = ', lr)
	# Adapt the hidden dimension to the problem
	min_dim, min_hidden_dim = 16, 64
	max_dim, max_hidden_dim = 512, 400
	a = (max_hidden_dim - min_hidden_dim) / (max_dim - min_dim)
	b = min_hidden_dim - min_dim * a
	hidden_dim = int(a * dim + b)
	print('hidden_dim = ', hidden_dim)
	# Adapt the hidden depth to the problem
	min_dim, min_hidden_depth = 16, 3
	max_dim, max_hidden_depth = 512, 3
	a = (max_hidden_depth - min_hidden_depth) / (max_dim - min_dim)
	b = min_hidden_depth - min_dim * a
	hidden_depth = int(a * dim + b)
	print('hidden_depth = ', hidden_depth)
	# Adapt the number of realnvp blocks to the problem
	min_dim, min_n_realnvp_blocks = 16, 3
	max_dim, max_n_realnvp_blocks = 512, 12
	a = (max_n_realnvp_blocks - min_n_realnvp_blocks) / (max_dim - min_dim)
	b = min_n_realnvp_blocks - min_dim * a
	n_realnvp_blocks = int(a * dim + b)
	print('n_realnvp_blocks = ', n_realnvp_blocks)
	return n_steps, patience, lr, hidden_dim, hidden_depth, n_realnvp_blocks

# Get the hyperparameters for each sampler
def get_hyperparams_sampler(dim):
	# i-SIR
	if dim == 16: N = 60
	elif dim == 32: N = 60
	elif dim == 64: N = 60
	elif dim == 128: N = 60
	elif dim == 256: N = 60
	elif dim == 512: N = 60
	else: return None
	isir_params = { 'N' : N }
	# MALA
	mala_params = {
		'step_size' : 1e-2,
		'target_acceptance' : 0.75
	}
	# FlEx2MCMC
	if dim == 16:
		flex2_params = {
			'params' : {
				'N' : N,
				'n_local_mcmc' : 10
			},
			'local_sampler_params' : {
				'step_size' : 0.01,
				'target_acceptance' : 0.75
			}
		}
	elif dim == 32:
		flex2_params = {
			'params' : {
				'N' : N,
				'n_local_mcmc' : 10
			},
			'local_sampler_params' : {
				'step_size' : 0.01,
				'target_acceptance' : 0.75
			}
		}
	elif dim == 64:
		flex2_params = {
			'params' : {
				'N' : N,
				'n_local_mcmc' : 10
			},
			'local_sampler_params' : {
				'step_size' : 0.01,
				'target_acceptance' : 0.75
			}
		}
	elif dim == 128:
		flex2_params = {
			'params' : {
				'N' : N,
				'n_local_mcmc' : 10
			},
			'local_sampler_params' : {
				'step_size' : 0.01,
				'target_acceptance' : 0.75
			}
		}
	elif dim == 256:
		flex2_params = {
			'params' : {
				'N' : N,
				'n_local_mcmc' : 10
			},
			'local_sampler_params' : {
				'step_size' : 0.01,
				'target_acceptance' : 0.75
			}
		}
	elif dim == 512:
		flex2_params = {
			'params' : {
				'N' : N,
				'n_local_mcmc' : 10
			},
			'local_sampler_params' : {
				'step_size' : 0.01,
				'target_acceptance' : 0.75
			}
		}
	return isir_params, mala_params, flex2_params

# Main function
def main(savefile, loss_type, dim, n_steps, batch_size, batch_size_mcmc, device, seed, n_random_projections=256, n_bins=1024):

	# Set the seed
	set_seed(seed)

	# Make the proposal
	proposal = torch.distributions.MultivariateNormal(
		loc=torch.zeros((dim,), device=device),
		covariance_matrix=torch.eye(dim, device=device)
	)

	# Make the target
	target = Banana(dim=dim, device=device)

	# Sample the target
	samples_target = target.sample(sample_shape=(10 *n_steps,))

	# Number of random projections for the TV
	n_random_projections = 128
	projs = compute_random_projections(dim, device, n_random_projections=n_random_projections)
	_, samples2_proj, samples2_kdes, samples2_mean = compute_total_variation(projs, samples_target, samples_target)

	# Make the flow
	n_train_steps, patience, lr, hidden_dim, hidden_depth, n_realnvp_blocks = get_hyperparams_flow(dim)
	flow = RealNVP(
		dim=dim,
		channels=1,
		n_realnvp_blocks=n_realnvp_blocks,
		block_depth=1,
		init_weight_scale=1e-6,
		hidden_dim=hidden_dim,
		hidden_depth=hidden_depth,
		device=device
	)

	# Setup the optimizer
	optimizer = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=1e-4)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(
		optimizer=optimizer,
		step_size=patience,
		gamma=0.98
	)

	# Get all the parameters
	isir_params, mala_params, flex2_params = get_hyperparams_sampler(dim)

	# Make i-SIR
	sampler_isir = iSIR(
		**isir_params,
		proposal=proposal,
		flow=flow
	)

	# Make IS
	sampler_is = IS(
		N=int(isir_params['N'] * n_steps / 4),
		proposal=proposal,
		flow=flow
	)

	# Make FlEx2MCMC
	sampler_flex2 = Ex2MCMC(
		N=flex2_params['params']['N'],
		local_mcmc=MALA(**flex2_params['local_sampler_params']),
		n_local_mcmc=flex2_params['params']['n_local_mcmc'],
		proposal=proposal,
		flow=flow
	)

	# Make NeuTra MALA
	sampler_neutra = NeuTra(
		inner_sampler=MALA(**mala_params),
		flow=flow
	)

	# Make TESS
	sampler_tess = NeuTra(
		inner_sampler=ESS(),
		flow=flow
	)

	# Save all metrics
	metrics_all = []

	# Callback while learning the flow
	def callback(train_obj, step_id, x, z, log_jac, loss):

		# Doesn't perform the callback every time
		if not (step_id % int(n_train_steps / 20) == 0): return

		# Get the current training time
		cur_time = time.time()

		# Sample with everything
		metrics = {}
		start_orig, _ = flow.forward(proposal.sample((batch_size_mcmc,)))
		start_orig = start_orig.detach().clone().to(device)
		for sampler, sampler_name in [(sampler_isir, 'isir'),(sampler_flex2,'flex2'),(sampler_neutra,'neutra'),(sampler_tess, 'tess'),(sampler_is,'is')]:
			# Get samples
			samples = sampler.sample(
				x_s_t_0=start_orig.clone(),
				n_steps=n_steps,
				target=target.log_prob,
				verbose=False
			).detach()
			# Compute the slices TV
			ret = torch.zeros((samples.shape[1], n_random_projections))
			for batch_id in range(samples.shape[1]):
				try:
					ret[batch_id] = compute_total_variation(projs, samples[:,batch_id], samples_target, samples2_proj=samples2_proj,
	                                samples2_kdes=samples2_kdes, samples2_mean=samples2_mean)
				except:
					ret[batch_id] = float('inf')
			metrics[sampler_name] = (float(ret.mean()), float(ret.std()))
		# Save everything
		metrics_all.append((metrics, cur_time - start_train_time))

		# Save everything
		with open(savefile, 'wb') as f:
			pickle.dump(metrics_all, f)

	# Make the learner
	start_train_time = time.time()
	learner = LearnMCMC(
		n_mcmc_steps=int(batch_size/batch_size_mcmc),
		n_train_steps=n_train_steps,
		opt=optimizer,
		opt_scheduler=lr_scheduler,
		loss_type=loss_type,
		inner_sampler=Ex2MCMC(
			N=flex2_params['params']['N'],
			local_mcmc=MALA(**flex2_params['local_sampler_params']),
			n_local_mcmc=flex2_params['params']['n_local_mcmc'],
			proposal=proposal,
			flow=flow
		),
		flow=flow
	)

	# Launch the training
	learner.train_flow(
		x_s_t_0=proposal.sample((batch_size_mcmc,)).clone().to(device),
		target=target.log_prob,
		base=proposal,
		verbose=True,
		callback=callback
	)

	# Save everything
	with open(savefile, 'wb') as f:
		pickle.dump(metrics_all, f)


if __name__ == "__main__":
    # Libraries
    import argparse
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("savefile")
    parser.add_argument('--loss_type', type=str)
    parser.add_argument('--dim', type=int)
    parser.add_argument('--n_steps', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--batch_size_mcmc', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    # Freeze the seed
    set_seed(args.seed)
    # Get the Pytorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Run the experiment
    main(args.savefile, args.loss_type, args.dim, args.n_steps, args.batch_size, args.batch_size_mcmc,
    	device, args.seed)