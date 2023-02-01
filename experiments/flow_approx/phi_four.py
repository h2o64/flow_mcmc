# Sampling for PhiFour

# Libraries
import torch
import pyro.ops.stats as stats
from distribution.phi_four import PhiFour, PhiFourBase
from flow.realnvp import RealNVP
from experiments.utils import set_seed, create_mcmc, algorithms_with_flow
from experiments.utils import compute_random_projections, compute_total_variation
import time
from tqdm import trange
from mcmc.mala import MALA

# Weighted variance
def var_weighted(values, dim, weights):
	if weights is None:
		return torch.var(values, dim=dim)
	else:
		average = torch.sum(weights * values, dim=dim) / weights.sum()
		variance = torch.sum(torch.square(values - average) * weights, dim=dim) / weights.sum()
		return variance

# Main function
def main(config, device, seed, save_samples=False):

	# Browse all dimensions
	for dim_name, dim_info in config['dimensions'].items():

		# Show debug info
		print("============ {} =============".format(dim_name))

		# Create the target
		dim = dim_info['dim']
		target = PhiFour(
		    a=torch.tensor(config['a']),
		    b=torch.tensor(config['b']),
		    dim_grid=torch.tensor(dim)
		)
		beta = config['beta']

		# Make sampler
		proposal = PhiFourBase(
			dim=dim,
			device=device,
			prior_type='coupled',
			alpha=target.a,
			beta=beta
		)

		# Make the RealNVP instance
		flow = RealNVP(
			dim=dim,
			channels=1,
			n_realnvp_blocks=dim_info['n_realnvp_blocks'],
			block_depth=1,
			init_weight_scale=1e-6,
			hidden_dim=dim_info['hidden_dim'],
			hidden_depth=dim_info['hidden_depth'],
			residual=True,
			equivariant=False,
			device=device
		)

		# Load the weights of the flow
		flow.load_state_dict(torch.load(dim_info['weights_filepath'], map_location=device))

		# Origin of the samples
		start_orig = torch.ones((config['batch_size'], dim)).to(device).requires_grad_()
		opt_orig = torch.optim.SGD([start_orig], lr=1e-3)
		orig_r = trange(35000, unit='step')
		for _ in orig_r:
			opt_orig.zero_grad()
			cur_energy = target.U(start_orig).sum()
			cur_energy.backward()
			opt_orig.step()
			orig_r.set_description('energy = {:.2e}'.format(float(cur_energy.item())))
		start_orig = start_orig.detach().clone().to(device)

		# Origin of the samples in both modes
		start_orig_both_modes = torch.ones((20, dim))
		start_orig_both_modes[:10] = -1
		start_orig_both_modes = start_orig_both_modes.to(device).requires_grad_()
		opt_orig_both_modes = torch.optim.SGD([start_orig_both_modes], lr=1e-3)
		orig_r = trange(35000, unit='step')
		for _ in orig_r:
			opt_orig.zero_grad()
			cur_energy = target.U(start_orig_both_modes).sum()
			cur_energy.backward()
			opt_orig_both_modes.step()
			orig_r.set_description('energy = {:.2e}'.format(float(cur_energy.item())))
		start_orig_both_modes = start_orig_both_modes.detach().clone().to(device)

		# Get true samples
		true_sampler_single_mode = MALA(step_size=0.01, target_acceptance=0.75)
		true_samples_single_mode = true_sampler_single_mode.sample(
			x_s_t_0=start_orig.clone()[:20].to(device),
			n_steps=256,
			warmup_steps=256,
			target=target.log_prob,
			temp=1.0/beta
		).detach().view((-1, dim))
		true_sampler_both_mode = MALA(step_size=0.01, target_acceptance=0.75)
		true_samples_both_mode = true_sampler_both_mode.sample(
			x_s_t_0=start_orig_both_modes.clone().to(device),
			n_steps=256,
			warmup_steps=256,
			target=target.log_prob,
			temp=1.0/beta
		).detach().view((-1, dim))

		# Number of random projections for the TV
		n_random_projections = 128
		projs = compute_random_projections(dim, device, n_random_projections=n_random_projections)
		samples2_single_proj, samples2_both_proj = None, None
		samples2_single_kdes, samples2_both_kdes = None, None
		samples2_single_mean, samples2_both_mean = None, None

		# Browse all configurations
		for method_name, info in config['methods'].items():

			# Reset the seed
			set_seed(seed)

			# Show debug info
			print("============ {} =============".format(method_name))

			# Create the MCMC sampler
			if (info['name'] in algorithms_with_flow) or info['neutralize']:
			    flow_ = flow
			else:
			    flow_ = None
			sampler = create_mcmc(info, dim, proposal, flow_)

			# Run the sampler
			s = time.time()
			samples = sampler.sample(
			    x_s_t_0=start_orig.clone().to(device),
			    n_steps=info['n_steps'],
			    target=target.log_prob,
			    warmup_steps=info['warmup_steps'],
			    temp=1.0/beta,
			    verbose=True
			).detach()
			samples_shape = samples.shape
			e = time.time()
			elapsed = e - s
			print(f"Elapsed: {elapsed:.2f} s")

			# Save elapsed time
			torch.save(torch.tensor(elapsed), '{}/elapsed_time_{}_dim_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, seed))

			# Compute and save ESS
			if info['name'] != 'adaptive_is':
				ess = stats.effective_sample_size(samples, chain_dim=0, sample_dim=1).cpu()
				ess_per_sec = ess / elapsed
				torch.save(ess.clone().cpu(), '{}/ess_{}_dim_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, seed))
				torch.save(ess_per_sec.clone().cpu(), '{}/ess_per_sec_{}_dim_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, seed))

			# Compute the energy of the chains
			samples = samples.view((-1, dim))
			if not sampler.has_diagnostics('weights'):
				energy = -target.log_prob(samples).mean()
			else:
				weights = sampler.get_diagnostics('weights').clone()
				energy = -(weights[:,None] * target.log_prob(samples)).sum() / weights.sum()
			torch.save(energy.clone().cpu(), '{}/energy_{}_dim_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, seed))

			# Weight of the mode
			samples = samples.view(samples_shape)
			mask = samples[...,int(dim / 2)] > 0
			if sampler.has_diagnostics('weights'):
				mode_weight = (sampler.get_diagnostics('weights')[:,None] * mask).float().sum(dim=0)
				mode_weight /= sampler.get_diagnostics('weights').sum()
				mode_weight = mode_weight.mean()
			else:
				mode_weight = mask.float().mean(dim=0).mean()
			torch.save(mode_weight.clone().cpu(), '{}/mode_weight_{}_dim_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, seed))

			# Evaluate the variance of each mode
			var = torch.ones((samples.shape[1], ))
			for i in range(samples.shape[1]):
				if sampler.has_diagnostics('weights'):
					weights_a = sampler.get_diagnostics('weights')[mask[:,i]]
					weights_b = sampler.get_diagnostics('weights')[~mask[:,i]]
				else:
					weights_a, weights_b = None, None
				var[i] = torch.sqrt(var_weighted(samples[mask[:,i].flatten(),i,int(dim/2)], dim=0, weights=weights_a)
						 + var_weighted(samples[~mask[:,i].flatten(),i,int(dim/2)], dim=0, weights=weights_b))
			perc_var_nan = torch.sum(torch.isnan(var)) / var.shape[0]
			var = torch.nanmedian(var)
			torch.save(perc_var_nan.clone().cpu(), '{}/perc_var_nan_{}_dim_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, seed))
			torch.save(var.clone().cpu(), '{}/var_{}_dim_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, seed))

			# Estimate the sliced total variation on a single mode
			single_sliced_tvs = torch.zeros((samples_shape[1], n_random_projections))
			for batch_id in range(samples_shape[1]):
			    if samples2_single_proj is None:
			        single_sliced_tvs[batch_id], samples2_single_proj, samples2_single_kdes, samples2_single_mean = compute_total_variation(projs,
			        	samples[:,batch_id], true_samples_single_mode)
			    else:
			        if sampler.has_diagnostics('weights'):
			            weights = sampler.get_diagnostics('weights').detach().cpu().numpy()
			            bw = 'ISJ'
			        else:
			            weights = None
			            bw = 'scott'
			        try:
			            single_sliced_tvs[batch_id] = compute_total_variation(projs, samples[:,batch_id], true_samples_single_mode, weigths_sample1=weights,
			            	samples2_proj=samples2_single_proj, samples2_kdes=samples2_single_kdes, samples2_mean=samples2_single_mean, bw=bw)
			        except:
			            single_sliced_tvs[batch_id] = float('inf')
			torch.save(single_sliced_tvs.clone().cpu(), '{}/single_sliced_tv_{}_dim_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, seed))

			# Estimate the sliced total variation on a both modes
			both_sliced_tvs = torch.zeros((samples_shape[1], n_random_projections))
			for batch_id in range(samples_shape[1]):
			    if samples2_both_proj is None:
			        both_sliced_tvs[batch_id], samples2_both_proj, samples2_both_kdes, samples2_both_mean = compute_total_variation(projs,
			        	samples[:,batch_id], true_samples_both_mode)
			    else:
			        if sampler.has_diagnostics('weights'):
			            weights = sampler.get_diagnostics('weights').detach().cpu().numpy()
			            bw = 'ISJ'
			        else:
			            weights = None
			            bw = 'scott'
			        try:
			            both_sliced_tvs[batch_id] = compute_total_variation(projs, samples[:,batch_id], true_samples_both_mode, weigths_sample1=weights,
			            	samples2_proj=samples2_both_proj, samples2_kdes=samples2_both_kdes, samples2_mean=samples2_both_mean, bw=bw)
			        except:
			            both_sliced_tvs[batch_id] = float('inf')
			torch.save(both_sliced_tvs.clone().cpu(), '{}/both_sliced_tv_{}_dim_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, seed))

			# Save the samples
			if save_samples:
			    torch.save(samples.clone().cpu(), '{}/samples_{}_dim_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, seed))

			# Save the MC ESS
			if sampler.has_diagnostics('ess'):
			    mc_ess = sampler.get_diagnostics('ess')
			    if isinstance(mc_ess, torch.Tensor):
			        mc_ess = mc_ess.mean()
			    else:
			        mc_ess = torch.tensor(mc_ess)
			    if info['neutralize']:
			        N = sampler.inner_sampler.N
			    else:
			        N = sampler.N
			    torch.save(mc_ess.clone().cpu() / N, '{}/mc_ess_{}_dim_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, seed))

			# Print and save the acceptance
			if sampler.has_diagnostics('local_acceptance'):
			    local_acceptance = sampler.get_diagnostics('local_acceptance')
			    if isinstance(local_acceptance, torch.Tensor):
			        local_acceptance = local_acceptance.mean()
			    else:
			        local_acceptance = torch.tensor(local_acceptance)
			    print('local acceptance (mean) = {:.2f}%'.format(100 * float(local_acceptance)))
			    torch.save(local_acceptance.clone().cpu(), '{}/acceptance_local_{}_dim_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, seed))
			if sampler.has_diagnostics('global_acceptance'):
			    global_acceptance = sampler.get_diagnostics('global_acceptance')
			    if isinstance(global_acceptance, torch.Tensor):
			        global_acceptance = global_acceptance.mean()
			    else:
			        global_acceptance = torch.tensor(global_acceptance)
			    print('global acceptance (mean) = {:.2f}%'.format(100 * float(global_acceptance)))
			    torch.save(global_acceptance.clone().cpu(), '{}/acceptance_global_{}_dim_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, seed))

if __name__ == "__main__":
    # Libraries
    import argparse
    import yaml
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument('--seed', type=int)
    parser.add_argument('--save_samples', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    # Freeze the seed
    set_seed(args.seed)
    # Load the config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Get the Pytorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Run the experiment
    main(config, device, args.seed, save_samples=args.save_samples)