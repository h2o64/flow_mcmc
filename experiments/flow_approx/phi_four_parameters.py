# Hyper-parameters for PhiFour

# Libraries
import torch
from experiments.utils import set_seed, create_mcmc
from distribution.phi_four import PhiFour, PhiFourBase
from flow.realnvp import RealNVP
from mcmc.learn_flow import LearnMCMC
import pickle
from tqdm import trange
from mcmc.utils import compute_imh_is_weights

# Main function
def main(config, device, seed, save_flow=False):

	# Create the target
	dim = config['dim']
	dist = PhiFour(
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
		alpha=dist.a,
		beta=beta
	)

	# Starting point for training
	x_init_training = torch.ones((config['batch_size_mcmc'], dim)).to(device)
	x_init_training[int(0.5*config['batch_size_mcmc']):,:] = -1
	x_init_training = x_init_training.requires_grad_()
	opt_orig = torch.optim.SGD([x_init_training], lr=1e-3)
	orig_r = trange(35000, unit='step')
	for _ in orig_r:
		opt_orig.zero_grad()
		cur_energy = dist.U(x_init_training).sum()
		cur_energy.backward()
		opt_orig.step()
		orig_r.set_description('energy = {:.2e}'.format(float(cur_energy.item())))
	x_init_training = x_init_training.detach().clone().to(device)

	# Starting point for sampling
	x_init_sampling = torch.ones((config['batch_size_mcmc'], dim)).to(device)
	x_init_sampling = x_init_sampling.requires_grad_()
	opt_orig = torch.optim.SGD([x_init_sampling], lr=1e-3)
	orig_r = trange(35000, unit='step')
	for _ in orig_r:
		opt_orig.zero_grad()
		cur_energy = dist.U(x_init_sampling).sum()
		cur_energy.backward()
		opt_orig.step()
		orig_r.set_description('energy = {:.2e}'.format(float(cur_energy.item())))
	x_init_sampling = x_init_sampling.detach().clone().to(device)

	# Record of all losses
	all_losses = {}
	all_infos = {}

	# Browse all configurations
	for method_name, info in config['methods'].items():

		# Debug
		print("============ {} =============".format(method_name))

		# Set the seed
		set_seed(seed)

		# Make the flow
		flow = RealNVP(
	        dim=dim,
	        channels=1,
	        n_realnvp_blocks=config['n_realnvp_blocks'],
	        block_depth=1,
	        init_weight_scale=1e-6,
	        hidden_dim=config['hidden_dim'],
	        hidden_depth=config['hidden_depth'],
	        residual=True,
	        equivariant=False,
	        device=device
	    )

		# Make the sampler
		if config['mala_sampler']:
			info_new = {
				'name' : 'mala',
				'neutralize' : False,
				'params' : {
					'step_size' : 0.001,
					'target_acceptance' : 0.75
				}
			}
		else:
			info_new = info
		inner_sampler = sampler = create_mcmc(info_new, dim, proposal, flow)

		# Train the flow
		opt = torch.optim.Adam(flow.parameters(), lr=config['lr'])
		opt_scheduler = torch.optim.lr_scheduler.StepLR(
			optimizer=opt,
			step_size=config['step_size_scheduler'],
			gamma=config['gamma']
		)
		batch_size = config['batch_size']
		batch_size_mcmc = config['batch_size_mcmc']
		sampler = LearnMCMC(
			n_mcmc_steps=int(batch_size/batch_size_mcmc),
			n_train_steps=config['n_train_steps'],
			opt=opt,
			opt_scheduler=opt_scheduler,
			loss_type=info['loss_type'],
			inner_sampler=inner_sampler,
			flow=flow
		)

		# Launch training
		sampler.train_flow(
			x_s_t_0=x_init_training,
			target=dist.log_prob,
			base=proposal,
			temp=1.0/beta,
			verbose=True
		)

		# Sample the target from the flow
		samples = sampler.sample(
			x_s_t_0=x_init_sampling,
			warmup_steps=int(0.25 * info['n_steps']),
			n_steps=info['n_steps'],
			target=dist.log_prob,
			temp=1.0/beta,
			verbose=True
		).detach()

		# Weight of the mode
		mask = samples[...,int(dim / 2)] > 0
		mode_weight = mask.float().mean(dim=0).mean()

		# Evaluate the variance of each mode
		var = torch.ones((config['batch_size_mcmc'], ))
		for i in range(config['batch_size_mcmc']):
			var[i] = torch.sqrt(samples[mask[:,i].flatten(),i,int(dim/2)].var(dim=0) + samples[~mask[:,i].flatten(),i,int(dim/2)].var(dim=0))
		perc_var_nan = torch.sum(torch.isnan(var)) / var.shape[0]
		var = torch.nanmedian(var)

		# Compute IMH weights and the participation ratio
		samples = samples.view((-1, dim))
		imh_weights, participation_ratio = compute_imh_is_weights(flow, dist, proposal,
			batch_size=512, samples_target=samples[torch.randint(0, samples.shape[0], (512,))])

		all_losses[method_name] = {
			'mode_weight' : float(mode_weight.detach().cpu().clone()),
			'perc_var_nan' : float(perc_var_nan.detach().cpu().clone()),
			'var' : float(var.detach().cpu().clone()),
			'imh_weights' : float(imh_weights.mean().detach().cpu().clone()),
			'participation_ratio' : float(participation_ratio.detach().cpu().clone())
		}
		all_infos[method_name] = info

		# Add acceptances if possible
		if sampler.has_diagnostics('global_acceptance'):
			all_losses[method_name]['global_acceptance'] = float(sampler.get_diagnostics('global_acceptance').mean())
		if sampler.has_diagnostics('local_acceptance'):
			all_losses[method_name]['local_acceptance'] = float(sampler.get_diagnostics('local_acceptance').mean())

		# Dump everything
		with open('{}/{}_all_losses_dim_{}_{}.pkl'.format(config['save_path'], config['save_name'], dim, seed), 'wb') as f:
			pickle.dump(all_losses, f)
		with open('{}/{}_all_infos_dim_{}_{}.pkl'.format(config['save_path'], config['save_name'], dim, seed), 'wb') as f:
			pickle.dump(all_infos, f)

		# Save the flow
		if save_flow:
			torch.save(flow.state_dict(), '{}/{}_dim_{}.pth'.format(config['save_path'], info['save_name'], dim))

if __name__ == "__main__":
    # Libraries
    import argparse
    import yaml
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("global_config")
    parser.add_argument("local_config")
    parser.add_argument('--seed', type=int)
    parser.add_argument('--n_train_steps', type=int, default=0)
    parser.add_argument('--mala_sampler', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    # Freeze the seed
    set_seed(args.seed)
    # Load the config
    with open(args.global_config, 'r') as f:
        global_config = yaml.safe_load(f)
    with open(args.local_config, 'r') as f:
        local_config = yaml.safe_load(f)
    config = local_config | global_config
    # Check if flow must be saved
    if 'save_results' in config:
        save_flow = True
        config['save_path'] = config['save_results']
    else:
        save_flow = False
    if args.n_train_steps > 0:
    	config['n_train_steps'] = args.n_train_steps
    config['mala_sampler'] = args.mala_sampler
    # Get the Pytorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Run the experiment
    main(config, device, args.seed, save_flow=save_flow)