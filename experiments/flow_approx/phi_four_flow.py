# Hyper-parameters for PhiFour

# Libraries
import torch
from experiments.utils import set_seed, create_mcmc
from distribution.phi_four import PhiFour, PhiFourBase
from flow.realnvp import RealNVP
from mcmc.learn_flow import LearnMCMC
import matplotlib.pyplot as plt
from tqdm import trange
from mcmc.utils import compute_imh_is_weights

# Main function
def main(config, device, seed, save_flow=False):

	# Set the seed
	set_seed(seed)

	# Use the trained flow
	use_trained_flow = len(config['flow_path']) > 0

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
		inner_sampler = sampler = create_mcmc(
				conf_={
					'name' : 'mala',
					'neutralize' : False,
					'params' : {
						'step_size' : 0.001,
						'target_acceptance' : 0.75
					}
				},
				dim=dim,
				proposal=proposal,
				flow=flow
			)
	else:
		inner_sampler = sampler = create_mcmc(
			conf_={
				'name' : 'flex2mcmc',
				'neutralize' : False,
				'params' : {
					'local_sampler' : 'mala',
					'n_local_mcmc': config['n_local_mcmc'],
					'N': config['N']
				},
				'local_sampler_params' : {
					'step_size' : 0.001,
					'target_acceptance' : 0.75
				}
			},
			dim=dim,
			proposal=proposal,
			flow=flow
		)

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
		loss_type=config['loss_type'],
		inner_sampler=inner_sampler,
		flow=flow
	)

	# Starting point
	if not use_trained_flow or config['use_init_training']:
		x_init_training = torch.ones((batch_size_mcmc, dim)).to(device)
		x_init_training[int(0.5*batch_size_mcmc):,:] = -1
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

	# Train the flow
	if not use_trained_flow:

		# Launch the training
		sampler.train_flow(
			x_s_t_0=x_init_training.clone().to(device),
			target=dist.log_prob,
			base=proposal,
			temp=1.0/beta,
			verbose=True
		)

		# Record the loss
		loss = torch.FloatTensor(sampler.get_diagnostics('losses'))

	else:
		flow.load_state_dict(torch.load(config['flow_path'], map_location=device))

	# Sample the target from the flow
	if not config['use_init_training']:
		x_init_sampling = torch.ones((batch_size_mcmc, dim)).to(device).requires_grad_()
		opt_orig = torch.optim.SGD([x_init_sampling], lr=1e-3)
		orig_r = trange(35000, unit='step')
		for _ in orig_r:
			opt_orig.zero_grad()
			cur_energy = dist.U(x_init_sampling).sum()
			cur_energy.backward()
			opt_orig.step()
			orig_r.set_description('energy = {:.2e}'.format(float(cur_energy.item())))
		x_init_sampling = x_init_sampling.detach().clone().to(device)
	else:
		x_init_sampling = x_init_training.clone()
	samples = sampler.sample(
		x_s_t_0=x_init_sampling,
		warmup_steps=int(0.25 * config['n_steps']),
		n_steps=config['n_steps'],
		target=dist.log_prob,
		temp=1.0/beta,
		verbose=True
	)
	samples_shape = samples.shape
	samples = samples.view((-1, dim))
	log_prob = dist.log_prob(samples).mean()

	# Sample from the flow
	z, log_jac = flow.inverse(samples)
	log_prob_flow = (proposal.log_prob(z) + log_jac).mean()

	# Forward KL
	forward_kl = log_prob - log_prob_flow

	# Weight of the mode
	samples = samples.view(samples_shape)
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

	# Plot various metrics
	print('energy = ', float(-log_prob.detach().cpu().clone()))
	print('log_prob = ', float(log_prob.detach().cpu().clone()))
	print('log_prob_flow = ', float(log_prob_flow.detach().cpu().clone()))
	print('forward_kl = ', float(forward_kl.detach().cpu().clone()))
	print('mode_weight = ', float(mode_weight.detach().cpu().clone()))
	print('var = ', float(var.detach().cpu().clone()))
	print('perc_var_nan = ', float(perc_var_nan.detach().cpu().clone()))
	print('imh_weights = ', float(imh_weights.mean().detach().cpu().clone()))
	print('participation_ratio = ', float(participation_ratio.detach().cpu().clone()))
	if sampler.has_diagnostics('local_acceptance'):
	    local_acceptance = sampler.get_diagnostics('local_acceptance')
	    if isinstance(local_acceptance, torch.Tensor):
	        local_acceptance = local_acceptance.mean()
	    else:
	        local_acceptance = torch.tensor(local_acceptance)
	    print('local acceptance (mean) = {:.2f}%'.format(100 * float(local_acceptance)))
	if sampler.has_diagnostics('global_acceptance'):
	    global_acceptance = sampler.get_diagnostics('global_acceptance')
	    if isinstance(global_acceptance, torch.Tensor):
	        global_acceptance = global_acceptance.mean()
	    else:
	        global_acceptance = torch.tensor(global_acceptance)
	    print('global acceptance (mean) = {:.2f}%'.format(100 * float(global_acceptance)))

	# Show the loss
	if not use_trained_flow:
		plt.figure(figsize=(20,10))
		plt.plot(loss.detach().cpu().clone())
		plt.xlabel('Iteration')
		plt.ylabel('Loss ({})'.format(config['loss_type']))
		plt.savefig('loss.png')

	# Show the samples
	plt.figure(figsize=(10,10))
	samples = samples.detach().cpu().clone().view(samples_shape)
	for i in range(samples.shape[0]):
		plt.plot(samples[i,0], color='green', alpha=0.1)
	plt.savefig('samples.png')

	# Show the samples from the flow
	plt.figure(figsize=(10,10))
	samples_flow, _ = flow.forward(proposal.sample((config['n_steps'],)))
	samples_flow = samples_flow.detach().cpu().clone()
	for i in range(samples_flow.shape[0]):
		plt.plot(samples_flow[i], color='pink', alpha=0.1)
	plt.savefig('samples_flow.png')

if __name__ == "__main__":
	# Libraries
	import argparse
	# Parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--dim', type=int, default=128)
	parser.add_argument('--a', type=float, default=0.1)
	parser.add_argument('--b', type=float, default=0.0)
	parser.add_argument('--beta', type=float, default=20.0)
	parser.add_argument('--n_realnvp_blocks', type=int, default=6)
	parser.add_argument('--hidden_dim', type=int, default=64)
	parser.add_argument('--hidden_depth', type=int, default=2)
	parser.add_argument('--loss_type', type=str, default='forward_kl')
	parser.add_argument('--n_train_steps', type=int, default=7500)
	parser.add_argument('--lr', type=float, default=0.005)
	parser.add_argument('--batch_size', type=int, default=4096)
	parser.add_argument('--batch_size_mcmc', type=int, default=32)
	parser.add_argument('--gamma', type=float, default=0.95)
	parser.add_argument('--step_size_scheduler', type=int, default=100)
	parser.add_argument('--n_local_mcmc', type=int, default=25)
	parser.add_argument('--N', type=int, default=64)
	parser.add_argument('--n_steps', type=int, default=512)
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--mala_sampler', action=argparse.BooleanOptionalAction)
	parser.add_argument('--flow_path', type=str, default='')
	parser.add_argument('--use_init_training', action=argparse.BooleanOptionalAction)
	args = parser.parse_args()
	# Freeze the seed
	set_seed(args.seed)
	# Load the config
	config = vars(args)
	# Get the Pytorch device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# Run the experiment
	main(config, device, args.seed)