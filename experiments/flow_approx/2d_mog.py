# Libraries
import torch
from tqdm import trange
from flow.realnvp import RealNVP
from experiments.utils import set_seed, create_mcmc

# Make the target distribution
def make_target():
	mix = torch.distributions.Categorical(torch.ones((4,), device=device))
	locs = 1.5 * torch.ones((4,2), device=device)
	locs[0] *= -1
	locs[1,0] *= -1
	locs[2] = - locs[1]
	comp = torch.distributions.MultivariateNormal(
	    loc=locs,
	    covariance_matrix=torch.stack([0.25 * torch.eye(2, device=device)] * 4)
	)
	dist = torch.distributions.MixtureSameFamily(mix, comp)
	return dist

# Forward Kullback-Leiber loss
def forward_kl(target, proposal, flow, batch_size):
    y = target.sample(sample_shape=(batch_size,))
    u, minus_log_jac = flow.inverse(y)
    est = target.log_prob(y) - proposal.log_prob(u) - minus_log_jac
    return est.mean()

# Train a normalizing flow
def train_flow(target, proposal, device, n_training_steps=4000, n_realnvp_blocks=4, hidden_dim=64, hidden_depth=3, lr=1e-2,
	patience=250, decay_rate=0.98, batch_size=4096):

	# Make the flow
	flow = RealNVP(
			dim=2,
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
		gamma=decay_rate
	)

	# Training loop
	r = trange(n_training_steps, unit='step', desc='loss', leave=True)
	for step_id in r:

		# Reset optimizer
		optimizer.zero_grad()

		# Compute the loss
		loss = forward_kl(target, proposal, flow, batch_size)
		loss += 5e-5 * flow.get_weight_scale()

		# Backward pass
		loss.backward()

		# Optimizer step
		optimizer.step()
		lr_scheduler.step()

		# Debug
		r.set_description('loss = {}'.format(round(loss.item(), 4)), refresh=True)

	return flow

# Main function
x_min = -3.375
x_max = 3.375
y_min = -3.375
y_max = 3.375
def main(save_path, device, grid_size=256):

	# Make the target and the proposal
	target = make_target()
	proposal = torch.distributions.MultivariateNormal(
		loc=torch.zeros((2,), device=device),
		covariance_matrix=torch.eye(2, device=device)
	)

	# Starting point
	start_orig = target.component_distribution.sample(sample_shape=(8,))[:,0]

	# Train the flow
	flow = train_flow(target, proposal, device)

	# Make a grid
	xx, yy = torch.meshgrid(
	    torch.linspace(x_min, x_max, grid_size),
	    torch.linspace(y_min, y_max, grid_size)
	)
	zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2).to(device)

	# Get a mapping of the push-forward
	zz_inv, zz_inv_log_jac = flow.inverse(zz)
	log_prob_forward = proposal.log_prob(zz_inv) + zz_inv_log_jac
	torch.save(log_prob_forward.reshape((grid_size, grid_size)).clone().cpu(), '{}/log_prob_forward.pt'.format(save_path))

	# Get a mapping of the push-backward
	zz_fwd, zz_fwd_log_jac = flow.forward(zz)
	log_prob_backward = target.log_prob(zz_fwd) + zz_fwd_log_jac
	torch.save(log_prob_backward.reshape((grid_size, grid_size)).clone().cpu(), '{}/log_prob_backward.pt'.format(save_path))

	# Get a mapping of the target
	log_prob_target = target.log_prob(zz)
	torch.save(log_prob_target.reshape((grid_size, grid_size)).clone().cpu(), '{}/log_prob_target.pt'.format(save_path))

	# Get a mapping of the proposal
	log_prob_proposal = proposal.log_prob(zz)
	torch.save(log_prob_proposal.reshape((grid_size, grid_size)).clone().cpu(), '{}/log_prob_proposal.pt'.format(save_path))

	# Sample the target with NeuTra MCMC
	sampler = create_mcmc({
		'name' : 'mala',
		'neutralize' : True,
		'params' : {
			'step_size' : 0.01,
			'target_acceptance' : 0.75
		}

	}, 2, proposal, flow)
	samples_data = sampler.sample(
		x_s_t_0=start_orig.clone().to(device),
		n_steps=128,
		target=target.log_prob,
		warmup_steps=0,
		verbose=True
	).detach()
	samples_data[0] = start_orig
	samples_latent, _ = flow.inverse(samples_data)
	torch.save(samples_data.clone().cpu(), '{}/samples_data_neutra.pt'.format(save_path))
	torch.save(samples_latent.clone().cpu(), '{}/samples_latent_neutra.pt'.format(save_path))

	# Sample the target with adaptive i-SIR
	sampler = create_mcmc({
		'name' : 'adaptive_isir',
		'neutralize' : False,
		'params' : {
			'N' : 64,
		}

	}, 2, proposal, flow)
	samples_data = sampler.sample(
		x_s_t_0=start_orig.clone().to(device),
		n_steps=128,
		target=target.log_prob,
		warmup_steps=0,
		verbose=True
	).detach()
	samples_data[0] = start_orig
	samples_latent, _ = flow.inverse(samples_data)
	torch.save(samples_data.clone().cpu(), '{}/samples_data_isir.pt'.format(save_path))
	torch.save(samples_latent.clone().cpu(), '{}/samples_latent_isir.pt'.format(save_path))

	# Sample the target with TESS
	sampler = create_mcmc({
		'name' : 'ess',
		'neutralize' : True,
		'params' : {
			'dummy' : None,
		}

	}, 2, proposal, flow)
	samples_data = sampler.sample(
		x_s_t_0=start_orig.clone().to(device),
		n_steps=128,
		target=target.log_prob,
		warmup_steps=0,
		verbose=True
	).detach()
	samples_data[0] = start_orig
	samples_latent, _ = flow.inverse(samples_data)
	torch.save(samples_data.clone().cpu(), '{}/samples_data_tess.pt'.format(save_path))
	torch.save(samples_latent.clone().cpu(), '{}/samples_latent_tess.pt'.format(save_path))

	# Sample the target with RWMH
	sampler = create_mcmc({
		'name' : 'rwmh',
		'neutralize' : True,
		'params' : {
			'step_size' : 0.01,
			'target_acceptance' : 0.75
		}

	}, 2, proposal, flow)
	samples_data = sampler.sample(
		x_s_t_0=start_orig.clone().to(device),
		n_steps=128,
		target=target.log_prob,
		warmup_steps=0,
		verbose=True
	).detach()
	samples_data[0] = start_orig
	samples_latent, _ = flow.inverse(samples_data)
	torch.save(samples_data.clone().cpu(), '{}/samples_data_rwmh.pt'.format(save_path))
	torch.save(samples_latent.clone().cpu(), '{}/samples_latent_rwmh.pt'.format(save_path))

if __name__ == "__main__":
    # Libraries
    import argparse
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('save_path', type=str)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    # Freeze the seed
    set_seed(args.seed)
    # Get the Pytorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Run the experiment
    main(args.save_path, device)