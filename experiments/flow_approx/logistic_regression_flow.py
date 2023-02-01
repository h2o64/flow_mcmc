# Flow training for logistic regression

# Libraries
import torch
from experiments.utils import set_seed, create_mcmc
from models.logistic_regression import HorseshoeLogisticRegression, load_german_credit, make_iaf_flow
from mcmc.learn_flow import LearnMCMC
from flow.realnvp import RealNVP
import matplotlib.pyplot as plt
from mcmc.utils import compute_imh_is_weights
from tqdm import trange
from sklearn.model_selection import train_test_split

# Main function
def main(config, device, seed):

	# Set NeuTra config
	if config['neutra_flow']:
		config['loss_type'] = 'backward_kl'
		config['lr'] = 1e-2
		config['batch_size'] = 4096
		config['gamma'] = 0.10
		config['step_size_scheduler'] = 1000

	# Set the seed
	set_seed(seed)

	# Load the data
	X, y = load_german_credit()
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

	# Make the logistic regression
	target_train = HorseshoeLogisticRegression(X=X_train, y=y_train, device=device)
	target_test = HorseshoeLogisticRegression(X=X_test, y=y_test, device=device)
	dim = X.shape[1] * 2 + 1

	# Make the proposal
	proposal = torch.distributions.MultivariateNormal(
		loc=torch.zeros((dim,)).to(device),
		covariance_matrix=torch.eye(dim).to(device) * config['base_var'],
		validate_args=False
	)

	# Make the sampler
	if config['neutra_flow']:
		flow = make_iaf_flow(dim).to(device)
		inner_sampler = sampler = create_mcmc(
			conf_={
				'name' : 'adaptive_isir',
				'neutralize' : False,
				'params' : {
					'N': 64
				},
			},
			dim=dim,
			proposal=proposal,
			flow=flow
		)
		opt = torch.optim.Adam(flow.parameters(), lr=config['lr'], weight_decay=1e-4)
	else:
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
		inner_sampler = sampler = create_mcmc(
			conf_={
				'name' : 'flex2mcmc',
				'neutralize' : False,
				'params' : {
					'local_sampler' : 'hmc',
					'n_local_mcmc': config['n_local_mcmc'],
					'N': config['N']
				},
				'local_sampler_params' : {
					'step_size' : 0.001,
					'trajectory_length' : 8,
					'adapt_step_size': True
				}
			},
			dim=dim,
			proposal=proposal,
			flow=flow
		)
		opt = torch.optim.Adam(flow.parameters(), lr=config['lr'])

	# Train the flow
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
		flow=flow,
		use_weight_reg=False
	)

	# Good init point
	x_init_training = proposal.sample(sample_shape=(batch_size_mcmc,)).to(device)
	x_init_training = x_init_training.detach().clone().to(device)

	# Launch training
	sampler.train_flow(
		x_s_t_0=x_init_training,
		target=target_train.log_prob,
		base=proposal,
		verbose=True,
		warmup_steps=int(batch_size/batch_size_mcmc)
	)

	# Save the flow
	if len(config['save_path']) > 0:
		if config['neutra_flow']:
			torch.save(flow.state_dict(), '{}/logistic_regression_flow_neutra.pth'.format(config['save_path']))
		else:
			torch.save(flow.state_dict(), '{}/logistic_regression_flow.pth'.format(config['save_path']))

	# Record the loss
	loss = torch.FloatTensor(sampler.get_diagnostics('losses'))

	# Sample the target from the flow
	x_init = flow.forward(proposal.sample((batch_size_mcmc,)))[0].detach().clone()
	samples = sampler.sample(
		x_s_t_0=x_init,
		warmup_steps=int(0.25 * config['n_steps']),
		n_steps=config['n_steps'],
		target=target_train.log_prob,
		verbose=True
	)
	samples = samples.reshape((-1, dim))
	log_prob = target_test.log_prob(samples).mean()

	# Sample from the flow
	z, log_jac = flow.inverse(samples)
	log_prob_flow = (proposal.log_prob(z) + log_jac).mean()

	# Load ground truth samples
	ground_truth_samples = torch.load(config['ground_truth_location'], map_location=device).reshape((-1,dim))
	ground_truth_log_prob = target_test.log_prob(ground_truth_samples).mean()

	# Forward KL
	forward_kl = ground_truth_log_prob - log_prob_flow

	# Biais, variance and squared error
	bias = ground_truth_samples.mean(dim=0) - samples.mean(dim=0)
	bias_squared = torch.square(bias)
	variance = torch.square(samples - samples.mean(dim=0)).mean(dim=0)
	inst_biais = bias
	inst_biais_squared = torch.square(bias)
	inst_variance = torch.square(ground_truth_samples.mean(dim=0) - samples).mean(dim=0)
	error_sq = torch.square(ground_truth_samples.mean(dim=0) - samples.mean(dim=0)).mean()

	# Compute IMH weights and the participation ratio
	imh_weights, participation_ratio = compute_imh_is_weights(flow, target_test, proposal,
		batch_size=512, samples_target=samples[torch.randint(0, samples.shape[0], (512,))])

	# Plot various metrics
	print('energy = ', float(-log_prob.detach().cpu().clone()))
	print('log_prob = ', float(log_prob.detach().cpu().clone()))
	print('log_prob_flow = ', float(log_prob_flow.detach().cpu().clone()))
	print('forward_kl = ', float(forward_kl.detach().cpu().clone()))
	print('mean(biais) = ', float(bias.mean().detach().cpu().clone()))
	print('mean(bias_squared) = ', float(bias_squared.mean().detach().cpu().clone()))
	print('mean(variance) = ', float(variance.mean().detach().cpu().clone()))
	print('mean(inst_biais) = ', float(inst_biais.mean().detach().cpu().clone()))
	print('mean(inst_biais_squared) = ', float(inst_biais_squared.mean().detach().cpu().clone()))
	print('mean(inst_variance) = ', float(inst_variance.mean().detach().cpu().clone()))
	print('error_sq = ', float(error_sq.detach().cpu().clone()))
	print('imh_weights = ', float(imh_weights.mean().detach().cpu().clone()))
	print('participation_ratio = ', float(participation_ratio.detach().cpu().clone()))

	# Show the loss
	plt.figure(figsize=(20,10))
	plt.plot(loss.detach().cpu().clone())
	plt.xlabel('Iteration')
	plt.ylabel('Loss ({})'.format(config['loss_type']))
	plt.yscale('log')
	plt.savefig('loss.png')
	# plt.show()

if __name__ == "__main__":
	# Libraries
	import argparse
	# Parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_var', type=float, default=1e-2)
	parser.add_argument('--n_realnvp_blocks', type=int, default=6)
	parser.add_argument('--hidden_dim', type=int, default=64)
	parser.add_argument('--hidden_depth', type=int, default=3)
	parser.add_argument('--loss_type', type=str, default='forward_kl')
	parser.add_argument('--n_train_steps', type=int, default=3000)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--batch_size', type=int, default=4096)
	parser.add_argument('--batch_size_mcmc', type=int, default=16)
	parser.add_argument('--gamma', type=float, default=0.98)
	parser.add_argument('--step_size_scheduler', type=int, default=75)
	parser.add_argument('--n_local_mcmc', type=int, default=20)
	parser.add_argument('--N', type=int, default=128)
	parser.add_argument('--n_steps', type=int, default=512)
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--save_path', type=str, default='')
	parser.add_argument('--ground_truth_location', type=str, default='/tmp/ground_truth_samples.pt')
	parser.add_argument('--neutra_flow', action=argparse.BooleanOptionalAction)
	args = parser.parse_args()
	# Freeze the seed
	set_seed(args.seed)
	# Load the config
	config = vars(args)
	# Get the Pytorch device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# Run the experiment
	main(config, device, args.seed)
