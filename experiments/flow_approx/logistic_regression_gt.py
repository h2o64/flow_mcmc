# Flow training for logistic regression

# Libraries
import torch
from experiments.utils import set_seed, create_mcmc
from models.logistic_regression import HorseshoeLogisticRegression, load_german_credit
from tqdm import trange

# Main function
def main(config, device, seed, save_flow=False):

	# Set the seed
	set_seed(seed)

	# Load the data
	X, y = load_german_credit()

	# Make the logistic regression
	target = HorseshoeLogisticRegression(X=X, y=y, device=device)
	dim = X.shape[1] * 2 + 1

	# Make the sampler
	sampler = create_mcmc(
		conf_={
			'name' : 'nuts',
			'neutralize' : False,
			'params' : {
				'step_size' : 0.001,
				'adapt_step_size' : True,
				'adapt_mass_matrix' : True
			}
		},
		dim=dim,
		proposal=None,
		flow=None
	)

	# Good init point
	x_init = torch.randn((config['batch_size'], dim)).to(device)
	x_init = x_init.detach().clone().to(device)

	# Sample the target from the flow
	samples = sampler.sample(
		x_s_t_0=x_init,
		warmup_steps=config['warmup_steps'],
		n_steps=config['n_steps'],
		target=target.log_prob,
		verbose=True
	)

	# Save the samples
	torch.save(samples.detach().cpu(), '{}/ground_truth_samples.pt'.format(config['save_path']))


if __name__ == "__main__":
	# Libraries
	import argparse
	# Parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--save_path', type=str)
	parser.add_argument('--batch_size', type=int, default=4)
	parser.add_argument('--warmup_steps', type=int, default=512)
	parser.add_argument('--n_steps', type=int, default=8192)
	parser.add_argument('--seed', type=int, default=42)
	args = parser.parse_args()
	# Freeze the seed
	set_seed(args.seed)
	# Load the config
	config = vars(args)
	# Get the Pytorch device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# Run the experiment
	main(config, device, args.seed)