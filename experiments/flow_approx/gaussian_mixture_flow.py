# Train flows for the gaussian mixture

# Libraries
import math
import torch
from tqdm import trange
from experiments.utils import set_seed

# Gaussian mixture model
from models.gaussian_mixture import make_target, make_proposal

# Flow
from flow.realnvp import RealNVP

# Set seed for cpu and CUDA, get device
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Forward Kullback-Leiber loss
def forward_kl(target, proposal, flow, batch_size):
    y = target.sample(sample_shape=(batch_size,))
    u, minus_log_jac = flow.inverse(y)
    est = target.log_prob(y) - proposal.log_prob(u) - minus_log_jac
    return est.mean()

def main(dim, n_steps, batch_size, lr, decay_rate, patience, hidden_dim, hidden_depth, n_realnvp_blocks, checkpoint_path, wd=1e-4):

	# Make the target and the proposal
	target = make_target(dim, device)
	proposal = make_proposal(target)

	# Make the flow
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
	optimizer = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=wd)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(
		optimizer=optimizer,
		step_size=patience,
		gamma=decay_rate
	)

	# Training loop
	r = trange(n_steps, unit='step', desc='loss', leave=True)
	losses = torch.ones((n_steps,))
	for step_id in r:

		# Reset optimizer
		optimizer.zero_grad()

		# Compute the loss
		loss = forward_kl(target, proposal, flow, batch_size)
		loss += 5e-5 * flow.get_weight_scale()

		# Backward pass
		loss.backward()
		losses[step_id] = loss.item()

		# Optimizer step
		optimizer.step()
		lr_scheduler.step()

		# Debug
		r.set_description('loss = {}'.format(round(loss.item(), 4)), refresh=True)

		# Save the progress
		if step_id % int(0.05 * n_steps) == 0:
			torch.save(flow.state_dict(), checkpoint_path + '/net_flow_{:>06d}.pth'.format(step_id))
			torch.save(losses, checkpoint_path + '/losses_{:>06d}.pth'.format(step_id))

	torch.save(flow.state_dict(), checkpoint_path + '/net_flow_{:>06d}.pth'.format(step_id))
	torch.save(losses, checkpoint_path + '/losses_{:>06d}.pth'.format(step_id))

if __name__ == "__main__":
	# Libraries
	import argparse
	import os

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--dim", type=int, default=3, required=True)
	parser.add_argument("--checkpoint_path", type=str, required=True)
	#parser.add_argument("--n_steps", type=int, defaut=25000)
	parser.add_argument("--batch_size", type=int, default=8192)
	#parser.add_argument("--lr", type=float, default=1e-2)
	parser.add_argument("--decay_rate", type=float, default=0.99)
	#parser.add_argument("--patience", type=int, default=100)
	#parser.add_argument("--hidden_dim", type=int, default=64)
	parser.add_argument('--seed', type=int, default=42)
	args = parser.parse_args()

	# Show the dimension
	print('dim = ', args.dim)

	# Set the seed
	set_seed(args.seed)

	# Get the Pytorch device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Check if folder exist
	if not os.path.isdir(args.checkpoint_path):
		os.mkdir(args.checkpoint_path)

	# Adapt the number of steps to the problem
	min_dim, min_it = 16, 50000
	max_dim, max_it = 256, 90000
	a = (max_it - min_it) / (max_dim - min_dim)
	b = min_it - min_dim * a
	n_steps = int(a * args.dim + b)
	# Multiply the number of steps by a batch_size factor
	#n_steps *= 1 + (1 - (args.batch_size / 4096))
	n_steps = int(n_steps)
	print('n_steps = ', n_steps)

	# Adapt the patience to the problem
	min_dim, min_patience = 16, 100
	max_dim, max_patience = 256, 200
	a = (max_patience - min_patience) / (max_dim - min_dim)
	b = min_patience - min_dim * a
	patience = int(a * args.dim + b)
	print('patience = ', patience)

	# Adapt the learning rate to the problem
	min_dim, min_log_lr = 16, math.log(1e-2)
	max_dim, max_log_lr = 256, math.log(1e-4)
	a = (max_log_lr - min_log_lr) / (max_dim - min_dim)
	b = min_log_lr - min_dim * a
	lr = math.exp(a * args.dim + b)
	print('lr = ', lr)

	# Adapt the hidden dimension to the problem
	min_dim, min_hidden_dim = 16, 64
	max_dim, max_hidden_dim = 256, 256
	a = (max_hidden_dim - min_hidden_dim) / (max_dim - min_dim)
	b = min_hidden_dim - min_dim * a
	hidden_dim = int(a * args.dim + b)
	print('hidden_dim = ', hidden_dim)

	# Adapt the hidden depth to the problem
	min_dim, min_hidden_depth = 16, 3
	max_dim, max_hidden_depth = 256, 3
	a = (max_hidden_depth - min_hidden_depth) / (max_dim - min_dim)
	b = min_hidden_depth - min_dim * a
	hidden_depth = int(a * args.dim + b)
	print('hidden_depth = ', hidden_depth)

	# Adapt the number of realnvp blocks to the problem
	min_dim, min_n_realnvp_blocks = 16, 4
	max_dim, max_n_realnvp_blocks = 256, 8
	a = (max_n_realnvp_blocks - min_n_realnvp_blocks) / (max_dim - min_dim)
	b = min_n_realnvp_blocks - min_dim * a
	n_realnvp_blocks = int(a * args.dim + b)
	print('n_realnvp_blocks = ', n_realnvp_blocks)

	# Launch the training
	main(args.dim, n_steps, args.batch_size, lr, args.decay_rate, patience, hidden_dim, hidden_depth, n_realnvp_blocks, args.checkpoint_path, wd=1e-4)
