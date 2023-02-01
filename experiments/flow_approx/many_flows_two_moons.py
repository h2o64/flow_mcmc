# Libraries
import numpy as np
import torch
import normflows as nf
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
from tqdm import trange
from experiments.utils import set_seed
from flow.normflows_wrappers import WrappedNormFlowModel, BaseDistributionWrapper
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
import os.path

# Make planar flow
def make_planar_flow(K=16):
	flows = []
	for i in range(K):
		flows.append(nf.flows.Planar((2,)))
	q0 = nf.distributions.DiagGaussian(2, trainable=False)
	nfm = nf.NormalizingFlow(q0=q0, flows=flows)
	return nfm

# Make planar flow
def make_radial_flow(K=16):
	flows = []
	for i in range(K):
		flows.append(nf.flows.Radial((2,)))
	q0 = nf.distributions.DiagGaussian(2, trainable=False)
	nfm = nf.NormalizingFlow(q0=q0, flows=flows)
	return nfm

# Make NICE flow
def make_nice_flow(K=16):
	b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(2)])
	flows = []
	for i in range(K):
	    net = nf.nets.MLP([2, 16, 16, 2], init_zeros=True)
	    if i % 2 == 0:
	        flows += [nf.flows.MaskedAffineFlow(b, net)]
	    else:
	        flows += [nf.flows.MaskedAffineFlow(1 - b, net)]
	    flows += [nf.flows.ActNorm(2)]
	q0 = nf.distributions.DiagGaussian(2, trainable=False)
	nfm = nf.NormalizingFlow(q0=q0, flows=flows)
	return nfm

# Make RealNVP flow
def make_rnvp_flow(K=16):
	b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(2)])
	flows = []
	for i in range(K):
	    s = nf.nets.MLP([2, 16, 16, 2], init_zeros=True)
	    t = nf.nets.MLP([2, 16, 16, 2], init_zeros=True)
	    if i % 2 == 0:
	        flows += [nf.flows.MaskedAffineFlow(b, t, s)]
	    else:
	        flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
	    flows += [nf.flows.ActNorm(2)]
	q0 = nf.distributions.DiagGaussian(2, trainable=False)
	nfm = nf.NormalizingFlow(q0=q0, flows=flows)
	return nfm

# Make Neural Spline flow
def make_nsp_flow(K=16, hidden_units=64, hidden_layers=2):
	flows = []
	for i in range(K):
	    flows += [nf.flows.AutoregressiveRationalQuadraticSpline(2, hidden_layers, hidden_units)]
	    flows += [nf.flows.LULinearPermute(2)]
	q0 = nf.distributions.DiagGaussian(2, trainable=False)
	nfm = nf.NormalizingFlow(q0=q0, flows=flows)
	return nfm

# Make IAF flow
def make_iaf_flow(K=16, hidden_features=16):
	flows = []
	for i in range(K):
	    flows.append(nf.flows.MaskedAffineAutoregressive(features=2, hidden_features=hidden_features))
	    flows.append(nf.flows.Permute(2))
	q0 = nf.distributions.DiagGaussian(2, trainable=False)
	nfm = nf.NormalizingFlow(q0=q0, flows=flows)
	return nfm

# Make residual flow
def make_residual_flow(K=16, hidden_units=64, hidden_layers=2):
	flows = []
	for i in range(K):
	    net = nf.nets.LipschitzMLP([2] + [hidden_units] * (hidden_layers - 1) + [2],
	                               init_zeros=True, lipschitz_const=0.9)
	    flows += [nf.flows.Residual(net, reduce_memory=True)]
	    flows += [nf.flows.ActNorm(2)]
	q0 = nf.distributions.DiagGaussian(2, trainable=False)
	nfm = nf.NormalizingFlow(q0=q0, flows=flows)
	return nfm

# Various flows
flows = {
	# 'planar' : ('Planar Flow (Rezende & Mohamed, 2015)', make_planar_flow),
	# 'radial' : ('Radial Flow (Rezende & Mohamed, 2015)', make_radial_flow),
	'nice' : ('NICE (Dinh et al., 2014)', make_nice_flow),
	'rnvp' : ('Real NVP (Dinh et al., 2016)', make_rnvp_flow),
	'iaf' : ('Masked Autoregressive Flow (Papamakarios et al., 2017)', make_iaf_flow),
	'nsp': ('Neural Spline Flow (Durkan et al., 2019)', make_nsp_flow),
	'residual' : ('Residual Flow (Chen et al., 2019)', make_residual_flow)
}

# Taken from https://stackoverflow.com/questions/32607552/scipy-speed-up-kernel-density-estimations-score-sample-method
import multiprocessing
def parrallel_score_samples(kde, samples, thread_count=int(0.875 * multiprocessing.cpu_count())):
    with multiprocessing.Pool(thread_count) as p:
        return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))

# Train a given flow
def train_flow(flow_builder, device, max_iter=30000, num_samples=1024, noise=0.1, lr=1e-3):

	# Make the flow
	nfm = flow_builder()
	nfm = nfm.to(device)

	# Initialize ActNorm
	x_np, _ = make_moons(num_samples, noise=noise)
	x = torch.tensor(x_np).float().to(device)
	_ = nfm.log_prob(x)

	# Train model
	optimizer = torch.optim.Adam(nfm.parameters(), lr=lr, weight_decay=1e-5)
	r = trange(max_iter)
	for it in r:
	    optimizer.zero_grad()
	    
	    # Get training samples
	    x_np, _ = make_moons(num_samples, noise=noise)
	    x = torch.tensor(x_np).float().to(device)
	    
	    # Compute loss
	    loss = nfm.forward_kld(x)
	    
	    # Do backprop and optimizer step
	    if ~(torch.isnan(loss) | torch.isinf(loss)):
	        loss.backward()
	        optimizer.step()
	    
	    # Log loss
	    r.set_description('loss = {:.2e}'.format(loss.item()))

	return nfm

# Main function
def main(save_path, load_path, grid_size, device, n_samples=1000):

	# Train all flows
	if len(save_path) > 0:
		# Train the flows
		for flow_id, (flow_name, flow_builder) in flows.items():
			print('-> ', flow_name)
			if flow_id == 'residual':
				nfm = train_flow(flow_builder, device, max_iter=20000, num_samples=2048, lr=1e-4)
			else:
				nfm = train_flow(flow_builder, device)
			torch.save(nfm.state_dict(), '{}/{}.pth'.format(save_path, flow_id))
		# Make a nice KDE estimate
		model_cv = GridSearchCV(
			KernelDensity(),
			{ 'bandwidth' : np.logspace(-4, -1, 256) },
			n_jobs=-1
		)
		dataset, _ = make_moons(int(1e5), noise=0.1)
		model_cv.fit(dataset)
		print(model_cv.best_params_)
		# {'bandwidth': 0.043598585425769235}
		# Take the best model
		kde = KernelDensity(**model_cv.best_params_)
		# kde = KernelDensity(bandwidth=0.043598585425769235)
		kde.fit(dataset)
		# Save the best model
		dump(kde, '{}/kde.joblib'.format(save_path))

	# Grid sizes
	data_x_min, data_x_max, data_y_min, data_y_max = -1.5, 2.5, -1, 1.5
	latent_x_min, latent_x_max, latent_y_min, latent_y_max = -3, 3, -3, 3

	# Display the push-backward and push-forward
	if len(load_path) > 0:
		# Make the grids
		xx_data, yy_data = torch.meshgrid(torch.linspace(data_x_min, data_x_max, grid_size), torch.linspace(data_y_min, data_y_max, grid_size))
		zz_data = torch.cat([xx_data.unsqueeze(2), yy_data.unsqueeze(2)], 2).view(-1, 2)
		zz_data = zz_data.float().to(device)
		xx_latent, yy_latent = torch.meshgrid(torch.linspace(latent_x_min, latent_x_max, grid_size), torch.linspace(latent_y_min, latent_y_max, grid_size))
		zz_latent = torch.cat([xx_latent.unsqueeze(2), yy_latent.unsqueeze(2)], 2).view(-1, 2)
		zz_latent = zz_latent.float().to(device)
		# Load the KDE
		kde = load('{}/kde.joblib'.format(load_path))
		# Make the reference plot
		proposal = BaseDistributionWrapper(flows['rnvp'][1]().to(device).q0)
		plt.figure(figsize=(10,5))
		plt.subplot(1, 2, 1)
		push_backward_ref = proposal.log_prob(zz_latent).detach().cpu().numpy().reshape((grid_size, grid_size))
		plt.contourf(xx_latent, yy_latent, np.exp(push_backward_ref), 20, cmap='RdGy')
		plt.grid(False)
		plt.title('Latent space')
		plt.subplot(1, 2, 2)
		push_forward_ref = parrallel_score_samples(kde, zz_data.detach().cpu().numpy()).reshape((grid_size, grid_size))
		plt.contourf(xx_data, yy_data, np.exp(push_forward_ref), 20, cmap='GnBu')
		plt.grid(False)
		plt.title('Data space')
		plt.tight_layout()
		plt.savefig('{}/{}.pdf'.format(load_path, 'reference'), bbox_inches="tight")
		plt.savefig('{}/{}.png'.format(load_path, 'reference'), bbox_inches="tight")
		# Browse the flows
		for flow_id, (flow_name, flow_builder) in flows.items():
			print('-> ', flow_name)
			# Load the flow
			nfm = flow_builder()
			nfm = nfm.to(device)
			nfm.load_state_dict(
				torch.load('{}/{}.pth'.format(load_path, flow_id), map_location=device)
			)
			flow = WrappedNormFlowModel(nfm)
			proposal = BaseDistributionWrapper(nfm.q0)
			# Compute the push-forward
			z, log_jac_z = flow.inverse(zz_data)
			push_forward = proposal.log_prob(z) + log_jac_z
			push_forward = push_forward.reshape((grid_size, grid_size)).detach().cpu().numpy()
			# Compute the push-backward
			x, log_jac_x = flow.forward(zz_latent)
			push_backward = torch.from_numpy(parrallel_score_samples(kde, x.detach().cpu().numpy())).float().to(device) + log_jac_x
			push_backward = push_backward.reshape((grid_size, grid_size)).detach().cpu().numpy()
			# # Make the plot
			# plt.figure(figsize=(10,5))
			# plt.subplot(1, 2, 1)
			# plt.contourf(xx_latent, yy_latent, np.exp(push_backward), 20, cmap='RdGy')
			# plt.grid(False)
			# plt.title('Latent space')
			# plt.subplot(1, 2, 2)
			# plt.contourf(xx_data, yy_data, np.exp(push_forward), 20, cmap='GnBu')
			# plt.grid(False)
			# plt.title('Data space')
			# plt.tight_layout()
			# plt.savefig('{}/{}.pdf'.format(load_path, flow_id), bbox_inches="tight")
			# plt.savefig('{}/{}.png'.format(load_path, flow_id), bbox_inches="tight")
			# Save the stuff
			for filepath, data in [
				('{}/xx_latent.pt'.format(load_path), xx_latent),
				('{}/yy_latent.pt'.format(load_path), yy_latent),
				('{}/xx_data.pt'.format(load_path), xx_data),
				('{}/yy_data.pt'.format(load_path), yy_data),
				('{}/push_backward_{}.pt'.format(load_path, flow_id), push_backward),
				('{}/push_forward_{}.pt'.format(load_path, flow_id), push_forward),
				('{}/push_backward_ref.pt'.format(load_path), push_backward_ref),
				('{}/push_forward_ref.pt'.format(load_path), push_forward_ref)
			]:
				if not os.path.isfile(filepath): torch.save(data, filepath)


if __name__ == "__main__":
    # Libraries
    import argparse
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--grid_size', type=int, default=512)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--force_cpu', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    # Freeze the seed
    set_seed(args.seed)
    # Get the Pytorch device
    if args.force_cpu:
    	device = torch.device('cpu')
    else:
    	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Run the experiment
    main(args.save_path, args.load_path, args.grid_size, device)