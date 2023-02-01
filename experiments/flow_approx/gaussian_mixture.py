# Mixture of gaussians
# Details :
# - Sampling a gaussian mixture in high-dimension with different
#   samplers using a same RealNVP flow

# Libraries
import torch
import time
import glob
import pyro.ops.stats as stats
from experiments.utils import set_seed, create_mcmc, algorithms_with_flow
from models.gaussian_mixture import make_target, make_proposal
from flow.realnvp import RealNVP

# Compute the histogram of our distribution of modes
def compute_histogram(x, target, weights=None):
    # Get the means
    means = target.component_distribution.loc
    # Get the closest ones 
    closest = torch.argmin(torch.linalg.norm(x[:,None,:] - means, dim=2), dim=1)
    # Compute the histogram
    ret_ = torch.bincount(closest, weights=weights) / x.shape[0]
    ret = torch.zeros((4,)).to(means.device)
    ret[:ret_.shape[0]] = ret_
    return ret

# Compute average of the KL of each mode
def average_kl(samples, target, weights=None):
    # Get the means of the target
    means = target.component_distribution.loc
    dim = means.shape[-1]
    n_modes = means.shape[0]
    # Compute the closest modes
    samples_shape = samples.shape
    samples = samples.view((-1,dim))
    closest = torch.argmin(torch.linalg.norm(samples[:,None,:] - means, dim=2), dim=1)
    closest = closest.view(samples_shape[:-1])
    samples = samples.view(samples_shape)
    # Compute the mean and covariance of each mode
    est_means = torch.zeros((samples_shape[1], n_modes, dim)).to(means.device)
    est_covs = torch.zeros((samples_shape[1], n_modes, dim, dim)).to(means.device)
    ret_kl = torch.zeros((samples_shape[1],))
    for batch_id in range(samples_shape[1]):
        for mode_id in range(n_modes):
            mode_mask = closest[:,batch_id] == mode_id
            samples_mode = samples[mode_mask, batch_id]
            if weights is not None:
                est_means[batch_id, mode_id] = (weights[mode_mask, None] * samples_mode).sum(dim=0)
                est_means[batch_id, mode_id] /= weights[mode_mask].sum()
                est_covs[batch_id, mode_id] = samples_mode.T.cov(aweights=weights[mode_mask])
            else:
                est_means[batch_id, mode_id] = samples_mode.mean(dim=0)
                est_covs[batch_id, mode_id] = samples_mode.T.cov()
        # Make the multivariate normal out of it
        try:
            est_dist = torch.distributions.MultivariateNormal(
                loc=est_means[batch_id],
                covariance_matrix=est_covs[batch_id]
            )
            ret_kl[batch_id] = torch.distributions.kl.kl_divergence(target.component_distribution, est_dist).mean()
        except:
            ret_kl[batch_id] = float('inf')
    return ret_kl

# Get hyperparameters for the flow
def get_hyperparams_flow(dim):
    # Adapt the hidden dimension to the problem
    min_dim, min_hidden_dim = 16, 64
    max_dim, max_hidden_dim = 256, 256
    a = (max_hidden_dim - min_hidden_dim) / (max_dim - min_dim)
    b = min_hidden_dim - min_dim * a
    hidden_dim = int(a * dim + b)
    # Adapt the hidden depth to the problem
    min_dim, min_hidden_depth = 16, 3
    max_dim, max_hidden_depth = 256, 3
    a = (max_hidden_depth - min_hidden_depth) / (max_dim - min_dim)
    b = min_hidden_depth - min_dim * a
    hidden_depth = int(a * dim + b)
    # Adapt the number of realnvp blocks to the problem
    min_dim, min_n_realnvp_blocks = 16, 4
    max_dim, max_n_realnvp_blocks = 256, 8
    a = (max_n_realnvp_blocks - min_n_realnvp_blocks) / (max_dim - min_dim)
    b = min_n_realnvp_blocks - min_dim * a
    n_realnvp_blocks = int(a * dim + b)
    # Return all three
    return hidden_dim, hidden_depth, n_realnvp_blocks

# Main experiment
def main(config, device, seed, compute_rhat):

    # Browse all the dimensions
    if 'debug' in config:
        dims = [32, 64, 128]
    else:
        dims = [16, 32, 64, 128, 256]
    for dim in dims:

        # Load the corresponding weight saves
        weight_files = list(glob.glob('{}/dim_{}/net_flow_*.pth'.format(config['flow_path'], dim)))
        if not (len(weight_files) > 0): continue
        weight_files = sorted(weight_files, key=lambda x : int(x.split('/')[-1].split('_')[-1].split('.')[0]), reverse=False)

        # Make the target and proposal
        target = make_target(dim, device)
        proposal = make_proposal(target)

        # Number of steps
        if dim == 256:
            n_steps = 1500
        elif dim == 128:
            n_steps = 1300
        elif dim == 64:
            n_steps = 1100
        elif dim == 32:
            n_steps = 900
        else:
            n_steps = 700

        # Build the initial MCMC state
        start_orig = target.component_distribution.sample(sample_shape=(int(config['batch_size'] / 4),))
        start_orig = start_orig.view((-1, dim))

        # Build the flow
        hidden_dim, hidden_depth, n_realnvp_blocks = get_hyperparams_flow(dim)
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
        if dim >= 256:
            flow.load_state_dict(torch.load(weight_files[int(len(weight_files) / 8)], map_location=device))
        else:
            flow.load_state_dict(torch.load(weight_files[1], map_location=device))

        # Browse all methods
        for method_name, info in config['methods'].items():

            # Reset the seed
            set_seed(seed)

            # Show debug info
            print("============ {} (dim = {}) =============".format(method_name, dim))

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
                n_steps=n_steps,
                target=target.log_prob,
                warmup_steps=info['warmup_steps'],
                verbose=True
            ).detach()
            e = time.time()
            elapsed = e - s
            print(f"Elapsed: {elapsed:.2f} s")

            # Compute Rhat
            if compute_rhat and samples.shape[1] > 2 and not 'debug' in config:
                rhat = torch.stack([torch.max(stats.gelman_rubin(samples[:i], chain_dim=0, sample_dim=1)) for i in range(10, samples.shape[0], 10)]).clone().cpu()
                rhat_idx = torch.min(torch.nonzero(rhat < 1.05))
                torch.save(rhat_idx.clone().cpu(), '{}/rhat_idx_{}_dim_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, seed))
                torch.save(rhat.clone().cpu(), '{}/rhat_{}_dim_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, seed))
                continue

            # Save elapsed time
            if not 'debug' in config:
                torch.save(torch.tensor(elapsed), '{}/elapsed_time_{}_dim_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, seed))

            # Compute and save ESS
            if info['name'] != 'adaptive_is' and not 'debug' in config:
                ess = stats.effective_sample_size(samples, chain_dim=0, sample_dim=1).cpu()
                ess_per_sec = ess / elapsed
                torch.save(ess.clone().cpu(), '{}/ess_{}_dim_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, seed))
                torch.save(ess_per_sec.clone().cpu(), '{}/ess_per_sec_{}_dim_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, seed))

            # Compute the histograms
            hists = []
            for i in range(samples.shape[1]):
                hists.append(compute_histogram(samples[:,i], target, weights=sampler.get_diagnostics('weights')))
            hist = torch.stack(hists)
            torch.save(hist.clone().cpu(), '{}/hist_{}_dim_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, seed))
            hist_sorted = hist.sort(dim=-1).values
            torch.save(hist_sorted.clone().cpu(), '{}/hist_sorted_{}_dim_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, seed))

            # Compute the average KL of the modes
            avg_kl = average_kl(samples, target, weights=sampler.get_diagnostics('weights'))
            torch.save(avg_kl.clone().cpu(), '{}/avg_kl_{}_dim_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, seed))

            # Save the MC ESS
            if sampler.has_diagnostics('ess') and not 'debug' in config:
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
            if not 'debug' in config:
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
    parser.add_argument('--compute_rhat', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    # Freeze the seed
    set_seed(args.seed)
    # Load the config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Get the Pytorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Run the experiment
    main(config, device, args.seed, args.compute_rhat)
