# Logistic regression

# Libraries
import torch
import time
import pyro.ops.stats as stats
from experiments.utils import set_seed, create_mcmc, algorithms_with_flow
from flow.realnvp import RealNVP
from models.logistic_regression import HorseshoeLogisticRegression, load_german_credit, make_iaf_flow
from sklearn.model_selection import train_test_split
from tqdm import trange

# Main experiment
def main(config, device, seed, save_samples=False, neutra_flow=False):

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
        covariance_matrix=torch.eye(dim).to(device) * 1e-2
    )

    # Make the flow
    if neutra_flow:
        flow = make_iaf_flow(dim).to(device)
        flow.load_state_dict(torch.load('experiments/flow_approx/models/logistic_regression/logistic_regression_flow_neutra.pth', map_location=device))
        config['save_path'] = config['save_path'] + '/neutra_flow/'
    else:
        flow = RealNVP(
            dim=dim,
            channels=1,
            n_realnvp_blocks=config['flow_params']['n_realnvp_blocks'],
            block_depth=1,
            init_weight_scale=1e-6,
            hidden_dim=config['flow_params']['hidden_dim'],
            hidden_depth=config['flow_params']['hidden_depth'],
            residual=True,
            equivariant=False,
            device=device
        )
        flow.load_state_dict(torch.load('experiments/flow_approx/models/logistic_regression/logistic_regression_flow.pth', map_location=device))
        config['save_path'] = config['save_path'] + '/flex2_flow/'

    # Build the initial MCMC state
    start_orig, _ = flow.forward(proposal.sample((config['batch_size'],)))
    start_orig = start_orig.detach().clone().to(device)

    # Load ground truth samples
    ground_truth_samples = torch.load('/tmp/ground_truth_samples.pt',
        map_location=device).reshape((-1,dim))

    # Define the random projections
    n_random_projections = 128
    random_projs = torch.randn((n_random_projections, dim))
    random_projs /= torch.linalg.norm(random_projs, axis=-1)[...,None]

    # Compute the true histogram
    n_bins = 1024
    samples_real_random_proj = torch.matmul(ground_truth_samples.cpu(), random_projs[...,None])[...,0]
    real_random_proj_min_x, real_random_proj_max_x = samples_real_random_proj.min(dim=-1), samples_real_random_proj.max(dim=-1)
    real_random_proj_hist = torch.zeros((n_random_projections, n_bins))
    for i in range(n_random_projections):
        real_random_proj_hist[i] = torch.histc(samples_real_random_proj[i], bins=n_bins,
            min=float(real_random_proj_min_x.values[i]), max=float(real_random_proj_max_x.values[i]))
    real_random_proj_hist /= real_random_proj_hist.sum(dim=-1)[...,None]
    real_random_proj_cdf = real_random_proj_hist.cumsum(dim=-1)

    # Browse all methods
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
        sampler = create_mcmc(info, dim, proposal, flow_, use_reshape=True)

        # Run the sampler
        s = time.time()
        samples = sampler.sample(
            x_s_t_0=start_orig.clone().to(device).detach(),
            n_steps=info['n_steps'],
            target=target_train.log_prob,
            warmup_steps=info['warmup_steps'],
            verbose=True
        ).detach()
        e = time.time()
        elapsed = e - s
        print(f"Elapsed: {elapsed:.2f} s")

        # Save elapsed time
        torch.save(torch.tensor(elapsed), '{}/elapsed_time_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))

        # # Compute and save ESS
        # if info['name'] != 'adaptive_is':
        #     ess = stats.effective_sample_size(samples, chain_dim=0, sample_dim=1).cpu()
        #     ess_per_sec = ess / elapsed
        #     torch.save(ess.clone().cpu(), '{}/ess_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))
        #     torch.save(ess_per_sec.clone().cpu(), '{}/ess_per_sec_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))

        # # Compute and save Rhat
        # if info['name'] != 'adaptive_is':
        #     rhat = torch.max(stats.gelman_rubin(samples, chain_dim=0, sample_dim=1)).cpu()
        #     torch.save(rhat.clone().cpu(), '{}/rhat_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))

        # Compute the energy of the chains
        samples = samples.reshape((-1, dim))
        if not sampler.has_diagnostics('weights'):
            energy = -target_test.log_prob(samples).mean()
        else:
            weights = sampler.get_diagnostics('weights').clone()
            energy = -(weights[:,None] * target_test.log_prob(samples)).sum() / weights.sum()
        torch.save(energy.clone().cpu(), '{}/energy_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))

        # Compute some values
        if sampler.has_diagnostics('weights'):
            weights = sampler.get_diagnostics('weights')[:,None]
            samples_mean = torch.sum(weights * samples, dim=0) / weights.sum()
        else:
            samples_mean = samples.mean(dim=0)

        # Biais, variance and squared error
        bias = ground_truth_samples.mean(dim=0) - samples_mean
        bias_squared = torch.square(bias)
        if sampler.has_diagnostics('weights'):
            variance = (torch.square(samples - samples_mean) * weights).sum(dim=0)
            variance /= weights.sum()
        else:
            variance = torch.square(samples - samples_mean).mean(dim=0)
        # inst_biais = bias
        # inst_biais_squared = torch.square(bias)
        if sampler.has_diagnostics('weights'):
            inst_variance = (torch.square(ground_truth_samples.mean(dim=0) - samples) * weights).sum(dim=0)
            inst_variance /= weights.sum()
        else:
            inst_variance = torch.square(ground_truth_samples.mean(dim=0) - samples).mean(dim=0)
        error_sq = torch.square(ground_truth_samples.mean(dim=0) - samples_mean).mean()
        #torch.save(bias.mean().clone().cpu(), '{}/bias_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))
        torch.save(bias_squared.mean().clone().cpu(), '{}/bias_squared_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))
        torch.save(variance.mean().clone().cpu(), '{}/variance_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))
        #torch.save(inst_biais.mean().clone().cpu(), '{}/inst_biais_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))
        #torch.save(inst_biais_squared.mean().clone().cpu(), '{}/inst_biais_squared_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))
        torch.save(inst_variance.mean().clone().cpu(), '{}/inst_variance_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))
        torch.save(error_sq.clone().cpu(), '{}/error_sq_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))

        # Save the samples
        if save_samples:
            # Save the samples
            torch.save(samples.clone().cpu(), '{}/samples_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))

        # Compute the empirical histogram on random projections
        samples = samples.reshape((-1, dim))
        if sampler.has_diagnostics('weights'):
            weights = sampler.get_diagnostics('weights').cpu()
        else:
            weights = None
        samples = samples.cpu()
        samples_random_proj = torch.matmul(samples, random_projs[...,None])[...,0].cpu()
        random_proj_hist = torch.zeros((n_random_projections, n_bins))
        for i in range(n_random_projections):
            random_proj_hist[i] = torch.histogram(samples_random_proj[i], bins=n_bins,
                range=(float(real_random_proj_min_x.values[i]), float(real_random_proj_max_x.values[i])), weight=weights).hist
        random_proj_hist /= random_proj_hist.sum(dim=-1)[...,None]
        random_proj_cdf = random_proj_hist.cumsum(dim=-1)

        # Compute the Kolmogorov-Smirnov distance on random projections
        random_ks_dist = torch.max(torch.abs(random_proj_cdf - real_random_proj_cdf), dim=-1).values
        torch.save(random_ks_dist.clone().cpu(), '{}/random_ks_dist_{}__{}.pt'.format(config['save_path'], info['save_name'], seed))

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
            torch.save(mc_ess.clone().cpu() / N, '{}/mc_ess_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))

        # Print and save the acceptance
        if sampler.has_diagnostics('local_acceptance'):
            local_acceptance = sampler.get_diagnostics('local_acceptance')
            if isinstance(local_acceptance, torch.Tensor):
                local_acceptance = local_acceptance.mean()
            else:
                local_acceptance = torch.tensor(local_acceptance)
            print('local acceptance (mean) = {:.2f}%'.format(100 * float(local_acceptance)))
            torch.save(local_acceptance.clone().cpu(), '{}/acceptance_local_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))
        if sampler.has_diagnostics('global_acceptance'):
            global_acceptance = sampler.get_diagnostics('global_acceptance')
            if isinstance(global_acceptance, torch.Tensor):
                global_acceptance = global_acceptance.mean()
            else:
                global_acceptance = torch.tensor(global_acceptance)
            print('global acceptance (mean) = {:.2f}%'.format(100 * float(global_acceptance)))
            torch.save(global_acceptance.clone().cpu(), '{}/acceptance_global_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))

if __name__ == "__main__":
    # Libraries
    import argparse
    import yaml
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument('--seed', type=int)
    parser.add_argument('--save_samples', action=argparse.BooleanOptionalAction)
    parser.add_argument('--neutra_flow', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    # Freeze the seed
    set_seed(args.seed)
    # Load the config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Get the Pytorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Run the experiment
    main(config, device, args.seed, save_samples=args.save_samples, neutra_flow=args.neutra_flow)
