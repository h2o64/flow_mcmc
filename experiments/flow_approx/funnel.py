# Neal's Funnel
# Details :
# - Neal's funnel with various imperfect flows on different samplers

# Libraries
import torch
import math
import time
import pyro.ops.stats as stats
from experiments.utils import set_seed, create_mcmc, algorithms_with_flow
from distribution.funnel import Funnel


# A parametric flow adapted to a Neal's funnel
class FunnelFlow:

    def __init__(self, dim=2, alpha=1.0, a=3.0, b=1.0):
        self.dim = dim
        self.a = a
        self.b = b
        self.log_sqrt_a = 0.5 * math.log(a)
        self.sqrt_alpha = math.sqrt(alpha)
        self.log_alpha = math.log(alpha)
        self.sqrt_inv_alpha = math.sqrt(1.0 / alpha)
        self.sqrt_a_over_alpha = math.sqrt(a / alpha)

    def forward(self, z):
        x1 = torch.unsqueeze(self.sqrt_a_over_alpha * z[:,0], dim=-1)
        x_ = self.sqrt_inv_alpha * torch.exp(0.5 * self.b * x1) * z[:,1:]
        x = torch.concat([x1,x_], dim=-1).to(z.device)
        log_jac = self.log_sqrt_a  - (self.dim/2) * self.log_alpha + 0.5 * (self.dim-1) * self.b * self.sqrt_a_over_alpha * z[:,0]
        return x, log_jac

    def inverse(self, x):
        z1 = torch.unsqueeze(x[:,0] / self.sqrt_a_over_alpha, dim=-1)
        z_ = self.sqrt_alpha  * torch.exp(-0.5 * self.b * torch.unsqueeze(x[:,0], dim=-1)) * x[:,1:]
        z = torch.concat([z1,z_], dim=-1).to(x.device)
        log_jac = (-0.5 * self.b * (self.dim-1) * x[:,0]) - self.log_sqrt_a + (self.dim/2) * self.log_alpha
        return z, log_jac

    def __call__(self, x):
        return self.forward(x)


# Main experiment
def main(config, device, seed, compute_rhat):

    # Browse all the dimensions
    if 'debug' in config:
        dims = [32, 64, 128]
    else:
        dims = [16, 32, 64, 128, 256]
    for dim in dims:

        # Number of steps
        if dim == 256:
            n_steps = 700
        elif dim == 128:
            n_steps = 650
        elif dim == 64:
            n_steps = 600
        elif dim == 32:
            n_steps = 550
        else:
            n_steps = 500

        # Make the target and proposal
        target = Funnel(dim=dim, device=device)
        proposal = torch.distributions.MultivariateNormal(
            loc=torch.zeros((dim,)).to(device),
            covariance_matrix=torch.eye(dim).to(device)
        )

        # Define the random projections
        n_random_projections = 128
        random_projs = torch.randn((n_random_projections, dim))
        random_projs /= torch.linalg.norm(random_projs, axis=-1)[...,None]

        # Compute the true histogram
        n_bins = 1024
        n_real_samples = 3 * n_steps * config['batch_size']
        samples_real = target.sample(sample_shape=(n_real_samples,)).cpu()
        samples_real_random_proj = torch.matmul(samples_real, random_projs[...,None])[...,0]
        real_random_proj_min_x, real_random_proj_max_x = samples_real_random_proj.min(dim=-1), samples_real_random_proj.max(dim=-1)
        real_random_proj_hist = torch.zeros((n_random_projections, n_bins))
        for i in range(n_random_projections):
            real_random_proj_hist[i] = torch.histc(samples_real_random_proj[i], bins=n_bins,
                min=float(real_random_proj_min_x.values[i]), max=float(real_random_proj_max_x.values[i]))
        real_random_proj_hist /= real_random_proj_hist.sum(dim=-1)[...,None]
        real_random_proj_cdf = real_random_proj_hist.cumsum(dim=-1)

        # Change the type of flow
        if 'debug' in config:
            len_beta = 5
        else:
            len_beta = 12
        beta_space = torch.logspace(-1.0, 1.0, len_beta)
        for flow_type in ['perfect'] + ['beta_{}'.format(i) for i in range(len_beta)]:

            # Make the flow
            if flow_type == 'perfect':
                flow = FunnelFlow(dim=dim)
            else:
                beta = float(beta_space[int(flow_type.split('_')[-1])])
                flow = FunnelFlow(
                    dim=dim,
                    alpha=float(beta * 1.0),
                    a=float(beta * 3.0),
                    b=1.0,
                )

            # Build the initial MCMC state
            start_orig = proposal.sample((config['batch_size'],))
            start_orig = flow.forward(start_orig)[0].detach()

            # Browse all methods
            for method_name, info in config['methods'].items():

                # Reset the seed
                set_seed(seed)

                # Show debug info
                print("============ {} (dim = {} | flow = {}) =============".format(method_name, dim, flow_type))

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
                    torch.save(rhat_idx.clone().cpu(), '{}/rhat_idx_{}_dim_{}_flow_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, flow_type, seed))
                    torch.save(rhat.clone().cpu(), '{}/rhat_{}_dim_{}_flow_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, flow_type, seed))
                    continue

                # Save elapsed time
                if not 'debug' in config:
                    torch.save(torch.tensor(elapsed), '{}/elapsed_time_{}_dim_{}_flow_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, flow_type, seed))

                # Compute and save ESS
                if info['name'] != 'adaptive_is' and not 'debug' in config:
                    ess = stats.effective_sample_size(samples, chain_dim=0, sample_dim=1).cpu()
                    ess_per_sec = ess / elapsed
                    torch.save(ess.clone().cpu(), '{}/ess_{}_dim_{}_flow_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, flow_type, seed))
                    torch.save(ess_per_sec.clone().cpu(), '{}/ess_per_sec_{}_dim_{}_flow_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, flow_type, seed))

                # Compute the log-likelihoods
                samples_shape = samples.shape
                samples = samples.view((-1,dim))
                if not sampler.has_diagnostics('weights'):
                    log_prob = target.log_prob(samples).mean()
                else:
                    weights = sampler.get_diagnostics('weights').clone()
                    log_prob = (weights[:,None] * target.log_prob(samples)).sum() / weights.sum()
                samples = samples.view(samples_shape)
                torch.save(log_prob.clone().cpu(), '{}/log_prob_{}_dim_{}_flow_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, flow_type, seed))

                # Compute the empirical histogram on random projections
                if sampler.has_diagnostics('weights'):
                    weights = sampler.get_diagnostics('weights').cpu()
                else:
                    weights = None
                samples = samples.cpu()
                samples_random_proj = torch.einsum('nbd,kd->nbk', samples, random_projs)
                random_proj_hist = torch.zeros((n_random_projections, n_bins))
                random_ks_dist = torch.zeros((samples_shape[1], n_random_projections))
                for batch_id in range(samples_shape[1]):
                    # Compute the histograms
                    for proj_id in range(n_random_projections):
                        random_proj_hist[proj_id] = torch.histogram(samples_random_proj[:,batch_id,proj_id], bins=n_bins,
                            range=(float(real_random_proj_min_x.values[proj_id]), float(real_random_proj_max_x.values[proj_id])), weight=weights).hist
                    random_proj_hist /= random_proj_hist.sum(dim=-1)[...,None]
                    # Compute the cumulative distribution function
                    random_proj_cdf = random_proj_hist.cumsum(dim=-1)
                    # Compute the Kolmogorov-Smirnov distance on random projections
                    random_ks_dist[batch_id] = torch.max(torch.abs(random_proj_cdf - real_random_proj_cdf), dim=-1).values
                torch.save(random_ks_dist.clone().cpu(), '{}/random_ks_dist_{}_dim_{}_flow_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, flow_type, seed))

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
                    torch.save(mc_ess.clone().cpu() / N, '{}/mc_ess_{}_dim_{}_flow_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, flow_type, seed))

                # Print and save the acceptance
                if not 'debug' in config:
                    if sampler.has_diagnostics('local_acceptance'):
                        local_acceptance = sampler.get_diagnostics('local_acceptance')
                        if isinstance(local_acceptance, torch.Tensor):
                            local_acceptance = local_acceptance.mean()
                        else:
                            local_acceptance = torch.tensor(local_acceptance)
                        print('local acceptance (mean) = {:.2f}%'.format(100 * float(local_acceptance)))
                        torch.save(local_acceptance.clone().cpu(), '{}/acceptance_local_{}_dim_{}_flow_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, flow_type, seed))
                    if sampler.has_diagnostics('global_acceptance'):
                        global_acceptance = sampler.get_diagnostics('global_acceptance')
                        if isinstance(global_acceptance, torch.Tensor):
                            global_acceptance = global_acceptance.mean()
                        else:
                            global_acceptance = torch.tensor(global_acceptance)
                        print('global acceptance (mean) = {:.2f}%'.format(100 * float(global_acceptance)))
                        torch.save(global_acceptance.clone().cpu(), '{}/acceptance_global_{}_dim_{}_flow_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, flow_type, seed))

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
