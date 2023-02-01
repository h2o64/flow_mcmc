# Gaussian 3 flows experiment
# Details :
# - Imperfect flow with a push-forward interpolated between 3 gaussians
#   of varying variance

# Libraries
import torch
import time
import pyro.ops.stats as stats
from experiments.utils import set_seed, create_mcmc, algorithms_with_flow
from experiments.utils import compute_random_projections, compute_total_variation

# Make a flow with a push-forward between 3 gaussians
class ThreeGaussiansFlow:
    
    def __init__(self, t, cov1, cov2, cov_target):
        # Compute necessities
        if t == 0.5:
            self.chol_forward = torch.linalg.cholesky(cov_target)
        elif t < 0.5:
            self.chol_a = torch.linalg.cholesky(cov1)
            self.chol_b = torch.linalg.cholesky(cov_target)
            self.chol_forward = (1.0 - 2.0 * t) * self.chol_a + 2.0 * t * self.chol_b
        else:
            self.chol_a = torch.linalg.cholesky(cov2)
            self.chol_b = torch.linalg.cholesky(cov_target)
            self.chol_forward = (2.0 * t - 1.0) * self.chol_a + 2.0 * (1.0 - t) * self.chol_b
        self.log_jac = torch.linalg.slogdet(self.chol_forward).logabsdet
        self.chol_inverse = torch.linalg.inv(self.chol_forward)
    
    def forward(self, z):
        x = torch.matmul(self.chol_forward, z[...,None]).squeeze(-1)
        log_jac = self.log_jac * torch.ones(z.shape[:-1]).to(z.device)
        return x, log_jac

    def inverse(self, x):
        z = torch.matmul(self.chol_inverse, x[...,None]).squeeze(-1)
        log_jac = -self.log_jac * torch.ones(x.shape[:-1]).to(x.device)
        return z, log_jac

    def __call__(self, z):
        return self.forward(z)

# Make a rotation matrix on the first two dimensions
def make_2d_rotation_matrix(angle, dim):
    angle_t = torch.tensor(angle)
    ret = torch.eye(dim)
    ret[0,0] = torch.cos(angle_t)
    ret[0,-1] = -torch.sin(angle_t)
    ret[-1,-1] = ret[0,0]
    ret[-1,0] = -ret[0,-1]
    return ret

# Main experiment
def main(config, device, seed, compute_rhat):

    # Browse all the dimensions
    if 'debug' in config:
        dims = [32, 64, 128]
    else:
        dims = [16, 32, 64, 128, 256]
    for dim in dims:

        # Make the target and proposal
        r = make_2d_rotation_matrix(torch.pi/4, dim)
        target = torch.distributions.MultivariateNormal(
            loc=torch.zeros((dim,)).to(device),
            covariance_matrix=torch.matmul(r, torch.matmul(torch.diag(torch.logspace(-1, 1, dim)), r.T)).to(device),
        )
        proposal = torch.distributions.MultivariateNormal(
            loc=torch.zeros((dim,)).to(device),
            covariance_matrix=torch.eye(dim).to(device)
        )

        # Number of steps
        if dim == 256:
            n_steps = 1500
        elif dim == 128:
            n_steps = 1400
        elif dim == 64:
            n_steps = 1300
        elif dim == 32:
            n_steps = 1200
        else:
            n_steps = 1100

        # Sample the target
        samples_target = target.sample(sample_shape=(10 * n_steps,))

        # Number of random projections for the TV
        n_random_projections = 128
        projs = compute_random_projections(dim, device, n_random_projections=n_random_projections)
        samples2_proj = None
        samples2_kdes = None
        samples2_mean = None

        # Change the type of flow
        if 'debug' in config:
            len_t = 5
        else:
            len_t = 10
        t_space = torch.linspace(0.0, 1.0, len_t)
        for flow_type in ['perfect'] + ['t_{}'.format(i) for i in range(len_t)]:

            # Build the flow
            cov1 = 1e-1 * torch.eye(dim).to(device)
            cov2 = 1e1 * torch.eye(dim).to(device)
            if flow_type == 'perfect':
                flow = ThreeGaussiansFlow(t=0.5, cov1=cov1, cov2=cov2, cov_target=target.covariance_matrix)
            else:
                flow = ThreeGaussiansFlow(t=float(t_space[int(flow_type.split('_')[-1])]), cov1=cov1, cov2=cov2, cov_target=target.covariance_matrix)

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

                # Estimate the mean and covariance
                samples_shape = samples.shape
                kl_forwards = torch.zeros((samples_shape[1],))
                kl_backwards = torch.zeros((samples_shape[1],))
                for batch_id in range(samples_shape[1]):
                    if sampler.has_diagnostics('weights'):
                        samples_mean = (sampler.get_diagnostics('weights')[:,None] * samples[:,batch_id]).sum(dim=0).to(device)
                        samples_mean /= sampler.get_diagnostics('weights').sum()
                        samples_cov = samples[:,batch_id].T.cov(aweights=sampler.get_diagnostics('weights')).to(device)
                    else:
                        samples_mean = samples[:,batch_id].mean(dim=0).to(device)
                        samples_cov = samples[:,batch_id].T.cov().to(device)
                    try:
                        # Estimate the distribution
                        samples_dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=samples_mean, covariance_matrix=samples_cov)
                        # Compute the KL between the real target and the estimated one
                        kl_forwards[batch_id] = torch.distributions.kl.kl_divergence(target, samples_dist)
                        kl_backwards[batch_id] = torch.distributions.kl.kl_divergence(samples_dist, target)
                    except:
                        kl_forwards[batch_id], kl_backwards[batch_id] = float('inf'), float('inf')
                torch.save(kl_forwards.clone().cpu(), '{}/kl_forward_{}_dim_{}_flow_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, flow_type, seed))
                torch.save(kl_backwards.clone().cpu(), '{}/kl_backward_{}_dim_{}_flow_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, flow_type, seed))

                # Estimate the sliced total variation
                sliced_tvs = torch.zeros((samples_shape[1], n_random_projections))
                for batch_id in range(samples_shape[1]):
                    if samples2_proj is None:
                        sliced_tvs[batch_id], samples2_proj, samples2_kdes, samples2_mean = compute_total_variation(projs, samples[:,batch_id], samples_target)
                    else:
                        if sampler.has_diagnostics('weights'):
                            weights = sampler.get_diagnostics('weights').cpu().numpy()
                            bw = 'ISJ'
                        else:
                            weights = None
                            bw = 'scott'
                        try:
                            sliced_tvs[batch_id] = compute_total_variation(projs, samples[:,batch_id], samples_target, weigths_sample1=weights, samples2_proj=samples2_proj,
                                samples2_kdes=samples2_kdes, samples2_mean=samples2_mean, bw=bw)
                        except:
                            sliced_tvs[batch_id] = float('inf')
                torch.save(sliced_tvs.clone().cpu(), '{}/sliced_tv_{}_dim_{}_flow_{}_{}.pt'.format(config['save_path'], info['save_name'], dim, flow_type, seed))

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
    import warnings
    # Remove warnings
    warnings.filterwarnings("ignore")
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
