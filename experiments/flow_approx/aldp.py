# Alanine dipeptide
# Sample from the distribution of alanine dipeptide in an implicit solvent,
# which is a molecule with 22 atoms

# Libraries
import torch
import time
import pyro.ops.stats as stats
from experiments.utils import set_seed, create_mcmc, algorithms_with_flow
from models.aldp import make_nf, AldpBoltzmann, get_phi_psi
from flow.normflows_wrappers import BaseDistributionWrapper, WrappedNormFlowModel
from collections import OrderedDict

# Make the target distribution
def make_target(device):
    ind_circ_dih = [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16]
    target = AldpBoltzmann(
        device=device,
        temperature=300,
        energy_cut=1e8,
        energy_max=1e20,
        n_threads=None,
        transform='internal',
        shift_dih=False,
        env='implicit',
        ind_circ_dih=ind_circ_dih,
        data_path='experiments/flow_approx/models/aldp/position_min_energy.pt'
    )
    target.to(device)
    return target

# Make the flow
def make_flow(target, device, dim=60):
    ind_circ_dih = [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16]
    flow = make_nf(
        target=target,
        seed=0,
        ndim=dim,
        ind_circ_dih=ind_circ_dih,
        type='circular-coup-nsf',
        base={
            'type' : 'gauss-uni',
            'params' : None
        },
        blocks=12,
        actnorm=False,
        mixing=None,
        circ_shift='random',
        blocks_per_layer=1,
        hidden_units=256,
        num_bins=8,
        init_identity=True,
        dropout=0.
    )
    flow.to(device)
    return flow

# Load custom weights
def load_custom_weights(flow, device, weight_filepath='experiments/flow_approx/models/aldp/flow_aldp.pt'):
    flow_weights = torch.load(weight_filepath, map_location=device)
    flow_weights = OrderedDict([(k.replace('_nf_model.',''),v) for k,v in flow_weights.items()])
    flow.load_state_dict(flow_weights)
    return flow

# Make the proposal
def make_proposal(flow):
    return BaseDistributionWrapper(flow.q0)

# Main experiment
def main(config, device, seed, save_samples=False):

    # Set the float type
    torch.set_default_dtype(torch.float64)

    # Set the dimension
    dim = 60

    # Make the target
    target = make_target(device)

    # Make the flow
    flow = make_flow(target, device)
    flow = load_custom_weights(flow, device)

    # Make the proposal
    proposal = make_proposal(flow)

    # Wrap the flow
    flow = WrappedNormFlowModel(flow)

    # Build the initial MCMC state
    start_orig = torch.load('experiments/flow_approx/models/aldp/init_points_aldp.pt', map_location=device)

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
        sampler = create_mcmc(info, dim, proposal, flow_)

        # Run the sampler
        s = time.time()
        samples = sampler.sample(
            x_s_t_0=start_orig.clone().to(device),
            n_steps=info['n_steps'],
            target=target.log_prob,
            warmup_steps=info['warmup_steps'],
            verbose=True
        ).detach()
        e = time.time()
        elapsed = e - s
        print(f"Elapsed: {elapsed:.2f} s")

        # Save elapsed time
        torch.save(torch.tensor(elapsed), '{}/elapsed_time_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))

        # Compute and save ESS
        if info['name'] != 'adaptive_is':
            ess = stats.effective_sample_size(samples, chain_dim=0, sample_dim=1).cpu()
            ess_per_sec = ess / elapsed
            torch.save(ess.clone().cpu(), '{}/ess_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))
            torch.save(ess_per_sec.clone().cpu(), '{}/ess_per_sec_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))

        # Compute the energy of the chains
        samples_shape = samples.shape
        samples = samples.view((-1, dim))
        if not sampler.has_diagnostics('weights'):
            energy = -target.log_prob(samples).mean()
        else:
            weights = sampler.get_diagnostics('weights').clone()
            energy = -(weights[:,None] * target.log_prob(samples)).sum() / weights.sum()
        torch.save(energy.clone().cpu(), '{}/energy_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))

        # Save the samples
        if save_samples:
            # Save the samples
            torch.save(samples.clone().cpu(), '{}/samples_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))
            # Compute phi and psi
            phi, psi = get_phi_psi(samples.double(), target)
            # Reshape phi and psi
            phi, psi = phi.reshape((*samples_shape[:-1], -1)), psi.reshape((*samples_shape[:-1], -1))
            phi_psi = torch.stack([torch.from_numpy(phi), torch.from_numpy(psi)], dim=-1)
            # Save phi and psi
            torch.save(phi_psi.clone().cpu(), '{}/phi_psi_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))
            # Save the weights
            if sampler.has_diagnostics('weights'):
                torch.save(sampler.get_diagnostics('weights').clone().cpu(), '{}/weights_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))

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
    args = parser.parse_args()
    # Freeze the seed
    set_seed(args.seed)
    # Load the config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Get the Pytorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Run the experiment
    main(config, device, args.seed, save_samples=args.save_samples)
