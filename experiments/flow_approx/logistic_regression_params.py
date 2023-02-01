# Logistic regression

# Libraries
import torch
import time
from experiments.utils import set_seed, create_mcmc, algorithms_with_flow
from flow.realnvp import RealNVP
from models.logistic_regression import HorseshoeLogisticRegression, load_german_credit, make_iaf_flow
from sklearn.model_selection import train_test_split
from tqdm import trange

# Main experiment
def main(config, device, seed, neutra_flow=False):

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
        covariance_matrix=torch.eye(dim).to(device) * 1e-2,
        validate_args=False
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
        try:
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
        except:
            continue

        # Compute the energy of the chains
        samples = samples.reshape((-1, dim))
        if not sampler.has_diagnostics('weights'):
            energy = -target_test.log_prob(samples).mean()
        else:
            weights = sampler.get_diagnostics('weights').clone()
            energy = -(weights[:,None] * target_test.log_prob(samples)).sum() / weights.sum()
        torch.save(energy.clone().cpu(), '{}/energy_{}_{}.pt'.format(config['save_path'], info['save_name'], seed))

if __name__ == "__main__":
    # Libraries
    import argparse
    import yaml
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument('--seed', type=int)
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
    main(config, device, args.seed, neutra_flow=args.neutra_flow)
