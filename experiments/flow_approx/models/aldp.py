# ALANINE DIPEPTIDE
# Taken from
#   - https://github.com/lollcat/fab-torch/blob/master/fab/target_distributions/aldp.py
#   - https://github.com/lollcat/fab-torch/blob/master/experiments/aldp/train.py
#   - https://github.com/lollcat/fab-torch/blob/master/experiments/make_flow/make_aldp_model.py
import torch
from torch import nn
import numpy as np

import normflows as nf
import boltzgen as bg
import openmm as mm
from openmm import unit
from openmm import app
from openmmtools import testsystems
import mdtraj
import tempfile

from collections import OrderedDict

# Make the normalizing flow
def make_nf(target, seed, ndim, ind_circ_dih, **config):

    # Flow parameters
    flow_type = config['type']

    ncarts = target.coordinate_transform.transform.len_cart_inds
    permute_inv = target.coordinate_transform.transform.permute_inv.cpu().numpy()
    dih_ind_ = target.coordinate_transform.transform.ic_transform.dih_indices.cpu().numpy()
    std_dih = target.coordinate_transform.transform.ic_transform.std_dih.cpu()

    ind = np.arange(ndim)
    ind = np.concatenate([ind[:3 * ncarts - 6], -np.ones(6, dtype=np.int32), ind[3 * ncarts - 6:]])
    ind = ind[permute_inv]
    dih_ind = ind[dih_ind_]

    ind_circ = dih_ind[ind_circ_dih]
    bound_circ = np.pi / std_dih[ind_circ_dih]

    tail_bound = 5. * torch.ones(ndim)
    tail_bound[ind_circ] = bound_circ

    circ_shift = None if not 'circ_shift' in config else config['circ_shift']

    # Base distribution
    if config['base']['type'] == 'gauss':
        base = nf.distributions.DiagGaussian(ndim,
                                             trainable=config['base']['learn_mean_var'])
    elif config['base']['type'] == 'gauss-uni':
        base_scale = torch.ones(ndim)
        base_scale[ind_circ] = bound_circ * 2
        base = nf.distributions.UniformGaussian(ndim, ind_circ, scale=base_scale)
        base.shape = (ndim,)
    elif config['base']['type'] == 'resampled-gauss-uni':
        base_scale = torch.ones(ndim)
        base_scale[ind_circ] = bound_circ * 2
        base_ = nf.distributions.UniformGaussian(ndim, ind_circ, scale=base_scale)
        pf = nf.utils.nn.PeriodicFeaturesCat(ndim, ind_circ, np.pi / bound_circ)
        resnet = nf.nets.ResidualNet(ndim + len(ind_circ), 1, 
                                     config['base']['params']['a_hidden_units'],
                                     num_blocks=config['base']['params']['a_n_blocks'], 
                                     preprocessing=pf)
        a = torch.nn.Sequential(resnet, torch.nn.Sigmoid())
        base = lf.distributions.ResampledDistribution(base_, a, 
                                                      config['base']['params']['T'], 
                                                      config['base']['params']['eps'])
        base.shape = (ndim,)
    else:
        raise NotImplementedError('The base distribution ' + config['base']['type']
                                  + ' is not implemented')

    # Flow layers
    layers = []
    n_layers = config['blocks']

    for i in range(n_layers):
        if flow_type == 'rnvp':
            # Coupling layer
            hl = config['hidden_layers'] * [config['hidden_units']]
            scale_map = config['scale_map']
            scale = scale_map is not None
            if scale_map == 'tanh':
                output_fn = 'tanh'
                scale_map = 'exp'
            else:
                output_fn = None
            param_map = nf.nets.MLP([(ndim + 1) // 2] + hl + [(ndim // 2) * (2 if scale else 1)],
                                    init_zeros=config['init_zeros'], output_fn=output_fn)
            layers.append(nf.flows.AffineCouplingBlock(param_map, scale=scale,
                                                       scale_map=scale_map))
        elif flow_type == 'circular-ar-nsf':
            bl = config['blocks_per_layer']
            hu = config['hidden_units']
            nb = config['num_bins']
            ii = config['init_identity']
            dropout = config['dropout']
            layers.append(nf.flows.CircularAutoregressiveRationalQuadraticSpline(ndim,
                                                                                 bl, hu, ind_circ,
                                                                                 tail_bound=tail_bound, num_bins=nb,
                                                                                 permute_mask=True,
                                                                                 init_identity=ii,
                                                                                 dropout_probability=dropout))
        elif flow_type == 'circular-coup-nsf':
            bl = config['blocks_per_layer']
            hu = config['hidden_units']
            nb = config['num_bins']
            ii = config['init_identity']
            dropout = config['dropout']
            if i % 2 == 0:
                mask = nf.utils.masks.create_random_binary_mask(ndim, seed=seed + i)
            else:
                mask = 1 - mask
            layers.append(nf.flows.CircularCoupledRationalQuadraticSpline(ndim,
                                                                          bl, hu, ind_circ, tail_bound=tail_bound,
                                                                          num_bins=nb, init_identity=ii,
                                                                          dropout_probability=dropout, mask=mask))
        else:
            raise NotImplementedError('The flow type ' + flow_type + ' is not implemented.')

        if config['mixing'] == 'affine':
            layers.append(nf.flows.InvertibleAffine(ndim, use_lu=True))
        elif config['mixing'] == 'permute':
            layers.append(nf.flows.Permute(ndim))

        if config['actnorm']:
            layers.append(nf.flows.ActNorm(ndim))

        if i % 2 == 1 and i != n_layers - 1:
            if circ_shift == 'constant':
                layers.append(nf.flows.PeriodicShift(ind_circ, bound=bound_circ,
                                                     shift=bound_circ))
            elif circ_shift == 'random':
                gen = torch.Generator().manual_seed(seed + i)
                shift_scale = torch.rand([], generator=gen) + 0.5
                layers.append(nf.flows.PeriodicShift(ind_circ, bound=bound_circ,
                                                     shift=shift_scale * bound_circ))

        # SNF
        if 'snf' in config:
            if (i + 1) % config['snf']['every_n'] == 0:
                prop_scale = config['snf']['proposal_std'] * np.ones(ndim)
                steps = config['snf']['steps']
                proposal = nf.distributions.DiagGaussianProposal((ndim,), prop_scale)
                lam = (i + 1) / n_layers
                dist = nf.distributions.LinearInterpolation(target, base, lam)
                layers.append(nf.flows.MetropolisHastings(dist, proposal, steps))

    # Map input to periodic interval
    layers.append(nf.flows.PeriodicWrap(ind_circ, bound_circ))

    # normflows model
    flow = nf.NormalizingFlow(base, layers)

    return flow

# Make the ALDP target
class AldpBoltzmann(nn.Module):
    def __init__(self, device, data_path=None, temperature=1000, energy_cut=1.e+8,
                 energy_max=1.e+20, n_threads=4, transform='internal',
                 ind_circ_dih=[], shift_dih=False,
                 shift_dih_params={'hist_bins': 100},
                 default_std={'bond': 0.005, 'angle': 0.15, 'dih': 0.2},
                 env='vacuum'):
        """
        Boltzmann distribution of Alanine dipeptide
        :param data_path: Path to the trajectory file used to initialize the
            transformation, if None, a trajectory is generated
        :type data_path: String
        :param temperature: Temperature of the system
        :type temperature: Integer
        :param energy_cut: Value after which the energy is logarithmically scaled
        :type energy_cut: Float
        :param energy_max: Maximum energy allowed, higher energies are cut
        :type energy_max: Float
        :param n_threads: Number of threads used to evaluate the log
            probability for batches
        :type n_threads: Integer
        :param transform: Which transform to use, can be mixed or internal
        :type transform: String
        """
        super(AldpBoltzmann, self).__init__()

        # Define molecule parameters
        self.ndim = 66
        if transform == 'mixed':
            z_matrix = [
                (0, [1, 4, 6]),
                (1, [4, 6, 8]),
                (2, [1, 4, 0]),
                (3, [1, 4, 0]),
                (4, [6, 8, 14]),
                (5, [4, 6, 8]),
                (7, [6, 8, 4]),
                (11, [10, 8, 6]),
                (12, [10, 8, 11]),
                (13, [10, 8, 11]),
                (15, [14, 8, 16]),
                (16, [14, 8, 6]),
                (17, [16, 14, 15]),
                (18, [16, 14, 8]),
                (19, [18, 16, 14]),
                (20, [18, 16, 19]),
                (21, [18, 16, 19])
            ]
            cart_indices = [6, 8, 9, 10, 14]
        elif transform == 'internal':
            z_matrix = [
                (0, [1, 4, 6]),
                (1, [4, 6, 8]),
                (2, [1, 4, 0]),
                (3, [1, 4, 0]),
                (4, [6, 8, 14]),
                (5, [4, 6, 8]),
                (7, [6, 8, 4]),
                (9, [8, 6, 4]),
                (10, [8, 6, 4]),
                (11, [10, 8, 6]),
                (12, [10, 8, 11]),
                (13, [10, 8, 11]),
                (15, [14, 8, 16]),
                (16, [14, 8, 6]),
                (17, [16, 14, 15]),
                (18, [16, 14, 8]),
                (19, [18, 16, 14]),
                (20, [18, 16, 19]),
                (21, [18, 16, 19])
            ]
            cart_indices = [8, 6, 14]

        # System setup
        if env == 'vacuum':
            system = testsystems.AlanineDipeptideVacuum(constraints=None)
        elif env == 'implicit':
            system = testsystems.AlanineDipeptideImplicit(constraints=None)
        else:
            raise NotImplementedError('This environment is not implemented.')
        if torch.cuda.is_available():
            platform_name = 'CUDA'
        else:
            platform_name = 'Reference'
        sim = app.Simulation(system.topology, system.system,
                             mm.LangevinIntegrator(temperature * unit.kelvin,
                                                   1. / unit.picosecond,
                                                   1. * unit.femtosecond),
                             mm.Platform.getPlatformByName(platform_name))

        # Generate trajectory for coordinate transform if no data path is specified
        if data_path is None:
            sim = app.Simulation(system.topology, system.system,
                                mm.LangevinIntegrator(temperature * unit.kelvin,
                                                      1.0 / unit.picosecond, 1.0 * unit.femtosecond),
                                platform=mm.Platform.getPlatformByName(platform_name))
            sim.context.setPositions(system.positions)
            sim.minimizeEnergy()
            state = sim.context.getState(getPositions=True)
            position = state.getPositions(True).value_in_unit(unit.nanometer)
            tmp_dir = tempfile.gettempdir()
            data_path = tmp_dir + '/aldp.pt'
            torch.save(torch.tensor(position.reshape(1, 66).astype(np.float64)), data_path)
        if data_path[-2:] == 'h5':
            # Load data for transform
            traj = mdtraj.load(data_path)
            traj.center_coordinates()

            # superpose on the backbone
            ind = traj.top.select("backbone")
            traj.superpose(traj, 0, atom_indices=ind, ref_atom_indices=ind)

            # Gather the training data into a pytorch Tensor with the right shape
            transform_data = traj.xyz
            n_atoms = transform_data.shape[1]
            self.n_dim = n_atoms * 3
            transform_data_npy = transform_data.reshape(-1, self.n_dim)
            transform_data = torch.from_numpy(transform_data_npy.astype("float64"))
        elif data_path[-2:] == 'pt':
            transform_data = torch.load(data_path)
        else:
            raise NotImplementedError('Loading data or this format is not implemented.')

        # Set distribution
        self.coordinate_transform = bg.flows.CoordinateTransform(transform_data,
                                        self.ndim, z_matrix, cart_indices, mode=transform,
                                        ind_circ_dih=ind_circ_dih, shift_dih=shift_dih,
                                        shift_dih_params=shift_dih_params,
                                        default_std=default_std)

        if n_threads is None or n_threads > 1:
            self.p = bg.distributions.TransformedBoltzmannParallel(system,
                            temperature, energy_cut=energy_cut, energy_max=energy_max,
                            transform=self.coordinate_transform, n_threads=n_threads).to(device)
        else:
            self.p = bg.distributions.TransformedBoltzmann(sim.context,
                            temperature, energy_cut=energy_cut, energy_max=energy_max,
                            transform=self.coordinate_transform).to(device)

    def log_prob(self, x: torch.tensor):
        return self.p.log_prob(x)

# Get phi and psi
def get_phi_psi(z, target):
    # Transform z into x and get the log_prob
    x, _ = target.coordinate_transform(z)
    x = x.detach().cpu().numpy()
    # Make the trajectory
    aldp = testsystems.AlanineDipeptideVacuum(constraints=None)
    topology = mdtraj.Topology.from_openmm(aldp.topology)
    traj = mdtraj.Trajectory(x.reshape(-1, 22, 3), topology)
    # Compute the angles
    psi = mdtraj.compute_psi(traj)[1].reshape(-1)
    phi = mdtraj.compute_phi(traj)[1].reshape(-1)
    is_nan = np.logical_or(np.isnan(psi), np.isnan(phi))
    is_absurd = np.logical_or(
        np.logical_or(psi < -torch.pi, psi > torch.pi),
        np.logical_or(phi < -torch.pi, phi > torch.pi)
    ) 
    not_nan_or_absurd = np.logical_not(np.logical_or(is_nan, is_absurd))
    psi = psi[not_nan_or_absurd]
    phi = phi[not_nan_or_absurd]
    # Return both
    return phi, psi

# Debug
if __name__ == "__main__":

    # Import wrappers
    from flow.normflows_wrappers import BaseDistributionWrapper, WrappedNormFlowModel

    # Set the float type
    torch.set_default_dtype(torch.float64)

    # Get the Pytorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Make AldpBoltzmann
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
    dim = 60

    # Make the flow
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

    # Load custom weights
    flow_weights = torch.load('experiments/flow_approx/models/aldp/flow_aldp.pt', map_location=device)
    flow_weights = OrderedDict([(k.replace('_nf_model.',''),v) for k,v in flow_weights.items()])
    flow.load_state_dict(flow_weights)

    # Make the proposal
    proposal = BaseDistributionWrapper(flow.q0)

    # Wrap the flow
    flow = WrappedNormFlowModel(flow)

    # Make FlEx2MCMC
    from mcmc.global_local import Ex2MCMC
    from mcmc.mala import MALA

    # Make sampler
    sampler = Ex2MCMC(
        N=64,
        proposal=proposal,
        local_mcmc=MALA(step_size=1e-4, target_acceptance=0.75),
        n_local_mcmc=16,
        flow=flow
    )

    # FlEx2MCMC
    samples = sampler.sample(
        x_s_t_0=flow.forward(proposal.sample((64,)))[0],
        warmup_steps=250,
        n_steps=250,
        target=target.log_prob,
        verbose=True
    )
    samples = samples.view((-1, dim))
    log_prob = target.log_prob(samples).mean().detach().cpu().numpy()

    print('acceptance_globale = ', sampler.get_diagnostics('global_acceptance').mean())
    print('acceptance_locale = ', sampler.get_diagnostics('local_acceptance').mean())
    print('log_prob = ', log_prob)
