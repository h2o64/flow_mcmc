# Various routines when running experiments

# Libraries
import numpy as np
import torch
import pyro
import copy
from KDEpy import FFTKDE

# MCMC algorithms
from mcmc.mala import MALA
from mcmc.ess import ESS
from mcmc.hmc import HMC
from mcmc.pyro_mcmc import NUTS
from mcmc.pyro_mcmc import HMC as HMC_alt
from mcmc.rwmh import RWMH
from mcmc.imh import IndependentMetropolisHastings
from mcmc.isir import iSIR
from mcmc.classic_is import IS
from mcmc.global_local import Ex2MCMC, AdaptiveMCMC
from mcmc.neutra import NeuTra

# Freeze the seed accross various libraries
def set_seed(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.cuda.manual_seed_all(seed)
	pyro.set_rng_seed(seed)

# Apply a scheduler
def apply_scheduler(info_, dim):
	info = copy.deepcopy(info_)
	if 'scheduler' in info:
		for params_type in info['scheduler'].keys():
			if params_type not in info: info[params_type] = {}
			for param in info['scheduler'][params_type].keys():
				info[params_type][param] = info['scheduler'][params_type][param][dim]
	return info

# Create the MCMC sampler
def create_base_mcmc(name, params, dim, proposal, flow=None):
	sampler = None
	if name == 'mala':
		sampler = MALA(**params)
	elif name == 'ess':
		sampler = ESS()
	elif name == 'nuts':
		sampler = NUTS(**params)
	elif name == 'hmc':
		sampler = HMC(**params)
	elif name == 'hmc_pyro':
		sampler = HMC_alt(**params)
	elif name == 'rwmh':
		sampler = RWMH(**params)
	elif name == 'imh':
		sampler = IndependentMetropolisHastings(proposal=proposal, flow=None)
	elif name == 'adaptive_imh':
		sampler = IndependentMetropolisHastings(proposal=proposal, flow=flow)
	elif name == 'isir':
		sampler = iSIR(N=params['N'], proposal=proposal, flow=None)
	elif name == 'adaptive_isir':
		sampler = iSIR(N=params['N'], proposal=proposal, flow=flow)
	elif name == 'is':
		sampler = IS(N=params['N'], proposal=proposal, flow=None)
	elif name == 'adaptive_is':
		sampler = IS(N=params['N'], proposal=proposal, flow=flow)
	return sampler
def create_mcmc(conf_, dim, proposal, flow, use_reshape=False):
	# Run through the scheduler
	conf = apply_scheduler(conf_, dim)
	# Create the sampler
	sampler = None
	if conf['name'] in ['ex2mcmc','flex2mcmc','adaptive_mcmc']:
		# Make the local kernel
		local_mcmc = create_base_mcmc(conf['params']['local_sampler'], conf['local_sampler_params'],
			dim, proposal, flow=None)
		# Make the sampler
		if conf['name'] == 'ex2mcmc':
			sampler = Ex2MCMC(N=conf['params']['N'], proposal=proposal, local_mcmc=local_mcmc, n_local_mcmc=conf['params']['n_local_mcmc'], flow=None)
		elif conf['name'] == 'flex2mcmc':
			sampler = Ex2MCMC(N=conf['params']['N'], proposal=proposal, local_mcmc=local_mcmc, n_local_mcmc=conf['params']['n_local_mcmc'], flow=flow)
		else:
			sampler = AdaptiveMCMC(proposal=proposal, local_mcmc=local_mcmc, n_local_mcmc=conf['params']['n_local_mcmc'], flow=flow)
	else:
		sampler = create_base_mcmc(conf['name'], conf['params'], dim, proposal, flow)
	# Neutralize it if needed
	if conf['neutralize']:
		return NeuTra(inner_sampler=sampler, flow=flow, use_reshape=use_reshape)
	else:
		return sampler

# Algorithms using flows
algorithms_with_flow = set([
	'flex2mcmc',
	'adaptive_mcmc',
	'adaptive_imh',
	'adaptive_isir',
	'adaptive_is'
])

# Compute the wasserstein distance between two multivariate normal distributions
# Stolen from https://gist.github.com/Flunzmas/6e359b118b0730ab403753dcc2a447df
def wasserstein_distance_mvn(mean1, cov1, mean2, cov2):
	# calculate Tr((cov1 * cov2)^(1/2))
	sq_tr_cov = torch.trace(torch.linalg.cholesky(torch.matmul(cov1, cov2)))
	# plug the sqrt_trace_component into Tr(cov1 + cov2 - 2(cov1 * cov2)^(1/2))
	trace_term = torch.trace(cov1 + cov2) - 2.0 * sq_tr_cov  # scalar
	# |mean1 - mean2|^2
	diff = mean1 - mean2  # [n, 1]
	mean_term = torch.sum(torch.mul(diff, diff))  # scalar
	# put it together
	return (trace_term + mean_term).float()

# Compute the hellinger distance between two multivariate normal distributions
def helliger_distance_mvn(mean1, cov1, mean2, cov2):
	# Form (cov1 + cov2) / 2
	half_cov1_cov2 = (cov1 + cov2) / 2
	ret = torch.sqrt((torch.sqrt(torch.linalg.det(cov1)) * torch.sqrt(torch.linalg.det(cov2))) / torch.sqrt(torch.linalg.det(half_cov1_cov2)))
	ret *= torch.exp(-(1/8)*torch.matmul((mean1-mean2).t(), torch.linalg.solve(half_cov1_cov2, mean1-mean2)))
	return 1.0 - ret

# Compute random projections
def compute_random_projections(dim, device, n_random_projections=256):
    projs = torch.randn((n_random_projections, dim), device=device)
    projs /= torch.linalg.norm(projs, dim=-1)[...,None]
    return projs

# Compute the sliced total variation
def compute_total_variation(projs, samples1, samples2, n_samples=2048, weigths_sample1=None,
            weights_sample2=None, samples2_proj=None, samples2_kdes=None, samples2_mean=None, bw='scott'):
	# Compute the mean of samples2
	if samples2_mean is None:
		samples2_mean = torch.mean(samples2, dim=0)
	# Project the samples
	samples1_proj = torch.matmul(samples1 - samples2_mean, projs.T).T.cpu().numpy()
	if samples2_proj is None:
		samples2_proj = torch.matmul(samples2 - samples2_mean, projs.T).T.cpu().numpy()
	# Compute the TVs in 1D
	ret = torch.zeros((projs.shape[0],), device=samples1.device)
	samples2_kdes_ = []
	for i in range(projs.shape[0]):
		# Compute min/max
		min_x = np.minimum(samples1_proj[i].min(), samples2_proj[i].min()) - 1e-14
		max_x = np.maximum(samples1_proj[i].max(), samples2_proj[i].max()) + 1e-14
		# Compute the KDE
		samples1_kde = FFTKDE(bw=bw).fit(samples1_proj[i], weights=weigths_sample1)
		if not samples1_kde.bw > 0:
			samples1_kde = FFTKDE(bw='ISJ').fit(samples1_proj[i], weights=weigths_sample1)
		if samples2_kdes is None:
			samples2_kde = FFTKDE(bw=bw).fit(samples2_proj[i], weights=weights_sample2)
			if not samples2_kde.bw > 0:
				samples2_kde = FFTKDE(bw='ISJ').fit(samples2_proj[i], weights=weights_sample2)
			samples2_kdes_.append(samples2_kde)
		else:
			samples2_kde = samples2_kdes[i]
		# Compute the actual TV
		points = np.linspace(min_x, max_x, n_samples)
		ret[i] = 0.5 * (max_x - min_x) * np.abs(
			samples1_kde.evaluate(points) - samples2_kde.evaluate(points)
		).mean()
	if len(samples2_kdes_) > 0:
		return ret, samples2_proj, samples2_kdes_, samples2_mean
	else:
		return ret