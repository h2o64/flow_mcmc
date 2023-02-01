# Flow MCMC

This is the official repository for the preprint "On Sampling with Approximate Transport Maps" (TODO : Arxiv link) in PyTorch. 

## Installation

Install dependencies with `pip install -r requirements.txt`. If you want to rerun the Alanine Dipeptide experiments, use conda and install the following packages `conda install -c conda-forge openmm openmmtools mdtraj` and also `pip install git+https://github.com/VincentStimper/boltzmann-generators.git`

# MCMC Samplers

Flow MCMC provide many popular MCMC samplers under a common API

* Metropolis-Adjusted Langevin Algorithm (MALA) with `mcmc.mala.MALA`
* Hamiltonian Monte Carlo (HMC) [(Neal et al., 2011)](https://arxiv.org/abs/1206.1901) with `mcmc.hmc.HMC`
* Random Walk Metropolis Hastings (RWMH) with `mcmc.rwhm.RWHM`
* Elliptical Slice Sampling (ESS) [(Murray et al., 2010)](https://proceedings.mlr.press/v9/murray10a.html) with `mcmc.ess.ESS`
* Independent Metropolis-Hastings (IMH) with `mcmc.imh.IndependentMetropolisHastings`
* Iterated Sampling Importance Resampling (i-SIR) [(Andrieu et al., 2010)](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2009.00736.x) with `mcmc.isir.iSIR`
* Pyro wrappers for HMC and NUTS [(Bingham et al., 2019)](https://jmlr.org/papers/v20/18-403.html) with `mcmc.pyro_mcmc.HMC` and `mcmc.pyro_mcmc.NUTS`

Sampling works by calling

```python
sampler.sample(x_s_t_0, n_steps, target, temp=1.0, warmup_steps=0, verbose=False)
```

Where

* `x_s_t_0` (tensor of shape `(batch_size, dim)` ) is the first sample of the chain
* `n_steps` (int) is the length of the chain
* `target` (callable) is the log-likelihood of the target distribution (it must support batched inputs)
* `temp` (float) is the temperature factor of the log-likelihood
* `warmup_steps` (int) is the number of burn-in steps (they will be wasted)
* `verbose` (bool) displays a progress bar during sampling

Note that unlike many MCMC samplers in PyTorch, all the MCMC samplers support sampling multiple chains in **parallel**. Each sampler collects diagnostics (acceptance rates, ...) which can be collected using `sampler.get_diagnostics(diag_name)`.

You can also use normalizing flows to enhance your sampler in two ways

* Using flow-MCMC algorithms (i.e., using the flow as global proposals)
  * In global samplers i-SIR or IMH by specifying the `flow` parameter in the constructor
  * In a global/local scheme see `mcmc.global_local.AdaptiveMCMC` [(Gabrie et al. , 2022) ](https://www.pnas.org/doi/abs/10.1073/pnas.2109420119) or `mcmc.global_local.Ex2MCMC` [(Samsonov et al. 2022)](https://arxiv.org/abs/2111.02702)
* Using neutra-MCMC algorithms (i.e., using the flow as a reparametrization map) [(Parno & Marzouk, 2018,](https://epubs.siam.org/doi/10.1137/17M1134640) [Hoffman et al., 2019)](https://arxiv.org/abs/1903.03704) by wrapping the sampler with `mcmc.neutra.NeuTra(inner_sampler, flow)`

Importance Sampling is also available at `mcmc.classic_is.IS` which the same API as MCMCs (`batch_size` and `n_steps` are ignored) and can be enhanced by a flow.

We also provide a way to perform adaptive learning of normalizing flows with `mcmc.learn_flow.LearnMCMC` [(Gabrie et al. , 2022) ](https://www.pnas.org/doi/abs/10.1073/pnas.2109420119).

We provide an implementation of RealNVP [(Dinh et al., 2016)](https://arxiv.org/abs/1605.08803) based on [marylou-gabrie/adapt-flow-ergo](https://github.com/marylou-gabrie/adapt-flow-ergo)'s implementation as well as a wrapper to flows from [VincentStimper/normalizing-flows](https://github.com/VincentStimper/normalizing-flows).

## "On Sampling with Approximate Transport Maps"

Here we explain how to rerun the experiments presented in the paper. **Note that the output paths are defined on top of the configuration files in `configs/flow_approx/`**.

### Synthetic case studies

All the experiments from the synthetic case studies can be rerun using the following commands

```bash
python experiments/flow_approx/gaussians_three_flows.py configs/flow_approx/gaussians_three_flows.yaml --seed {INSERT_SEED}
python experiments/flow_approx/funnel.py configs/flow_approx/funnel.yaml --seed {INSERT_SEED}
python experiments/flow_approx/gaussian_mixture.py configs/flow_approx/gaussians_mixture.yaml --seed {INSERT_SEED}
python experiments/flow_approx/banana.py {OUTPUT_PATH}/backward_dim{DIMENSION}.pkl --loss_type backward_kl --dim {DIMENSION} --seed {SEED}
```

The hyper-parameter grid search can be rerun by using the `*_debug.yaml` configs. The flows for the mixture of Gaussians can be re-trained using 

```bash
python experiments/flow_approx/gaussian_mixture_flow.py --dim {DIMENSION} --checkpoint_path {SAVE_PATH}/dim_{DIMENSION}/
```

### Benchmarks on real tasks

#### Alanine Dipeptide

The flow (also available in `experiments/flow_approx/models/aldp/flow_aldp.pt`) can be retrained using the procedure described in [lollcat/fab-torch](https://github.com/lollcat/fab-torch) [(Midgley et al., 2022)](https://arxiv.org/abs/2208.01893). Sampling can be performed using 

```bash
python experiments/flow_approx/aldp.py configs/flow_approx/aldp.yaml --seed {SEED} --save_samples
```

The data used for the ground truth are available on [authors' Zenodo](https://zenodo.org/record/6993124).

#### Logistic Regression

The flow for the logistic regression experiment can be obtained by running

```bash
python experiments/flow_approx/logistic_regression_flow.py --save_path {OUTPUT_PATH} --neutra_flow
```

and sampling can be done with

```bash
python experiments/flow_approx/logistic_regression.py configs/flow_approx/logistic_regression.yaml --seed {SEED} --neutra_flow
```

Note that you will need ground truth samples obtained using NUTS by running

```bash
python experiments/flow_approx/logistic_regression_gt.py --save_path {OUTPUT_PATH}
```

#### Phi Four

The flows for the Phi Four experiment can be obtained by running

```bash
python experiments/flow_approx/phi_four_parameters.py configs/flow_approx/phi_four_parameters/global_{DIMENSION}.yaml configs/flow_approx/phi_four_parameters/best_flows_{DIMENSION}.yaml --mala_sampler 
```

and sampling can be done with

```bash
python experiments/flow_approx/phi_four.py configs/flow_approx/phi_four.yaml --save_samples --seed {SEED}
```

### Appendix

The flows for the figure 8 can be retrained using

```bash
python experiments/flow_approx/many_flows_two_moons.py --load_path {OUTPUT_PATH} --seed {SEED}
```



## 🏗️ TODO

* Fix `mcmc.hmc.HMC` : right now the warmup phase is broken
* Allow learning a preconditioning matrix for `mcmc.mala.MALA`
