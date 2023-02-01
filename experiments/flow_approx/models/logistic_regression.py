# Horseshoe Logistic Regression
# Based on https://github.com/albcab/TESS/blob/master/distributions.py

# Libraries
import torch
import numpy as np
import pandas as pd
from nflows import transforms
from torch.nn import functional as F

# Make the IAF flow
def make_iaf_flow(dim, num_layers=3, num_blocks=2, use_residual_blocks=True):
	# Define an invertible transformation.
	flow_layers = []
	for i in range(num_layers):
		flow_layers.append(transforms.MaskedAffineAutoregressiveTransform(features=dim,
		                                                                  hidden_features=dim,
		                                                                  num_blocks=num_blocks,
		                                                                  use_residual_blocks=use_residual_blocks,
		                                                                  activation=F.elu))
		flow_layers.append(transforms.ReversePermutation(features=dim))
	transform = transforms.CompositeTransform(flow_layers)
	return transform

# Horseshoe Logistic Regression
class HorseshoeLogisticRegression:

	def __init__(self, X, y, device):
		# Dataset
		self.X = X.float().to(device)
		self.y = y.float().to(device)
		# Priors
		self.beta_prior = torch.distributions.Normal(
			loc=torch.zeros((X.shape[1],)).to(device),
			scale=torch.ones((X.shape[1],)).to(device),
			validate_args=False
		)
		self.lamda_prior = torch.distributions.transformed_distribution.TransformedDistribution(
			base_distribution=torch.distributions.Gamma(
				concentration=0.5 * torch.ones((X.shape[1],)).to(device),
				rate=0.5 * torch.ones((X.shape[1],)).to(device),
				validate_args=False
			),
			transforms=[torch.distributions.transforms.ExpTransform().inv],
			validate_args=False
		)
		self.tau_prior = torch.distributions.transformed_distribution.TransformedDistribution(
			base_distribution=torch.distributions.Gamma(
				concentration=torch.tensor(0.5).to(device),
				rate=torch.tensor(0.5).to(device),
				validate_args=False
			),
			transforms=[torch.distributions.transforms.ExpTransform().inv],
			validate_args=False
		)

	def log_prob(self, params):
		# Ensure the shape of the params
		params = params.reshape((-1, params.shape[-1]))
		# Unpack the parameters
		beta, lamda, tau = params[...,:self.X.shape[1]], params[...,self.X.shape[1]:2*self.X.shape[1]], params[...,-1]
		# Compute the prior
		ret = (self.beta_prior.log_prob(beta) + self.lamda_prior.log_prob(lamda)).sum(dim=-1) \
				+ self.tau_prior.log_prob(tau)
		# Compute the likelihood
		weights = torch.exp(tau)[:,None] * beta * torch.exp(lamda)
		probs = torch.special.expit(torch.matmul(self.X, weights.T).T)
		probs = torch.clip(probs, 1e-6, 1-1e-6)
		# Make a Bernouilli distribution
		bern = torch.distributions.bernoulli.Bernoulli(probs=probs, validate_args=False)
		ret += bern.log_prob(self.y).sum(dim=-1)
		return ret

# Load German credit data
def load_german_credit(data_path='experiments/flow_approx/models/logistic_regression/german.data-numeric'):
	# Load the dataframe
	data = pd.read_table(data_path, header=None, delim_whitespace=True)
	# Pre processing data as in NeuTra paper
	y = -1 * (data.iloc[:, -1].values - 2)
	X = data.iloc[:, :-1].apply(lambda x: -1 + (x - x.min()) * 2 / (x.max() - x.min()), axis=0).values
	X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
	# Return the data
	return torch.from_numpy(X), torch.from_numpy(y)

if __name__ == "__main__":

	# Get the Pytorch device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	device = torch.device('cpu')

	# Load the data
	X, y = load_german_credit()

	# Make the logistic regression
	target = HorseshoeLogisticRegression(X=X, y=y, device=device)
	dim = X.shape[1] * 2 + 1
