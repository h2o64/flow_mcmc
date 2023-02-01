# Base MCMC class

# Librairies
from abc import ABC, abstractmethod

# Base MCMC Class
class MCMC(ABC):

	def __init__(self):
		self.diagnostics = {}

	@abstractmethod
	def sample(self, x_s_t_0, n_steps, target, temp=1.0, warmup_steps=0, verbose=False):
		pass

	def reset_diagnostics(self):
		self.diagnostics = {}

	def get_diagnostics(self, k):
		if k not in self.diagnostics: return None
		return self.diagnostics[k]

	def has_diagnostics(self, k):
		return k in self.diagnostics