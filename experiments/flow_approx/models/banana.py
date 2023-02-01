# Banana distribution

# Libraries
import torch
import math

# Banana transform
class BananaTransform(torch.distributions.transforms.Transform):

	domain = torch.distributions.constraints.real_vector
	codomain = torch.distributions.constraints.real_vector
	bijective = True
	sign = +1

	def __init__(self, a, b, dim, cache_size=0):
		self.a = a
		self.b = b
		self.coef = b * a * a
		self.log_a = math.log(a)
		self.dim = dim
		self.even_mask = torch.arange(0, dim, 2)
		self.odd_mask = torch.arange(1, dim, 2)
		super(BananaTransform, self).__init__(cache_size=cache_size)

	def __eq__(self, other):
		return isinstance(other, BananaTransform)

	def __call__(self, x):
		y = x.clone()
		y[...,self.even_mask] = self.a * x[..., self.even_mask]
		y[...,self.odd_mask] += self.b * torch.square(y[...,self.even_mask])
		y[...,self.odd_mask] -= self.coef
		return y

	def _inverse(self, y):
		x = y.clone()
		x[...,self.even_mask] = y[...,self.even_mask] / self.a
		x[...,self.odd_mask] -= self.b * torch.square(y[...,self.even_mask])
		x[...,self.odd_mask] += self.coef
		return x

	def log_abs_det_jacobian(self, x, y):
		return torch.ones(x.shape[:-1], device=x.device) * (self.dim / 2) * self.log_a

# Banana distribution
class Banana:
    def __init__(self, dim, device, a=10.0, b=0.02):
        # Make the base multivariate normal distribution
        banana_loc = torch.zeros(dim, device=device)
        banana_covariance_matrix = torch.eye(dim, device=device) 
        # Make the transformed distribution
        self.distribution = torch.distributions.TransformedDistribution(
            base_distribution=torch.distributions.MultivariateNormal(
            	loc=banana_loc,
            	covariance_matrix=banana_covariance_matrix
        	),
            transforms=BananaTransform(a=a, b=b, dim=dim)
        )

    def log_prob(self, value):
        return self.distribution.log_prob(value)

    def sample(self, sample_shape):
        return self.distribution.sample(sample_shape=sample_shape)

if __name__ == "__main__":

	# Librairies
	import matplotlib.pyplot as plt
	import numpy as np

	# Get the Pytorch device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Make a banana distribution
	dim = 2
	dist = Banana(dim=dim, device=device)
	samples = dist.sample(sample_shape=(1024,)).cpu()

	# Plot the samples
	plt.scatter(samples[:,0], samples[:,1], alpha=0.2)
	plt.show()

	# Plot the log prob
	res = 128
	X = np.linspace(-20, 20, res)
	Y = np.linspace(-5, 5, res)
	X, Y = np.meshgrid(X,Y)
	Z = np.stack([X,Y], axis=-1)
	print(Z.shape)
	Z = torch.from_numpy(Z).to(device).reshape((-1,2))
	Z_ = dist.log_prob(Z).reshape((res,res))
	plt.contourf(X, Y, Z_.exp().cpu().numpy(), 20, cmap='GnBu')
	plt.show()

