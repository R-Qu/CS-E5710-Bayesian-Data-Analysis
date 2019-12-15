import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import numpy as np

mu_a = 0
sigma_a = 2
mu_b = 10
sigma_b = 10
corr = 0.5

mean = np.array([mu_a, mu_b])
cov = np.array([[sigma_a**2, corr * sigma_a * sigma_b], 
[corr * sigma_a * sigma_b, sigma_b**2]])

alpha, beta = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-10, 30, 100))


#print('Shape of the samples from prior: ', samples.shape)
#print(samples[:10],samples[:10,0])

dose = np.array([-0.86, -0.3, -0.05, 0.72])
deaths = np.array([0, 1, 3, 5])
animals = np.array([5, 5, 5, 5])


prior = stats.multivariate_normal(mean, cov)
#samples = prior.rvs(6)
samples = np.array([[1.896,24.76],[-3.6,20.04],[0.374,6.15],[0.964,18.65],[-3.123,8.16],[-1.581,17.4]])

print(samples)


theta = 1 / (1 + np.exp(-(samples[:, 0, None] + samples[:, 1, None] * dose)))

weights = np.prod(
    theta**deaths * (1 - theta)**(animals - deaths),axis=1)


weights_norm = (weights) / np.sum(weights)
print('weights_norm:',weights_norm )
S_eff = 1 / np.sum(weights_norm**2)
print('The effective sample size: ', S_eff)

mean_posterior = sum(weights[ : , None] * samples) / sum(weights)

print('The posterior mean of alpha : ', mean_posterior[0])
print('The posterior mean of beta  : ', mean_posterior[1])
