import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import numpy as np

def bioassaylp(a, b, x, y, n):
    # Log posterior density for the bioassay problem
    # last axis for the data points
    a = np.expand_dims(a, axis=-1)
    b = np.expand_dims(b, axis=-1)
    # these help using chain rule in derivation
    t = a + b*x
    et = np.exp(t)
    z = et/(1.+et)
    # negative log posterior (error function to be minimized)
    lp = np.sum(y*np.log(z)+ (n-y)*np.log(1.0-z), axis=-1)
    return lp

# Init all the params based on the description
mu_a = 0
sigma_a = 2
mu_b = 10
sigma_b = 10
corr = 0.5

mean = np.array([mu_a, mu_b])
print('mean:',mean)

cov = np.array([[sigma_a**2, corr * sigma_a * sigma_b], 
[corr * sigma_a * sigma_b, sigma_b**2]])
print('covariance:',cov)

# create a grid and it's points using x and y ranges
alpha, beta = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-10, 30, 100))
pos = np.dstack((alpha, beta))

# prior distribution

prior = stats.multivariate_normal(mean, cov)

plt.contourf(alpha, beta, prior.pdf(pos), cmap=plt.cm.hot)
plt.title('Prior Distribution')
plt.grid(linewidth=0.9, alpha=0.2)
plt.savefig('./1_prior_distribution.png')
plt.show()
plt.contourf(alpha, beta, prior.logpdf(pos), cmap=plt.cm.hot)
plt.show()