import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import numpy as np

def bioassaylp(a, b, x, y, n):
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
'''
a)
'''
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

alpha, beta = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-10, 30, 100))

'''
b)
'''

def p_log_prior(alpha, beta):

    prior= stats.multivariate_normal(mean, cov)
    pos = np.dstack((alpha, beta))
    log_prior = prior.logpdf(pos)

    return log_prior
#print('test',p_log_prior(3,9))

dose = np.array([-0.86, -0.3, -0.05, 0.72])
deaths = np.array([0, 1, 3, 5])
animals = np.array([5, 5, 5, 5])

'''
c) 
'''

def p_log_posterior(alpha, beta, x, y, n):
    prior= stats.multivariate_normal(mean, cov)
    pos = np.dstack((alpha, beta))
    log_prior = prior.logpdf(pos)

    alpha = np.expand_dims(alpha, axis=-1)
    beta = np.expand_dims(beta, axis=-1)
    t = alpha + beta*x
    et = np.exp(t)
    z = et/(1.+et)
    log_likelihood = np.sum(y*np.log(z)+ (n-y)*np.log(1.0-z), axis=-1)

    log_posterior = log_prior + log_likelihood

    return log_posterior 
#print('testposterior',p_log_posterior(3, 9, dose, deaths, animals))

'''
d) 
'''
alpha, beta = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-10, 30, 100))
posterior = np.exp(p_log_posterior(alpha, beta, dose, deaths, animals))
plt.contourf(alpha, beta,posterior, cmap=plt.cm.Greys)
plt.xlabel('alpha')
plt.ylabel('beta')
plt.title('Posterior Distribution')
plt.grid(linewidth=0.8, alpha=0.2)
plt.colorbar(plt.contourf(alpha, beta, posterior, cmap=plt.cm.Greys))
plt.savefig('./log_posterior.png')
plt.show()

'''
e) 2. Sample draws of alpha and beta from the prior distribution.
'''
prior = stats.multivariate_normal(mean, cov)
samples = prior.rvs(10000)
#print('Shape of the samples from prior: ', samples.shape)

theta = 1 / (1 + np.exp(-(samples[:, 0, None] + samples[:, 1, None] * dose)))
weights = np.prod(
    theta**deaths * (1 - theta)**(animals - deaths), axis=1)

weights_norm = (weights) / np.sum(weights)
#print('Shape of the weights of the likelihood: ', weights.shape)

'''
f) 
'''
posterior_mean = sum(weights[ : , None] * samples) / sum(weights)

print('posterior mean of alpha : ', posterior_mean[0])
print('posterior mean of beta  : ', posterior_mean[1])
'''
g) 
'''
s_eff = 1 / np.sum(weights_norm**2)
print('effective sample size: ', s_eff)

'''
h) 
'''
scode = np.random.choice(a=10000, size=1000, replace=False, p=weights_norm)
resamples = samples[scode]
print('mean of alpha: ', np.mean(resamples[:, 0]))
print('mean of beta: ', np.mean(resamples[:, 1]))

plt.xlim([-4, 4])
plt.ylim([-10, 30])
plt.xlabel('alpha')
plt.ylabel('beta')
plt.grid(linewidth=0.8, alpha=0.2)
plt.scatter(resamples[:, 0], resamples[:, 1],8,color='grey')
plt.title('Posterior Samples')
plt.savefig('./posterior_samples.png')
plt.show()

plt.xlim([-4, 4])
plt.ylim([-10, 30])
plt.xlabel('alpha')
plt.ylabel('beta')
plt.grid(linewidth=0.8, alpha=0.2)
plt.contourf(alpha, beta, posterior, cmap=plt.cm.Greys)
plt.colorbar(plt.contourf(alpha, beta, posterior, cmap=plt.cm.Greys))
plt.scatter(resamples[:, 0], resamples[:, 1], 8, alpha=.15, color='grey')
plt.title('Contourf & Ccatter Comparision')
plt.savefig('./contourf_scatter.png')
plt.show()

'''
i) 
'''
beta_resample = resamples[:, 1]
alpha_resample = resamples[:, 0]
pos = beta_resample > 0
p_harmful = (beta_resample[pos].size/(beta_resample.size + 1))
print('Probability that the drug is harmful:', p_harmful)

'''
j)
'''
sample_ld50 = - alpha_resample[pos]/beta_resample[pos]
y_range = np.arange(-0.4, 0.4, 0.01)
plt.hist(sample_ld50, y_range, ec='white', color='grey')
plt.xlabel('LD50')
plt.title('Histogram')
plt.savefig('./histogram.png')
plt.show()
