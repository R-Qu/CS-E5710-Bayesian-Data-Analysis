
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

MEAN = 0.2
VARIANCE = 0.01

alpha = MEAN * ( (MEAN * (1 - MEAN) / VARIANCE) - 1 )
beta = alpha * (1 - MEAN) / MEAN

x_range = np.linspace(0, 1, 100)
y_range = stats.beta.pdf(x_range, alpha, beta)

# a) Plot the density function of Beta-distribution
plt.plot(x_range, y_range)
plt.xlabel('probability')
plt.ylabel('density')
plt.savefig('./distribution.png')

# b) Take a sample of 1000 random numbers and plot a histogram 
random_samples = stats.beta.rvs(alpha, beta, size=1000)
plt.hist(random_samples, color='grey', density=True, alpha=0.5)
#plt.show()
plt.savefig('./distribution_hist.png')

# c) Compute the sample mean and variance from the drawn sample
sample_mean = np.mean(random_samples)
sample_variance = np.var(random_samples)

print('sample mean: ', sample_mean)
print('sample variance: ', sample_variance)

# d) Estimate the central 95%-interval of the distribution from the drawn samples
sample_percentile = np.percentile(random_samples, q=2.5), np.percentile(random_samples, q=97.5)
print('sample central percentile 95%: ', sample_percentile)
