import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import random
from psrf import psrf
from bioarraylp import bioassaylp

sigma_a = 2
sigma_b = 10
mu_a = 0
mu_b = 10
corr = 0.5
cov_matrix = np.array([[sigma_a**2, corr * sigma_a * sigma_b], 
[corr * sigma_a * sigma_b, sigma_b**2]])
mean = np.array([mu_a, mu_b])

doses = np.array([-0.86, -0.3, -0.05, 0.72])
deaths = np.array([0, 1, 3, 5])
animals = np.array([5, 5, 5, 5])

def get_next_pos(pos, cov):

    sample_pos = stats.multivariate_normal.rvs(pos, cov, size=1)
    sample_pos = np.array(sample_pos)

    likelihood_sample_pos = bioassaylp(sample_pos[0], sample_pos[1],doses,deaths,animals)
    likelihood_pos = bioassaylp(pos[0],pos[1],doses,deaths,animals)

    prior_multivar_nor = stats.multivariate_normal(mean, cov_matrix)
    prior_sample_pos = prior_multivar_nor.pdf(sample_pos)
    prior_pos = prior_multivar_nor.pdf(pos)

    post_sample_pos = np.exp(likelihood_sample_pos) * prior_sample_pos
    post_pos = np.exp(likelihood_pos) * prior_pos

    ratio = post_sample_pos / post_pos

    if ratio >= 1:
        return sample_pos
    else:
        uniform_random_sample = stats.uniform(0,1).rvs(1)[0]
        if uniform_random_sample < ratio:
            return sample_pos 

    return pos

def generate_chains(sample_size, number_of_chains,worm_up):
    print('number of draws:', sample_size)
    chains = []
    for i in range(number_of_chains):
        pos = [random.randint(-2, 4), random.randint(-5, 30)]
        chain = [pos]
        for j in range(sample_size):
            next_pos= get_next_pos(chain[-1], cov_matrix)
            chain.append(next_pos)
        print('starting point:', i, pos, ' PSRF:', psrf(chain))
        chains.append(chain)

    wormup_chains = []
    for chain in chains:
        wormup_chains.append(chain[worm_up:])
    return wormup_chains
    
chains = generate_chains(sample_size=2000, number_of_chains=15, worm_up=500)

for chain in chains:

    x = np.array(chain)[:, 0]
    y = np.array(chain)[:, 1]
    plt.xlim([-4, 10])
    plt.ylim([-10, 40])
    plt.plot(x,y,alpha=0.5,marker='.',linewidth=0,markersize=1)
  
plt.savefig('./1.png', dpi=150)
plt.show()

print('1 chain')
chain = generate_chains(sample_size=10000, number_of_chains=1, worm_up=500)[0]
print('Potential Scale Reduction Factor (PSRF)', psrf(chain))

x = np.array(chain)[:, 0]
y = np.array(chain)[:, 1]
plt.xlim([-4, 10])
plt.ylim([-10, 40])
plt.plot(x,y,alpha=0.5,marker='.',linewidth=0,markersize=1)
plt.savefig('./2', dpi=150)
plt.show()

