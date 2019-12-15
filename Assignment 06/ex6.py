import matplotlib.pyplot as plt
import numpy as np
import pystan

# Init all the params based on the description
sigma_a = 2
sigma_b = 10
mu_a = 0
mu_b = 10
cor = 0.5
cov_matrix = np.array([
    [sigma_a**2,                cor * sigma_a * sigma_b],
    [cor * sigma_a * sigma_b,   sigma_b**2]
])
mean = np.array([mu_a, mu_b])

doses = np.array([-0.86, -0.3, -0.05, 0.72])
deaths = np.array([0, 1, 3, 5])
number_of_animals = np.array([5, 5, 5, 5])

# stan code
stan_code = '''
data {
    int<lower=0> n;
    int<lower=0> deaths[n];
    int<lower=0> numb_of_animals[n];
    vector[n] doses;
    vector[2] mu;
    cov_matrix[2] cov_m;
}
parameters {
    vector[2] alpha_beta;
}
model {
    alpha_beta ~ multi_normal(mu, cov_m);
    deaths ~ binomial_logit(numb_of_animals, alpha_beta[1] + alpha_beta[2] * doses);
}
'''

# calculation code
sm = pystan.StanModel(model_code=stan_code)
data = dict(
    n=len(number_of_animals),
    deaths=deaths,
    numb_of_animals=number_of_animals,
    doses=doses,
    mu=mean,
    cov_m=cov_matrix,
)
fit = sm.sampling(data=data, chains=10, iter=10000, warmup=1000)
print('fit', fit)

# graph
extracted_samples = fit.extract()
samples = extracted_samples['alpha_beta']
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.2, s=1, color='grey')
plt.ylabel('beta')
plt.xlabel('alpha')
plt.savefig('./scatter.png', dpi=150)

# output
'''
                mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
alpha_beta[1]   0.99    0.03    0.9  -0.62   0.37   0.91   1.57   2.78    813    1.0
alpha_beta[2]  10.67    0.17   4.69   3.39   7.19  10.08   13.5  21.29    783    1.0
'''
