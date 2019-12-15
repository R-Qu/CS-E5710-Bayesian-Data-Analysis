import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import pystan
import psis

machines = pd.read_fwf('./factory.txt', header=None).values
machines_transposed = machines.T

def psisloo_computation(log_lik, fig_name, model_name):
    psis_loo = psis.psisloo(log_lik)

    lppd_loocv = psis_loo[0]
    psis_hist = psis_loo[2]

    S = np.size(log_lik, 0)
    lppd = sum(np.log([1/S*sum(np.exp(col)) for col in log_lik.T]))
    p_loocv = lppd - lppd_loocv

    print('PSIS-LOO: ', lppd_loocv)
    print('p_eff: ', p_loocv)

    plt.hist(psis_hist, bins= np.linspace(0, 1, num=50), color='grey',ec='white')
    plt.savefig('./{0}'.format(fig_name))
    plt.show()

'''
Pooled model
'''
stan_code_pooled = '''
data {
    int<lower=0> N;       // number of data points
    vector[N] y;          //
}
parameters {
    real mu;              // group means
    real<lower=0> sigma;  // common std
}
model {
    y ~ normal(mu, sigma);
}
generated quantities {
    real ypred6;
    vector[N] log_lik;
    ypred6 = normal_rng(mu, sigma);
    for (i in 1:N)
        log_lik[i] = normal_lpdf(y[i] | mu, sigma);
}
'''
machines_pooled = machines.flatten()
model_pooled = pystan.StanModel(model_code=stan_code_pooled)
data_pooled = dict(
    N=machines_pooled.size,
    y=machines_pooled
)

fit_pooled = model_pooled.sampling(data=data_pooled)
print(fit_pooled)

log_lik_pooled = fit_pooled.extract(permuted=True)['log_lik']
psisloo_computation(log_lik_pooled, 'pooled_hist.png', 'Pool')

'''
Separate model
'''
stan_code_separate = '''
data {
    int<lower=0> N;               // number of data points
    int<lower=0> K;               // number of groups
    int<lower=1,upper=K> x[N];    // group indicator
    vector[N] y;
}
parameters {
    vector[K] mu;                 // group means
    vector<lower=0>[K] sigma;     // group stds
}
model {
    y ~ normal(mu[x], sigma[x]);
}
generated quantities {
    real ypred6;
    vector[N] log_lik;
    ypred6 = normal_rng(mu[6], sigma[6]);
    for (i in 1:N)
        log_lik[i] = normal_lpdf(y[i] | mu[x[i]], sigma[x[i]]);
}
'''

model_seperate = pystan.StanModel(model_code=stan_code_separate)
data_separate = dict(
    N=machines_transposed.size,
    K=6,
    x=[
        1, 1, 1, 1, 1,
        2, 2, 2, 2, 2,
        3, 3, 3, 3, 3,
        4, 4, 4, 4, 4,
        5, 5, 5, 5, 5,
        6, 6, 6, 6, 6,
    ],
    y=machines_transposed.flatten()
)

fit_separate = model_seperate.sampling(data=data_separate, n_jobs=-1)
print(fit_separate)

log_lik_separate = fit_separate.extract(permuted=True)['log_lik']
psisloo_computation(log_lik_separate, 'separate_hist.png', 'Separate')

'''
Hierarchical model
'''
stan_code_hierarchical = '''
data {
    int<lower=0> N;             // number of data points
    int<lower=0> K;             // number of groups
    int<lower=1,upper=K> x[N];  // group indicator
    vector[N] y;
}
parameters {
    real mu0;                   // prior mean
    real<lower=0> sigma0;       // prior std
    vector[K] mu;               // group means
    real<lower=0> sigma;        // common std
}
model {
    mu ~ normal(mu0, sigma0);
    y ~ normal(mu[x], sigma);
}
generated quantities {
    real ypred6;
    real mu7;
    vector[N] log_lik;
    ypred6 = normal_rng(mu[6], sigma);
    mu7 = normal_rng(mu0, sigma0);
    for (i in 1:N)
        log_lik[i] = normal_lpdf(y[i] | mu[x[i]], sigma);
}
'''

model_hierarchical = pystan.StanModel(model_code=stan_code_hierarchical)
data_hierarchical = dict(
    N=machines_transposed.size,
    K=6,
    x=[
        1, 1, 1, 1, 1,
        2, 2, 2, 2, 2,
        3, 3, 3, 3, 3,
        4, 4, 4, 4, 4,
        5, 5, 5, 5, 5,
        6, 6, 6, 6, 6,
    ],
    y=machines_transposed.flatten()
)
fit_hierarchical = model_hierarchical.sampling(data=data_hierarchical, n_jobs=-1)
print(fit_hierarchical)

log_lik_hierarchical = fit_hierarchical.extract(permuted=True)['log_lik']
psisloo_computation(log_lik_hierarchical, 'hierarchical_hist.png', 'hierarchical')