#%%
import matplotlib
matplotlib.use('TkAgg')

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystan

#%% The data
machines = pd.read_fwf('./ex7/factory.txt', header=None).values
machines_transposed = machines.T

#%% Pooled model
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
    real ypred;
    ypred = normal_rng(mu, sigma);
}
'''

#%% fitting data to stan model
machines_pooled = machines.flatten()
model_pooled = pystan.StanModel(model_code=stan_code_pooled)
data_pooled = dict(
    N=machines_pooled.size,
    y=machines_pooled
)

#%% sampling
fit_pooled = model_pooled.sampling(data=data_pooled)
print(fit_pooled)

#%% hist
y_pred_pooled = fit_pooled.extract()['ypred']
plt.hist(y_pred_pooled, bins=20, ec='white')
plt.savefig('./ex7/report/pooled_hist.png')
plt.figure(0)

mu = fit_pooled.extract()['mu']
plt.hist(mu, bins=20, ec='white')
plt.savefig('./ex7/report/pooled_hist_mu.png')
plt.figure(0)

#%% Separate model
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
    real ypred;
    ypred = normal_rng(mu[6], sigma[6]);
}
'''

#%% fitting data into the stan model
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

#%% sampling
fit_separate = model_seperate.sampling(data=data_separate, n_jobs=-1)
print(fit_separate)

#%% hist
y_pred_separate = fit_separate.extract()['ypred']
plt.hist(y_pred_separate, bins=20, ec='white')
plt.savefig('./ex7/report/separate_hist.png')
plt.figure(0)

#%% hist
mu_data_separate = fit_separate.extract()['mu']
plt.hist(mu_data_separate[:, 5], bins=20, ec='white')
plt.savefig('./ex7/report/separate_hist_mu_six.png')
plt.figure(0)

#%% Hierarchical model
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
    ypred6 = normal_rng(mu[6], sigma);
    mu7 = normal_rng(mu0, sigma0);
}
'''

#%% fitting data into the stan model
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

#%% sampling
fit_hierarchical = model_hierarchical.sampling(data=data_hierarchical, n_jobs=-1)
print(fit_hierarchical)

#%% hist
mu_data_hierarchical = fit_hierarchical.extract()['mu']
plt.hist(mu_data_hierarchical[:, 5], bins=20, ec='white')
plt.savefig('./ex7/report/hierarchical_hist_mu_six.png')
plt.figure(0)

#%% hist
y_pred_hierarchical = fit_hierarchical.extract()['ypred6']
plt.hist(y_pred_hierarchical, bins=20, ec='white')
plt.savefig('./ex7/report/hierarchical_hist.png')
plt.figure(0)

#%% hist
mu_data_hierarchical_7 = fit_hierarchical.extract()['mu7']
plt.hist(mu_data_hierarchical_7, bins=20, ec='white')
plt.savefig('./ex7/report/hierarchical_hist_mu_7.png')
plt.figure(0)
