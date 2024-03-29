import numpy as np
import pandas as pd
import pystan

machines = pd.read_fwf('./factory.txt', header=None).values
machines_transposed = machines.T

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
    vector[K+1] ypred;
    real mu7;
    mu7 = normal_rng(mu0, sigma0);
    for (i in 1:K)
        ypred[i] = normal_rng(mu[i], sigma);
    ypred[K+1] = normal_rng(mu7, sigma);
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


utility = np.zeros(7)
ypred = fit_hierarchical.extract(permuted=True)['ypred']
ulist=[]
for i in range(7):
    for j in range(0, len(ypred)):
        if ypred[j, i] < 85:
            utility[i] -= 106
        else:
            utility[i] += (200-106)

        i_utility = utility[i]/len(ypred)
    
    ulist.append(('Machine', i+1, i_utility))
    #print('Machine', i+1, i_utility)

for u in ulist:
    print(u)

sorted_ulist= sorted(ulist, key=lambda x: x[2])
for s in sorted_ulist:
    print(s) 

