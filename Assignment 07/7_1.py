
import matplotlib
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystan

drowning_data = pd.read_fwf('./drowning.txt').values
years = drowning_data[:, 0]
drowning = drowning_data[:, 1]

print("mean:", np.mean(drowning))
print("standard deviation:", np.std(drowning,ddof=1))

plt.plot(years, drowning)
plt.savefig('./trend.png')
plt.show()

stan_code = '''
data {
  int<lower=0> N; // number of data points
  vector[N] x;    // observation year
  vector[N] y;    // observation number of drowned
  real xpred;     // prediction year
  real tau;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
transformed parameters {
  vector[N] mu;
  mu = alpha + beta * x;
}
model {
  beta ~ normal(0, tau*tau);
  y ~ normal(mu, sigma);
}
generated quantities {
  real ypred;
  ypred = normal_rng(alpha + beta * xpred, sigma);
}
'''

dist = norm(loc=0, scale=26.612146647843897)

print(dist.cdf(-69))

stan_model = pystan.StanModel(model_code=stan_code)

data = dict(
    N=len(years),
    x=years,
    y=drowning,
    xpred=2019,
    tau=26.612146647843897,
)

fit = stan_model.sampling(data=data)
print(fit)

y_pred = fit.extract()['ypred']
plt.hist(y_pred, bins=20, ec='white')
plt.savefig('./hist.png')
plt.show()
