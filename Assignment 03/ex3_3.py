from math import sqrt
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

def model(data):
    n = len(data)
    mean = np.mean(data)
    variance = stats.tvar(data)
    x_range = np.arange(
        mean - 3 * sqrt(variance),
        mean + 3 * sqrt(variance),
        0.01)
    mu = stats.t.pdf(x=x_range,df=n-1,loc=mean,scale=sqrt(variance/n))
    return n, mean, variance, x_range, mu

data_1 = [13.357,14.928,14.896,15.297,14.82,12.067,14.824,13.865,17.447]
data_2 = [15.98,14.206,16.011,17.25,15.993,15.722,17.143,15.23,15.125,16.609,14.735,15.881,15.789]
n_1, mean_1, variance_1, x_range_1, mu_1 = model(data_1)
n_2, mean_2, variance_2, x_range_2, mu_2 = model(data_2)

mu_1 = stats.t.rvs(df=n_1-1, loc=mean_1, scale=sqrt(variance_1/n_1), size=100000)
mu_2 = stats.t.rvs(df=n_2-1, loc=mean_2, scale=sqrt(variance_2/n_2), size=100000)
mu_d = mu_1 - mu_2

interval_1= stats.t.interval(0.95, df=n_1-1, loc=mean_1, scale=sqrt(variance_1/n_1))
interval_2= stats.t.interval(0.95, df=n_2-1, loc=mean_2, scale=sqrt(variance_2/n_2))

print('windshieldy1 mean',model(data_1)[1])
print('windshieldy2 mean',model(data_2)[1])
print('windshieldy1 interval',interval_1)
print('windshieldy2 interval',interval_2)
print('mean diff 95% mean', np.mean(mu_d))
print('mean diff 95% Intervals', np.percentile(mu_d, 2.5), np.percentile(mu_d, 97.5))

plt.hist(mu_d, bins=50, ec='white', color='grey', alpha=0.5)
plt.savefig('./3.png')
plt.show()
pdf1=stats.t.pdf(x=mu_1,df=n_1-1,loc=mean_1, scale=sqrt(variance_1/n_1))
plt.plot(mu_1,pdf1)
plt.show()
'''
b)
'''
print(stats.percentileofscore(mu_d, 0), '%')
