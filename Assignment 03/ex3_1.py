from math import sqrt
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

#data=[13.357, 14.928, 14.896, 14.820]#testdata
data=[13.357, 14.928, 14.896, 15.297, 14.82, 12.067, 14.824, 13.865, 17.447]
n = len(data)
estimated_mean = np.mean(data)
estimated_variance = stats.tvar(data)

'''
a) What can you say about the unknown μ?
'''
     
intervala = stats.t.interval(0.95,df=n-1,loc=estimated_mean,scale=sqrt(estimated_variance/n))

print('estimated mean:', estimated_mean)
print('estimated variance:', estimated_variance)
print('estimated standard deviation:', sqrt(estimated_variance))
print('a)95% intervals:', intervala)


x_range = np.arange(estimated_mean - 3 * sqrt(estimated_variance), \
        estimated_mean + 3 * sqrt(estimated_variance),0.01)
y_range1 = stats.t.pdf(x=x_range,df=n-1,\
          loc=estimated_mean,scale=sqrt(estimated_variance/n))
plt.plot(x_range, y_range1)
plt.savefig('./1apdf.png')
plt.title('pdf')
plt.show() 

y_range2 = stats.t.cdf(x=x_range,df=n-1,\
          loc=estimated_mean,scale=sqrt(estimated_variance/n)) 
plt.plot(x_range, y_range2)
plt.savefig('./1acdf')
plt.title('cdf')
plt.show()
'''
b)
'''
std_y = np.std(data, ddof=1)
scale = sqrt(1 + 1/n) * std_y
y_posterior_mu = stats.t.pdf(x=x_range,df=n-1,\
                loc=estimated_mean, scale=scale)

y_posterior_mu2 = stats.t.cdf(x=x_range,df=n-1,\
                loc=estimated_mean, scale=scale)
intervalb = stats.t.interval(0.95,df=n-1,\
                loc=estimated_mean, scale=scale)
print('b) 95%interval',intervalb)

figure = plt.plot(x_range, y_posterior_mu)
plt.savefig('./1bpdf.png')
plt.title('pdf')
plt.show()

figure = plt.plot(x_range, y_posterior_mu2)
plt.savefig('./1bcdf.png')
plt.title('cdf')
plt.show()
