from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 0.7, 0.001)
beta1 = stats.beta.pdf(x, a=2, b=8)
beta2= stats.beta.pdf(x, a=2, b=10)
beta3= stats.beta.pdf(x, a=2, b=12)
beta4= stats.beta.pdf(x, a=5, b=25)
beta5= stats.beta.pdf(x, a=10, b=50)
beta6= stats.beta.pdf(x, a=20, b=100)

plt.plot(x, beta1, label='Beta(2,8)', color='red')
plt.plot(x, beta2, label='Beta(2,10)', color='orange')
plt.plot(x, beta3, label='Beta(2,12)', color='yellow')
plt.plot(x, beta4, label='Beta(5,25)', color='green')
plt.plot(x, beta5, label='Beta(10,50)', color='blue')
plt.plot(x, beta6, label='Beta(20,100)', color='purple')

plt.xlabel('P(algae present)')
plt.ylabel('density')
plt.legend()
plt.savefig('./plotpdfs.png')
plt.show()

beta01 = stats.beta.cdf(x, a=2, b=8)
beta02= stats.beta.cdf(x, a=2, b=10)
beta03= stats.beta.cdf(x, a=2, b=12)
beta04= stats.beta.cdf(x, a=5, b=25)
beta05= stats.beta.cdf(x, a=10, b=50)
beta06= stats.beta.cdf(x, a=20, b=100)

plt.plot(x, beta01, label='Beta(2,8)', color='red')
plt.plot(x, beta02, label='Beta(2,10)', color='orange')
plt.plot(x, beta03, label='Beta(2,12)', color='yellow')
plt.plot(x, beta04, label='Beta(5,25)', color='green')
plt.plot(x, beta05, label='Beta(10,25)', color='blue')
plt.plot(x, beta06, label='Beta(20,100)', color='purple')

plt.xlabel('P(algae present)')
plt.ylabel('cumulative density')
plt.legend()
plt.savefig('./plotcdfs.png')
plt.show()
