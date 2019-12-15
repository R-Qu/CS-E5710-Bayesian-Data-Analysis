from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# a) summarize
x = np.arange(0, 1, 0.001)
prior = stats.beta.pdf(x, a=2, b=10)
posterior = stats.beta.pdf(x, a=46, b=240)

plt.plot(x, prior, label='Prior Beta(2,10)', color='red')
plt.plot(x, posterior, label='Posterior Beta(46,240)', color='green')

plt.xlabel('P(algae present) = π')
plt.ylabel('density')
plt.legend()
plt.savefig('./prob_distribution.png')
plt.show()

interval=stats.beta.interval(0.90, a=46, b=240)
print("90% interval estimate:",interval)

mean =stats.beta.mean(a=46, b=240, loc=0, scale=1)
print("mean:",mean)
median =stats.beta.median(a=46, b=240, loc=0, scale=1)
print("median:",median)
mode = (46-1)/(46+240-2)
print("mode:",mode)
# b) P(π0= 0.2)
cumulative = stats.beta.cdf(0.2, a=46, b=240)
print('cumulative at 0.2: ', cumulative)

x2_line = np.arange(0, 0.3, 0.001)
posterior2_line = stats.beta.pdf(x2_line, a=46, b=240)

x2 = np.arange(0.096, 0.2, 0.001)
posterior2 = stats.beta.pdf(x2, a=46, b=240)

plt.fill_between(x2, posterior2, alpha=0.7)
plt.plot(x2_line, posterior2_line, color='green')
plt.xlabel('P(algae exist) = π')
plt.legend()
plt.savefig('./cumulative.png')
plt.show()
