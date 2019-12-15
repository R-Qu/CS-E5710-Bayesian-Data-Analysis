from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

x_range = np.arange(0, 0.2, 0.001)

'''
a)
'''
control = 674
control_died = 39
control_a = control_died + 1
control_b = control - control_a + 1
control_posterior = control_a/control
control_pdf = stats.beta.pdf(x_range, control_a, control_b)

treatment = 680
treatment_died = 22
treatment_a = treatment_died + 1
treatment_b = treatment - treatment_a + 1
treatment_posterior = treatment_a/treatment

control_pdf = stats.beta.pdf(x_range, control_a, control_b)
treatment_pdf = stats.beta.pdf(x_range, treatment_a, treatment_b)
plt.plot(x_range, control_pdf,label='Control group')
plt.plot(x_range, treatment_pdf,label='Treatment group')
plt.legend()
plt.savefig('./2apdf.png')
plt.show()

control_cdf = stats.beta.cdf(x_range, control_a, control_b)
treatment_cdf = stats.beta.cdf(x_range, treatment_a, treatment_b)
plt.plot(x_range, control_cdf,label='Control group')
plt.plot(x_range, treatment_cdf,label='Treatment group')
plt.legend()
plt.savefig('./2acdf.png')
plt.show()

'''
b)
'''
p_control = stats.beta.rvs(control_a, control_b, size=100000)
p_treatment = stats.beta.rvs(treatment_a, treatment_b, size=100000)
odd_ratio = (p_treatment/(1-p_treatment))/(p_control/(1-p_control))

plt.hist(odd_ratio, alpha=0.5, bins=40, ec='white',color='grey')
plt.savefig('./2ahist.png')
plt.show()
'''
intervals
'''

mean = np.mean(odd_ratio)
print('mean',mean)
print('95% Intervals', np.percentile(odd_ratio, 2.5), np.percentile(odd_ratio, 97.5))
print('90% Intervals', np.percentile(odd_ratio, 5), np.percentile(odd_ratio, 95))

'''
control_interval=stats.beta.interval(0.95,control_a, control_b)
treatment_interval=stats.beta.interval(0.95,treatment_a, treatment_b)
print('Control group 95% interval',control_interval)
print('Treatment 95% interval',treatment_interval)

print('Control posterior mean: ', np.mean(control_posterior))
print('Treatment posterior mean: ', np.mean(treatment_posterior))'''