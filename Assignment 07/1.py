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
print("variance:", np.var(drowning))
print("standard deviation:", np.std(drowning,ddof=1))
