# An example in which data is generated from two different
# normal distributions and then later, a single normal distribution
# is fitted to this data. As expected, the final distribution that is
# fitted is a bad fit to the data. A mixture of distributions is a better option.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generating random data using a normal distribution
mu_one = 0
sigma_one = 10

mu_two = 60
sigma_two = 5

data_one = np.random.normal(mu_one, sigma_one, 1000)
data_two = np.random.normal(mu_two, sigma_two, 1000)

data = np.concatenate((data_one, data_two), axis=0)

# Plotting a normalised histogram of the data (bins are 10 by default)
count, bins, patches = plt.hist(data, alpha=0.6, density=True, color='b')

# Using this data to fit a new normal distribution
mu_new, sigma_new = norm.fit(data)

# Plotting the fitted normal distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)


pdf_new = norm.pdf(x, mu_new, sigma_new)
plt.plot(x, pdf_new, 'r', linewidth=2)
