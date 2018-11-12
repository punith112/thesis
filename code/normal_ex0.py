import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generating random data using a normal distribution
mu = 0
sigma = 10

data = np.random.normal(mu, sigma, 1000)

# Plotting a normalised histogram of the data (bins are 10 by default)
count, bins, patches = plt.hist(data, alpha=0.6, density=True, color='b')

# Using this data to fit a new normal distribution, to see if
# the mean and sigma values differ from that of the original distribution
mu_new, sigma_new = norm.fit(data)

# Plotting the original normal distribution and the fitted normal distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)

pdf_original = norm.pdf(x, mu, sigma)
pdf_new = norm.pdf(x, mu_new, sigma_new)

plt.plot(x, pdf_original, 'k', linewidth=2)
plt.plot(x, pdf_new, 'r', linewidth=2)
