import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn import mixture

# Generating random data using a normal distribution
mu_one = 0
sigma_one = 10

mu_two = 60
sigma_two = 5

data_one = np.random.normal(mu_one, sigma_one, 1000)
data_two = np.random.normal(mu_two, sigma_two, 1000)

data = np.concatenate((data_one, data_two), axis=0)
data = data.reshape(-1, 1)

# Plotting a normalised histogram of the data (bins are 10 by default)
count, bins, patches = plt.hist(data, alpha=0.6, density=True, color='b')

clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(data)

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
x = x.reshape(-1, 1)
pdf = np.exp(clf.score_samples(x))
plt.plot(x, pdf, 'r', linewidth=1)
