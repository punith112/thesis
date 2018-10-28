# An example that generates data from a 2D Gaussian
# and then fits a new Gaussian to this data and then
# generates a contour plot for the new Gaussian

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn import mixture
from matplotlib.colors import LogNorm

np.random.seed(0)

# Generating random data using a 2D Gaussian distribution
mean = np.array([1, 1])
cov = np.array([[1, 0.5], [0.5, 1]])

data = np.random.multivariate_normal(mean, cov, 50)
plt.plot(data[:, 0], data[:, 1], 'x')

# Fitting a new Gaussian distribution
clf = mixture.GaussianMixture(n_components=1, covariance_type='full')
clf.fit(data)

# # Generating the Guassian distribution's contour plot
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
xmin = 5 * xmin
xmax = 5 * xmax
ymin = 5 * ymin
ymax = 5 * ymax
x = np.linspace(xmin, xmax, 500)
y = np.linspace(ymin, ymax, 500)

X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = np.exp(clf.score_samples(XX))
Z = Z.reshape(X.shape)

# Contour lines and filled contour plot
# CS = plt.contour(X, Y, Z, 20, cmap='winter')
# plt.clabel(CS, inline=True, fontsize=8)
CS = plt.contourf(X, Y, Z, 20, cmap='winter')
plt.colorbar()

# Scatter plot of the original data
plt.scatter(data[:, 0], data[:, 1])

plt.xlim(-1, 3)
plt.ylim(-1, 3)
