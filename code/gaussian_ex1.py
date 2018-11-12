# An example that generates data from two separate 2D Gaussians
# and then fits an n-component Gaussian mixture
# (the number of components are optimised using AIC and BIC model selection criteria)
# and then plots the mixture as a 2D contour plot and a 3D surface plot

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn import mixture
from matplotlib.colors import LogNorm
from mpl_toolkits import mplot3d

np.random.seed(0)

# Generating random data using two separate 2D Gaussian distributions
mean_one = np.array([1, 1])
cov_one = np.array([[1, 0.5], [0.5, 1]])

mean_two = np.array([5, 3])
cov_two = np.array([[1, -0.8], [-0.8, 1]])

data_one = np.random.multivariate_normal(mean_one, cov_one, 100)
data_two = np.random.multivariate_normal(mean_two, cov_two, 100)

data = np.concatenate((data_one, data_two), axis=0)

plt.plot(data[:, 0], data[:, 1], 'x')

# Fitting an n-component Gaussian mixture
# Using AIC and BIC criteria to optimise number of components
models = []

for i in range(10):
    models.append(mixture.GaussianMixture(n_components=i+1, covariance_type='full'))
    models[i].fit(data)

best_aic, best_model_aic = min((models[i].aic(data), models[i]) for i in range(len(models)))
best_bic, best_model_bic = min((models[i].bic(data), models[i]) for i in range(len(models)))

clf = best_model_aic

# # Generating the Guassian distribution's contour plot
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
xmin = xmin - 5
xmax = xmax + 5
ymin = ymin - 5
ymax = ymax + 5
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

plt.xlim(-1, 7)
plt.ylim(-1, 7)

# 3D plot of the Gaussian (computationally heavy)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# # ax.contour3D(X, Y, Z, 50, cmap='Greens')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
