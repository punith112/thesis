# An example showing how to fit 1D Guassian mixtures to data
# and also to select the optimal number of components in the mixtures
# using AIC and BIC model selection criteria

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

models = []

for i in range(10):
    models.append(mixture.GaussianMixture(n_components=i+1, covariance_type='full'))
    models[i].fit(data)

best_aic, best_model_aic = min((models[i].aic(data), models[i]) for i in range(len(models)))
best_bic, best_model_bic = min((models[i].bic(data), models[i]) for i in range(len(models)))


# # Fitting a 2 component Gaussian Mixture
# clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
# clf.fit(data)

# Plotting the Guassian Mixture curve
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
x = x.reshape(-1, 1)
pdf = np.exp(best_model_aic.score_samples(x))
plt.plot(x, pdf, 'r', linewidth=1)
