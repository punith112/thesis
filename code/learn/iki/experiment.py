import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn import mixture

# # Generating random data using a normal distribution
# mu = 0
# sigma = 10
#
# data = np.random.normal(mu, sigma, 1000)
# data = np.array([0.504, 0.508, 0.502, 0.480, 0.479, 0.503, 0.502, 0.500])
# data = data.reshape(-1, 1)

# data = np.array([[0.504, 0.508, 0.502, 0.480, 0.479, 0.503, 0.502, 0.500],
#                  [0.291, 0.280, 0.278, 0.308, 0.303, 0.289, 0.288, 0.272]])

# data = np.array([[0.504, 0.291],
#                  [0.508, 0.280],
#                  [0.502, 0.278],
#                  [0.480, 0.308],
#                  [0.479, 0.303],
#                  [0.503, 0.289],
#                  [0.502, 0.288],
#                  [0.500, 0.272]])

data = np.array([[0.504, 0.291, 0],
                 [0.508, 0.280, 0],
                 [0.502, 0.278, 0],
                 [0.480, 0.308, 0],
                 [0.479, 0.303, 0],
                 [0.503, 0.289, 0],
                 [0.502, 0.288, 0],
                 [0.500, 0.272, 0]])

clf = mixture.GaussianMixture(n_components=1, covariance_type='full')
clf.fit(data)


# # Plotting a normalised histogram of the data (bins are 10 by default)
# count, bins, patches = plt.hist(data, alpha=0.6, density=True, color='b')
#
# # Using this data to fit a new normal distribution, to see if
# # the mean and sigma values differ from that of the original distribution
#
# mu_new, sigma_new = norm.fit(data)
#
# # Plotting the original normal distribution and the fitted normal distribution
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# #
# # pdf_original = norm.pdf(x, mu, sigma)
# pdf_new = norm.pdf(x, mu_new, sigma_new)
# #
# # plt.plot(x, pdf_original, 'k', linewidth=2)
# plt.plot(x, pdf_new, 'r', linewidth=2)
#
# plt.show(block=False)
