import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture

my_data = np.genfromtxt('mouse_features.csv', delimiter=',')

X = my_data[:, 0].reshape(-1, 1)

# n, bins, patches = plt.hist(X, bins=100)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('My Very Own Histogram')
# plt.xlim(-1000505, 5)
# plt.show()

clf = mixture.GaussianMixture(n_components=3, covariance_type="full")
clf.fit(X)

X_test = np.linspace(-3, 3, 400).reshape(-1, 1)

log_prob = clf.score_samples(X_test)
pdf = np.exp(log_prob)
plt.plot(X_test, pdf)
plt.show()
