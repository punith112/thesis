import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture

# Function that fits an n-component GMM to a column in a dataframe
# and returns the column in an np array format and the scores and also
# a set of points for plotting th GMM
def generate_gmm(df, col, n_components):
    data = df[col]
    data = data.values.reshape(-1, 1)

    clf = mixture.GaussianMixture(n_components=n_components, covariance_type='full')
    clf.fit(data)

    xmin, xmax = clf.means_[0, 0] - 5 * np.sqrt(clf.covariances_[0, 0, 0]), clf.means_[0, 0] + 5 * np.sqrt(clf.covariances_[0, 0, 0])
    x = np.linspace(xmin, xmax, 100)
    x = x.reshape(-1, 1)

    scores = np.exp(clf.score_samples(x))

    return data, x, scores


main_df = pd.read_csv('database', sep = '\t', index_col=0)

plt.figure(1)
spoon_x, x, scores_x = generate_gmm(main_df, 'spoon_x', 1)
hist = plt.hist(spoon_x, density=False, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='b')
plt.plot(x, scores_x, '-g')

plt.figure(2)
spoon_y, y, scores_y = generate_gmm(main_df, 'spoon_y', 1)
hist = plt.hist(spoon_y, density=False, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='b')
plt.plot(y, scores_y, '-g')

## For quick generation of plots of many columns of data
# plt.figure(1)
# data, x, scores = generate_gmm(main_df, 'spoon_x', 1)
# hist = plt.hist(data, density=False, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='b')
# plt.plot(x, scores, '-g')
#
# plt.figure(2)
# data, x, scores = generate_gmm(main_df, 'spoon_y', 1)
# hist = plt.hist(data, density=False, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='b')
# plt.plot(x, scores, '-g')

plt.show(block=False)
