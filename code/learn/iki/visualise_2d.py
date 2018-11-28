import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture

# Function that fits an n-component 2D GMM to a couple of columns in a
# dataframe and returns the required data for plotting the GMM
def generate_gmm(df, cols, n_components):

    data = df.loc[:, cols]

    clf = mixture.GaussianMixture(n_components=n_components, covariance_type='full')
    clf.fit(data)

    xmin, xmax = clf.means_[0, 0] - 5 * np.sqrt(clf.covariances_[0, 0, 0]), clf.means_[0, 0] + 5 * np.sqrt(clf.covariances_[0, 0, 0])
    ymin, ymax = clf.means_[0, 1] - 5 * np.sqrt(clf.covariances_[0, 1, 1]), clf.means_[0, 1] + 5 * np.sqrt(clf.covariances_[0, 1, 1])
    x = np.linspace(xmin, xmax, 500)
    y = np.linspace(ymin, ymax, 500)

    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = np.exp(clf.score_samples(XX))
    Z = Z.reshape(X.shape)

    return X, Y, Z


main_df = pd.read_csv('database', sep = '\t', index_col=0)

plt.figure(1)

# Filled contour plot and lined contour plot
X, Y, Z = generate_gmm(main_df, ['spoon_x', 'spoon_y'], 1)
CS = plt.contourf(X, Y, Z, levels=20, cmap='winter')
plt.colorbar()
# CS = plt.contour(X, Y, Z, levels=20, cmap='winter')
# plt.clabel(CS, inline=True, fontsize=8)

# Scatter plot of the data points
plt.scatter(main_df['spoon_x'], main_df['spoon_y'])

plt.figure(2)

# Filled contour plot and lined contour plot
X, Y, Z = generate_gmm(main_df, ['plate_x', 'plate_y'], 1)
CS = plt.contourf(X, Y, Z, levels=20, cmap='winter')
plt.colorbar()
# CS = plt.contour(X, Y, Z, levels=20, cmap='winter')
# plt.clabel(CS, inline=True, fontsize=8)

# Scatter plot of the data points
plt.scatter(main_df['plate_x'], main_df['plate_y'])

plt.show(block=False)
