import numpy as np
import matplotlib.pyplot as plt

my_data = np.genfromtxt('mouse_features.csv', delimiter = ',')
#
# np.random.seed(444)
# np.set_printoptions(precision=3)
#
# d = np.random.laplace(loc=15, scale=3, size=500)
#
# # hist, bin_edges = np.histogram(my_data[:,0])
#
n, bins, patches = plt.hist(my_data[:,0], bins=200)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Value')
plt.ylabel('Frequency')
# plt.title('My Very Own Histogram')
# plt.xlim(-1000505, 5)
# plt.show()

# x = np.random.random_integers(1, 100000, 10000)
# x = [1, 2, 2, 3, 4, 5, 6, 7, 8, 10]
# n, bins, patches = plt.hist(x, bins=9, rwidth=0.5)
# plt.ylabel('No of times')
# plt.show()
