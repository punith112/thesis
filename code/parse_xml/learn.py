import numpy as np
import matplotlib.pyplot as plt

my_data = np.genfromtxt('mouse_features.csv', delimiter = ',')

np.random.seed(444)
np.set_printoptions(precision=3)

d = np.random.laplace(loc=15, scale=3, size=500)

# hist, bin_edges = np.histogram(my_data[:,0])

plt.hist(my_data[1:50,0], bins='auto')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('My Very Own Histogram')
plt.show()
