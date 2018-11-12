import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture

data = np.genfromtxt('mouse_features.csv', delimiter=',')

# models = []
# aic = []
# bic = []
# 
# for i in range(10):
#     models.append(mixture.GaussianMixture(n_components=i+1, covariance_type='full'))
#     models[i].fit(data)
#     aic.append(models[i].aic(data))
#     bic.append(models[i].bic(data))
#
# best_aic, best_model_aic = min((models[i].aic(data), models[i]) for i in range(len(models)))
# best_bic, best_model_bic = min((models[i].bic(data), models[i]) for i in range(len(models)))
#
# clf = best_model_bic
#
# scores = clf.score_samples(data)
# count, bins, patches = plt.hist(scores, alpha=0.6, density=False, color='b')
#
# for i in range(len(scores)):
#     if scores[i] < -0.7:
#         print(data[i])
