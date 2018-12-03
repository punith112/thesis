import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
import pickle

def extract_single_obj_features(df, obj, features):
    """
    Extract feature vectors for an object from all scenes.

    Parameters
    ----------
    df: Pandas DataFrame
    Main DataFrame that has all the attributes of all objects from all scenes

    obj: String
    Object name

    features: List of Strings
    List of features to be extracted ['x', 'y', 'z']

    Returns
    -------
    df: Pandas DataFrame
    Reduced DataFrame with required feature vectors from all scenes
    """

    columns = []

    for i in range(len(features)):
        columns.append(obj + '_' + features[i])

    df = df[columns]

    return df


def fit_gmm(obj, obj_feature_set):
    """
    Fits an n-component Gaussian Mixture to given data.
    Uses AIC or BIC criteria (to be modified within the function)
    to optimise number of components between 1-10.

    Parameters
    ----------
    obj: String
    Object name

    obj_feature_set: Pandas DataFrame
    Set of features vectors to which the GMM is fit

    Returns
    -------
    param_series: Pandas Series
    Pandas Series that has the fitted GMM parameters
    (n-components, weights, means, covariances)
    """

    models = []
    param_series = pd.Series()

    for i in range(10):
        models.append(mixture.GaussianMixture(n_components=i+1, covariance_type='full'))
        models[i].fit(obj_feature_set)

    best_aic, best_model_aic = min((models[i].aic(obj_feature_set), models[i]) for i in range(len(models)))
    best_bic, best_model_bic = min((models[i].bic(obj_feature_set), models[i]) for i in range(len(models)))

    clf = best_model_aic
    # clf = best_model_bic

    if clf.converged_ == True:
        print("Object: {}. GMM has converged!".format(obj))
    else:
        print("Object: {}. Warning! GMM hasn't converged!".format(obj))

    param_series['n-components'] = clf.n_components
    param_series['weights'] = clf.weights_
    param_series['means'] = clf.means_
    param_series['covars'] = clf.covariances_

    return param_series


# Load the database, specify the features to be extracted
main_df = pd.read_csv('database', sep = '\t', index_col=0)
features = ['x', 'y', 'z', 'length', 'width', 'height']

# Extract all objects present in the scenes
with open("extracted_data.txt", "rb") as myFile:
    main_list = pickle.load(myFile)

objects_in_scene = []

for i in range(len(main_list)):
    objects_in_scene = list(set().union(objects_in_scene, main_list[i].keys()))

objects_in_scene.remove('file')

# Build database of GMM parameters for feature set of each object
gmm_df = pd.DataFrame()

for obj in objects_in_scene:
    feature_set = extract_single_obj_features(main_df, obj, features)
    filename = obj + '_' + 'features'
    feature_set.to_csv(filename, sep='\t')

    param_series = fit_gmm(obj, feature_set)
    gmm_df[obj] = param_series

gmm_df.to_csv('single_obj_features', sep='\t')
