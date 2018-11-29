import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
import pickle
import itertools
from collections import OrderedDict

def compute_x_diff(obj1, obj2, obj1_df, obj2_df):
    """
    Computes difference between x co-oordinates of two objects
    in all scenes.

    Parameters
    ----------
    obj1, obj2: Strings
    Object names.

    obj1_df, obj2_df: Pandas DataFrame
    DataFrame containing all attributes of the objects in all scenes.

    Returns
    -------
    x_diff_series: Pandas Series
    A Pandas Series containing the x differential between the two objects
    from all scenes.
    """

    x_diff_series = pd.Series()
    x_diff_series = obj1_df[obj1 + '_x'] - obj2_df[obj2 + '_x']

    return x_diff_series

def compute_y_diff(obj1, obj2, obj1_df, obj2_df):
    """
    Computes difference between y co-oordinates of two objects
    in all scenes.

    Parameters
    ----------
    obj1, obj2: Strings
    Object names.

    obj1_df, obj2_df: Pandas DataFrame
    DataFrame containing all attributes of the objects in all scenes.

    Returns
    -------
    y_diff_series: Pandas Series
    A Pandas Series containing the y differential between the two objects
    from all scenes.
    """

    y_diff_series = pd.Series()
    y_diff_series = obj1_df[obj1 + '_y'] - obj2_df[obj2 + '_y']

    return y_diff_series

def compute_z_diff(obj1, obj2, obj1_df, obj2_df):
    """
    Computes difference between z co-oordinates of two objects
    in all scenes.

    Parameters
    ----------
    obj1, obj2_df: Strings
    Object names.

    obj1_df, obj: Pandas DataFrame
    DataFrame containing all attributes of the objects in all scenes.

    Returns
    -------
    z_diff_series: Pandas Series
    A Pandas Series containing the z differential between the two objects
    from all scenes.
    """

    z_diff_series = pd.Series()
    z_diff_series = obj1_df[obj1 + '_z'] - obj2_df[obj2 + '_z']

    return z_diff_series

def compute_length_ratio(obj1, obj2, obj1_df, obj2_df):
    """
    Computes ratio of lengths of two objects in all scenes.

    Parameters
    ----------
    obj1, obj2: Strings
    Object names.

    obj1_df, obj2_df: Pandas DataFrame
    DataFrame containing all attributes of the objects in all scenes.

    Returns
    -------
    length_ratio_series: Pandas Series
    A Pandas Series containing the ratio of lengths of the two objects
    from all scenes.
    """

    length_ratio_series = pd.Series()
    length_ratio_series = obj1_df[obj1 + '_length'] / obj2_df[obj2 + '_length']

    return length_ratio_series

def compute_width_ratio(obj1, obj2, obj1_df, obj2_df):
    """
    Computes ratio of widths of two objects in all scenes.

    Parameters
    ----------
    obj1, obj2: Strings
    Object names.

    obj1_df, obj2_df: Pandas DataFrame
    DataFrame containing all attributes of the objects in all scenes.

    Returns
    -------
    width_ratio_series: Pandas Series
    A Pandas Series containing the ratio of widths of the two objects
    from all scenes.
    """

    width_ratio_series = pd.Series()
    width_ratio_series = obj1_df[obj1 + '_width'] / obj2_df[obj2 + '_width']

    return width_ratio_series

def compute_height_ratio(obj1, obj2, obj1_df, obj2_df):
    """
    Computes ratio of heights of two objects in all scenes.

    Parameters
    ----------
    obj1, obj2: Strings
    Object names.

    obj1_df, obj2_df: Pandas DataFrame
    DataFrame containing all attributes of the objects in all scenes.

    Returns
    -------
    height_ratio_series: Pandas Series
    A Pandas Series containing the ratio of heights of the two objects
    from all scenes.
    """

    height_ratio_series = pd.Series()
    height_ratio_series = obj1_df[obj1 + '_height'] / obj2_df[obj2 + '_height']

    return height_ratio_series

def extract_obj_pair_features(obj1, obj2, obj1_df, obj2_df, features):
    """
    Extracts feature vectors of an object pair from all scenes.

    Parameters
    ----------
    obj1, obj2: Strings
    Object names.

    obj1_df, obj2_df: Pandas DataFrame
    DataFrame containing all attributes of the objects in all scenes.

    features: OrderedDict
    An ordered dictionary that has the feature names as the keys
    and the feature computation functions as the values.

    Returns
    -------
    feature_set: Pandas DataFrame
    A DataFrame containing the set of feature vectors of the object pair
    from all scenes.
    """

    feature_set = pd.DataFrame()

    for key, value in features.items():
        feature_set[obj1 + '_' + obj2 + '_' + key] = value(obj1, obj2, obj1_df, obj2_df)

    return feature_set

def fit_gmm(obj1, obj2, obj_pair_feature_set):
    """
    Fits an n-component Gaussian Mixture to given data.
    Uses AIC or BIC criteria (to be modified within the function)
    to optimise number of components between 1-10.

    Parameters
    ----------
    obj1, obj2: Strings
    Object names.

    obj_pair_feature_set: Pandas DataFrame
    Set of features vectors to which the GMM is fit.

    Returns
    -------
    param_series: Pandas Series
    Pandas Series that has the fitted GMM parameters
    (n-components, weights, means, covariances).
    """

    models = []
    param_series = pd.Series()

    for i in range(10):
        models.append(mixture.GaussianMixture(n_components=i+1, covariance_type='full'))
        models[i].fit(obj_pair_feature_set)

    best_aic, best_model_aic = min((models[i].aic(obj_pair_feature_set), models[i]) for i in range(len(models)))
    best_bic, best_model_bic = min((models[i].bic(obj_pair_feature_set), models[i]) for i in range(len(models)))

    clf = best_model_aic
    # clf = best_model_bic

    if clf.converged_ == True:
        print("Objects: {}, {}. GMM has converged!".format(obj1, obj2))
    else:
        print("Object: {}, {}. Warning! GMM hasn't converged!".format(obj1, obj2))

    param_series['n-components'] = clf.n_components
    param_series['weights'] = clf.weights_
    param_series['means'] = clf.means_
    param_series['covars'] = clf.covariances_

    return param_series


# Load the database, specify the features to be extracted
main_df = pd.read_csv('database', sep = '\t', index_col=0)

# Create OrderedDict of features to be extracted with
# functions as values for computation of the features
features = OrderedDict()

features['x_diff'] = compute_x_diff
features['y_diff'] = compute_y_diff
features['z_diff'] = compute_z_diff
features['length_ratio'] = compute_length_ratio
features['width_ratio'] = compute_width_ratio
features['height_ratio'] = compute_height_ratio

# Extract all objects present in the scenes
# and then sort the objects for maintaining
# uniformity in object-pair naming
with open("extracted_data.txt", "rb") as myFile:
    main_list = pickle.load(myFile)

objects_in_scene = []

for i in range(len(main_list)):
    objects_in_scene = list(set().union(objects_in_scene, main_list[i].keys()))

objects_in_scene.remove('file')
objects_in_scene.sort()

# Create DataFrames for attributes of each object from all scenes
df_dict = {}

for obj in objects_in_scene:
    filename = obj + '_' + 'features'
    df_dict[obj] = pd.read_csv(filename, sep='\t', index_col=0)

# Build database of GMM parameters for feature set of each object-pair
gmm_df = pd.DataFrame()

for pair in itertools.combinations(objects_in_scene, 2):
    obj1 = pair[0]
    obj2 = pair[1]
    feature_set = extract_obj_pair_features(obj1, obj2, df_dict[obj1], df_dict[obj2], features)
    param_series = fit_gmm(obj1, obj2, feature_set)
    gmm_df[obj1 + '_' + obj2] = param_series
