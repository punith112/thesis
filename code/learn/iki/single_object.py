import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
import pickle
import itertools
from collections import OrderedDict

# Constants
MAX_GMM_COMPONENTS=2

class SingleObjectWrapper:

    def __init__(self, objects_in_scene, object_attributes_file, features):

        self.objects_in_scene = objects_in_scene
        self.object_attributes_df = pd.read_csv(object_attributes_file, sep = '\t', index_col=0)
        self.features = features

        self.single_object_dfs = {}
        self.single_obj_feature_gmms = {}


    def get_single_obj_features(self, obj):
        """
        Extract feature vectors for an object from all scenes.

        Parameters
        ----------
        obj: String
        Object name

        Returns
        -------
        df: Pandas DataFrame
        Reduced DataFrame with required feature vectors from all scenes
        """

        columns = []

        for i in range(len(self.features)):
            columns.append(obj + '_' + self.features[i])

        obj_df = self.object_attributes_df[columns]

        return obj_df


    def fit_gmm(self, obj, obj_df):
        """
        Fits an n-component Gaussian Mixture to given data.
        Uses AIC or BIC criteria (to be modified within the function)
        to optimise number of components between 1-10.

        Parameters
        ----------
        obj: String
        Object name

        obj_df: Pandas DataFrame
        Set of features vectors to which the GMM is fit

        Returns
        -------
        clf: Gaussian Mixture Model
        The GMM model that has been fitted to the data

        param_series: Pandas Series
        Pandas Series that has the fitted GMM parameters
        (n-components, weights, means, covariances)
        """

        models = []
        param_series = pd.Series()

        for i in range(MAX_GMM_COMPONENTS):
            models.append(mixture.GaussianMixture(n_components=i+1, covariance_type='full'))
            models[i].fit(obj_df)

        best_aic, best_model_aic = min((models[i].aic(obj_df), models[i]) for i in range(len(models)))
        best_bic, best_model_bic = min((models[i].bic(obj_df), models[i]) for i in range(len(models)))

        clf = best_model_aic
        # clf = best_model_bic

        if clf.converged_ == True:
            print("Object: {}. GMM has converged!".format(obj))
        else:
            print("Object: {}. Warning! GMM hasn't converged!".format(obj))


        param_series['n_components'] = clf.n_components
        param_series['weights'] = clf.weights_
        param_series['means'] = clf.means_
        param_series['covars'] = clf.covariances_

        return clf, param_series

    def get_gmm_params(self):
        """
        Extracts the feature vectors for each object from the
        object attributes database and segregates them into separate
        Pandas DataFrames.

        Fits GMMs to these segregated DataFrames and stores the results
        in the single_obj_feature_gmms dictionary.
        """

        # Sort the names of the objects in the training data
        self.objects_in_scene.sort()

        for obj in self.objects_in_scene:
            # Extract features of each object from attributes databse
            self.single_object_dfs[obj] = self.get_single_obj_features(obj)
            filename = obj + '_' + 'features'
            self.single_object_dfs[obj].to_csv(filename, sep='\t')

            # Fit GMMs for feature set of each object and store them in a dictionary
            gmm, param_series = self.fit_gmm(obj, self.single_object_dfs[obj])
            self.single_obj_feature_gmms[obj] = {}
            self.single_obj_feature_gmms[obj]['gmm'] = gmm
            self.single_obj_feature_gmms[obj]['params'] = param_series

        return self.single_obj_feature_gmms

if __name__ == '__main__':
    test = SingleObjectFeatures()

# print(222)
# # Extract all objects present in the scenes
# with open("extracted_data.txt", "rb") as myFile:
#     main_list = pickle.load(myFile)
#
# objects_in_scene = []
#
# for i in range(len(main_list)):
#     objects_in_scene = list(set().union(objects_in_scene, main_list[i].keys()))
#
# objects_in_scene.remove('file')
#
# features = ['x', 'y', 'z', 'length', 'width', 'height']
#
# # Load the database, specify the features to be extracted
# main_df = pd.read_csv('database', sep = '\t', index_col=0)
#
# test = SingleObjectFeatures(objects_in_scene, main_df, features)




# indices = ['table_1', 'table_2', 'table_3']
#
# monitor_x = pd.Series([1, 2, 3], index=indices)
# monitor_y = pd.Series([2, 4, 6], index=indices)
# monitor_z = pd.Series([3, 6, 9], index=indices)
# monitor_length = pd.Series([10, 20, 30], index=indices)
# monitor_width = pd.Series([10, 20, 30], index=indices)
# monitor_height = pd.Series([10.0, 20.0, 30.0], index=indices)
#
# monitor_df = pd.DataFrame({'monitor_x': monitor_x, 'monitor_y': monitor_y, 'monitor_z': monitor_z,
#                             'monitor_length': monitor_length, 'monitor_width': monitor_width, 'monitor_height': monitor_height})
#
# keyboard_x = pd.Series([1, 2, 3], index=indices)
# keyboard_y = pd.Series([-2.0, -4.0, -6.0], index=indices)
# keyboard_z = pd.Series([-3, -6, -9], index=indices)
# keyboard_length = pd.Series([10, 20, 30], index=indices)
# keyboard_width = pd.Series([0, 0, 0], index=indices)
# keyboard_height = pd.Series([10.1, 20.2, 30.3], index=indices)
#
# keyboard_df = pd.DataFrame({'keyboard_x': keyboard_x, 'keyboard_y': keyboard_y, 'keyboard_z': keyboard_z,
#                             'keyboard_length': keyboard_length, 'keyboard_width': keyboard_width, 'keyboard_height': keyboard_height})
#
# objects_in_scene = ['monitor', 'keyboard']
# main_df = pd.concat([monitor_df, keyboard_df], axis=1)
# features = ['x', 'y', 'z']
#
# test = SingleObjectFeatures(objects_in_scene, main_df, features)
