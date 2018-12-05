import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
import pickle
import itertools
from collections import OrderedDict

# Constants
MAX_GMM_COMPONENTS=10

class SingleObjectWrapper:

    def __init__(self, objects_in_scenes, object_attributes_file, features):
        """
        A Class implementation for extracting the single object features
        from the object attributes database
        """

        self.objects_in_scenes = objects_in_scenes
        self.object_attributes_df = pd.read_csv(object_attributes_file, sep = '\t', index_col=0)
        self.features = features

        self.single_object_frequencies = {}
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
        self.objects_in_scenes.sort()

        for obj in self.objects_in_scenes:
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

    def get_single_object_frequencies(self, scene_list):

        for obj in self.objects_in_scenes:
            self.single_object_frequencies[obj] = 0

        for scene in scene_list:
            for key in scene.keys():
                if key in self.objects_in_scenes:
                    self.single_object_frequencies[key] = self.single_object_frequencies[key] + 1

        return self.single_object_frequencies
