import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
import pickle
import itertools
from collections import OrderedDict

class SingleObjectFeatures:
    """
    A Class implementation for extracting single object features
    from the object attributes database
    """

    def __init__(self, extracted_data_file, obj_attributes_file, features):

        self.extracted_data_file = extracted_data_file
        self.obj_attributes_df = pd.read_csv(obj_attributes_file, sep = '\t', index_col=0)
        self.features = features

        self.objects_in_scene = []
        self.single_obj_dfs = {}
        self.single_obj_feature_gmms = {}

    def get_objects_in_scene(self):
        """
        Get the objects from all scenes in the training data

        Returns
        -------
        objects_in_scene: List
        List of objects in the training data
        """

        objects_in_scene = []

        with open(self.extracted_data_file, "rb") as myFile:
            temp_list = pickle.load(myFile)

        for i in range(len(temp_list)):
            objects_in_scene = list(set().union(objects_in_scene, temp_list[i].keys()))

        objects_in_scene.remove('file')
        objects_in_scene.sort()

        return objects_in_scene


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

        obj_df = self.obj_attributes_df[columns]

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

        for i in range(10):
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

        # Get the names of the objects in the training data
        self.objects_in_scene = self.get_objects_in_scene()

        for obj in self.objects_in_scene:
            # Extract features of each object from attributes databse
            self.single_obj_dfs[obj] = self.get_single_obj_features(obj)
            filename = obj + '_' + 'features'
            self.single_obj_dfs[obj].to_csv(filename, sep='\t')

            # Fit GMMs for feature set of each object and store them in a dictionary
            gmm, param_series = self.fit_gmm(obj, self.single_obj_dfs[obj])
            self.single_obj_feature_gmms[obj] = {}
            self.single_obj_feature_gmms[obj]['gmm'] = gmm
            self.single_obj_feature_gmms[obj]['params'] = param_series

        return self.single_obj_feature_gmms
