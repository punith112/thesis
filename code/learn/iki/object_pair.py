import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
import pickle
import itertools
from collections import OrderedDict

class ObjectPairFeatures:
    """
    A Class implementation for extracting object pair features
    from the object attributes database
    """

    def __init__(self, extracted_data_file):

        self.extracted_data_file = extracted_data_file
        self.features = OrderedDict()

        self.objects_in_scene = []
        self.object_dfs = {}
        self.object_pair_dfs = {}
        self.object_pair_feature_gmms = {}

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

    def get_object_dfs(self, objects_in_scene):
        """
        Reads the object attribute DataFrames stored while computing
        single object features

        Parameters
        ----------
        objects_in_scene: List
        List of objects in the training data

        Returns
        -------
        object_dfs: dictionary
        A dictionary with object names as keys and corresponding
        attribute DataFrames as values
        """

        object_dfs = {}

        for obj in objects_in_scene:
            filename = obj + '_' + 'features'
            object_dfs[obj] = pd.read_csv(filename, sep='\t', index_col=0)

        return object_dfs

    def compute_x_diff(self, obj1, obj2):
        """
        Computes difference between the x co-ordinate attribute of
        two objects in all scenes.

        Parameters
        ----------
        obj1, obj2: Strings
        Object names.

        obj1_df, obj2_df: Pandas DataFrame
        DataFrame containing all attributes of the objects in all scenes.

        Returns
        -------
        diff_series: Pandas Series
        A Pandas Series containing the x differential between the two objects
        from all scenes.
        """

        x_diff_series = pd.Series()
        x_diff_series = self.object_dfs[obj1][obj1 + '_x'] - self.object_dfs[obj2][obj2 + '_x']

        return x_diff_series

    def compute_y_diff(self, obj1, obj2):
        """
        Computes difference between the y co-ordinate attribute of
        two objects in all scenes.

        Parameters
        ----------
        obj1, obj2: Strings
        Object names.

        obj1_df, obj2_df: Pandas DataFrame
        DataFrame containing all attributes of the objects in all scenes.

        Returns
        -------
        diff_series: Pandas Series
        A Pandas Series containing the y differential between the two objects
        from all scenes.
        """

        y_diff_series = pd.Series()
        y_diff_series = self.object_dfs[obj1][obj1 + '_y'] - self.object_dfs[obj2][obj2 + '_y']

        return y_diff_series

    def compute_z_diff(self, obj1, obj2):
        """
        Computes difference between the z co-ordinate attribute of
        two objects in all scenes.

        Parameters
        ----------
        obj1, obj2: Strings
        Object names.

        obj1_df, obj2_df: Pandas DataFrame
        DataFrame containing all attributes of the objects in all scenes.

        Returns
        -------
        diff_series: Pandas Series
        A Pandas Series containing the z differential between the two objects
        from all scenes.
        """

        z_diff_series = pd.Series()
        z_diff_series = self.object_dfs[obj1][obj1 + '_z'] - self.object_dfs[obj2][obj2 + '_z']

        return z_diff_series

    def compute_length_ratio(self, obj1, obj2):
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
        length_ratio_series = self.object_dfs[obj1][obj1 + '_length'] / self.object_dfs[obj2][obj2 + '_length']

        return length_ratio_series

    def compute_width_ratio(self, obj1, obj2):
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
        width_ratio_series = self.object_dfs[obj1][obj1 + '_width'] / self.object_dfs[obj2][obj2 + '_width']

        return width_ratio_series

    def compute_height_ratio(self, obj1, obj2):
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
        height_ratio_series = self.object_dfs[obj1][obj1 + '_height'] / self.object_dfs[obj2][obj2 + '_height']

        return height_ratio_series


    def extract_object_pair_features(self, obj1, obj2, features):
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
            feature_set[obj1 + '_' + obj2 + '_' + key] = value(obj1, obj2)

        return feature_set

    def fit_gmm(self, obj1, obj2, obj_pair_df):
        """
        Fits an n-component Gaussian Mixture to given data.
        Uses AIC or BIC criteria (to be modified within the function)
        to optimise number of components between 1-10.

        Parameters
        ----------
        obj1, obj2: Strings
        Object names.

        obj_pair_df: Pandas DataFrame
        Set of features vectors to which the GMM is to be fit.

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
            models[i].fit(obj_pair_df)

        best_aic, best_model_aic = min((models[i].aic(obj_pair_df), models[i]) for i in range(len(models)))
        best_bic, best_model_bic = min((models[i].bic(obj_pair_df), models[i]) for i in range(len(models)))

        clf = best_model_aic
        # clf = best_model_bic

        if clf.converged_ == True:
            print("Objects: {}, {}. GMM has converged!".format(obj1, obj2))
        else:
            print("Object: {}, {}. Warning! GMM hasn't converged!".format(obj1, obj2))

        param_series['n_components'] = clf.n_components
        param_series['weights'] = clf.weights_
        param_series['means'] = clf.means_
        param_series['covars'] = clf.covariances_

        return clf, param_series


    def get_gmm_params(self):
        """
        Computes the feature vectors for each object pair as per the
        feature computation methods specified inside the method and
        segregates them into separate Pandas DataFrames.

        Fits GMMs to these segregated DataFrames and stores the results
        in the object_pair_feature_gmms dictionary.
        """

        # Get the names of the objects in the training data
        self.objects_in_scene = self.get_objects_in_scene()
        self.object_dfs = self.get_object_dfs(self.objects_in_scene)

        # Ordered Dictionary that maps the name of the feature to its
        # computation method
        self.features['x_diff'] = self.compute_x_diff
        self.features['y_diff'] = self.compute_y_diff
        self.features['z_diff'] = self.compute_z_diff
        self.features['length_ratio'] = self.compute_length_ratio
        self.features['width_ratio'] = self.compute_width_ratio
        self.features['height_ratio'] = self.compute_height_ratio

        # Iterating over each object pair
        for pair in itertools.combinations(self.objects_in_scene, 2):
            obj1 = pair[0]
            obj2 = pair[1]
            object_pair = obj1 + '_' + obj2

            # Extract feature set for the object pair from all scenes and
            # write to file
            self.object_pair_dfs[object_pair] = self.extract_object_pair_features(obj1, obj2, self.features)
            filename = obj1 + '_' + obj2 + '_' + 'features'
            self.object_pair_dfs[object_pair].to_csv(filename, sep = '\t')

            # Fit GMMs for feature set of each object pair and store them in a dictionary
            gmm, param_series = self.fit_gmm(obj1, obj2, self.object_pair_dfs[object_pair])
            self.object_pair_feature_gmms[object_pair] = {}
            self.object_pair_feature_gmms[object_pair]['gmm'] = gmm
            self.object_pair_feature_gmms[object_pair]['params'] = param_series

        return self.object_pair_feature_gmms
