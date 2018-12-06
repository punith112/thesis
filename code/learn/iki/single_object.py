import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
import pickle
import itertools
from collections import OrderedDict

# Constants
MAX_GMM_COMPONENTS=10
DATA_DUMP_FOLDER='data/'

class SingleObjectWrapper:
    """
    A Class implementation for extracting the single object features
    from the object attributes database
    """

    def __init__(self, objects_in_scenes, object_attributes_file):
        """
        Instantiation

        Parameters
        ----------
        objects_in_scenes: List
        List of all objects present in the dataset

        object_attributes_file: String
        File path to the file containing the Pandas DataFrame with data
        of all the attributes of all the objects in all scenes
        """

        self.objects_in_scenes = objects_in_scenes
        self.object_attributes_df = pd.read_csv(object_attributes_file, sep = '\t', index_col=0)
        self.features = OrderedDict()

        self.single_object_frequencies = {}
        self.single_object_dfs = {}
        self.single_object_feature_gmms = {}

    def compute_centroid_x(self, obj):
        """
        Computes the x-cordinate of the centroid of an object
        in every scene and builds a Pandas Series with the data.

        Parameters
        ----------
        obj: String
        Object name

        Returns
        -------
        centroid_x_series: Pandas Series
        Pandas Series with scene names as row labels and the x-cordinates
        of the centroids of the object from all scenes in the column.
        """

        centroid_x_series = pd.Series()
        centroid_x_series = self.object_attributes_df[obj + '_x'] + self.object_attributes_df[obj + '_length'] / 2

        return centroid_x_series

    def compute_centroid_y(self, obj):
        """
        Computes the y-cordinate of the centroid of an object
        in every scene and builds a Pandas Series with the data.

        Parameters
        ----------
        obj: String
        Object name

        Returns
        -------
        centroid_y_series: Pandas Series
        Pandas Series with scene names as row labels and the y-cordinates
        of the centroids of the object from all scenes in the column.
        """

        centroid_y_series = pd.Series()
        centroid_y_series = self.object_attributes_df[obj + '_y'] + self.object_attributes_df[obj + '_width'] / 2

        return centroid_y_series

    def compute_centroid_z(self, obj):
        """
        Computes the z-cordinate of the centroid of an object
        in every scene and builds a Pandas Series with the data.

        Parameters
        ----------
        obj: String
        Object name

        Returns
        -------
        centroid_z_series: Pandas Series
        Pandas Series with scene names as row labels and the z-cordinates
        of the centroids of the object from all scenes in the column.
        """

        centroid_z_series = pd.Series()
        centroid_z_series = self.object_attributes_df[obj + '_z'] + self.object_attributes_df[obj + '_height'] / 2

        return centroid_z_series

    def compute_length(self, obj):
        """
        Computes the length of an object in every scene, which
        is basically grabbing the length attribute from the main
        DataFrame file and builds a Pandas Series with the data.

        Parameters
        ----------
        obj: String
        Object name

        Returns
        -------
        length_series: Pandas Series
        Pandas Series with scene names as row labels and the lengths
        of the object from all scenes in the column.
        """

        length_series = pd.Series()
        length_series = self.object_attributes_df[obj + '_length']

        return length_series

    def compute_width(self, obj):
        """
        Computes the width of an object in every scene, which
        is basically grabbing the width attribute from the main
        DataFrame file and builds a Pandas Series with the data.

        Parameters
        ----------
        obj: String
        Object name

        Returns
        -------
        width_series: Pandas Series
        Pandas Series with scene names as row labels and the widths
        of the object from all scenes in the column.
        """

        width_series = pd.Series()
        width_series = self.object_attributes_df[obj + '_width']

        return width_series

    def compute_height(self, obj):
        """
        Computes the height of an object in every scene, which
        is basically grabbing the height attribute from the main
        DataFrame file and builds a Pandas Series with the data.

        Parameters
        ----------
        obj: String
        Object name

        Returns
        -------
        height_series: Pandas Series
        Pandas Series with scene names as row labels and the heights
        of the object from all scenes in the column.
        """

        height_series = pd.Series()
        height_series = self.object_attributes_df[obj + '_height']

        return height_series

    def extract_single_object_features(self, obj, features):
        """
        Extracts feature vectors of an object from all scenes per
        the specified features.

        Parameters
        ----------
        obj: String
        Object name

        features: OrderedDict
        An ordered dictionary that has the feature names as the keys
        and the feature computation functions as the values.

        Returns
        -------
        feature_set: Pandas DataFrame
        A DataFrame containing the set of feature vectors of the object
        from all scenes.
        """

        feature_set = pd.DataFrame()

        for key, value in features.items():
            feature_set[obj + '_' + key] = value(obj)

        return feature_set


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
        Computes the feature vectors for each object as per the
        feature computation methods specified inside the method and
        segregates them into separate Pandas DataFrames for each object.

        Fits GMMs to these segregated DataFrames and stores the results
        in the single_object_feature_gmms dictionary.

        Returns
        -------
        self.single_object_feature_gmms: Dict
        Dictionary that has object names as keys and the corresponding
        Gaussian Mixture Models and their parameters as values.
        """

        # Sort the names of the objects in the training data
        self.objects_in_scenes.sort()

        # Ordered Dictionary that maps the name of the feature to its
        # computation method
        self.features['centroid_x'] = self.compute_centroid_x
        self.features['centroid_y'] = self.compute_centroid_y
        self.features['centroid_z'] = self.compute_centroid_z
        self.features['length'] = self.compute_length
        self.features['width'] = self.compute_width
        self.features['height'] = self.compute_height

        # Iterating over each object pair
        for obj in self.objects_in_scenes:
            # Extract feature set for the object from all scenes and write to file
            self.single_object_dfs[obj] = self.extract_single_object_features(obj, self.features)
            filename = DATA_DUMP_FOLDER + obj + '_'  + 'features'
            self.single_object_dfs[obj].to_csv(filename, sep='\t')

            # Fit GMMs for feature set of each object and store them in a dictionary
            gmm, param_series = self.fit_gmm(obj, self.single_object_dfs[obj])
            self.single_object_feature_gmms[obj] = {}
            self.single_object_feature_gmms[obj]['gmm'] = gmm
            self.single_object_feature_gmms[obj]['params'] = param_series

        return self.single_object_feature_gmms


    def get_single_object_frequencies(self, scenes_list):
        """
        Computes the number of scenes in which an object is present,
        for every object.

        Parameters
        ----------
        scenes_list: List
        List of dictionaries, each dict representing a scene and all
        the attributes info of all objects in the scene.

        Returns
        -------
        self.single_object_frequencies: Dict
        Dictionary with object names as keys and their frequencies as values.
        """

        # Iterating over each object pair
        for obj in self.objects_in_scenes:
            self.single_object_frequencies[obj] = 0

        for scene in scenes_list:
            for key in scene.keys():
                if key in self.objects_in_scenes:
                    self.single_object_frequencies[key] = self.single_object_frequencies[key] + 1

        return self.single_object_frequencies
