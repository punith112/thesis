import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
import pickle
import itertools
from collections import OrderedDict

class SimScoreComputer:
    """
    A class implemntation for computing similarity scores of
    test scenes with respect to the training scenes.
    """

    def __init__(self, single_object_frequencies, single_object_gmms, object_pair_frequencies,
                    object_pair_gmms, number_of_training_scenes):
        """
        Instantiation

        Parameters
        ----------
        single_object_frequencies: Dict
        Dictionary with object names as keys and their frequency of occurence in
        the training dataset as the corresponding values.

        single_object_gmms: Dict
        Dictionary that has object names as keys and the corresponding Gaussian
        Mixture Models from the training dataset and their parameters as values.

        object_pair_frequencies: Dict
        Dictionary with object pair names as keys and their frequency of occurence
        in the training dataset as the corresponding values.

        object_pair_gmms: Dict
        Dictionary that has object pair names as keys and the corresponding Gaussian
        Mixture Models from the training dataset and their parameters as values.

        number of training_scenes: Int
        Number of scenes in the training data
        """

        self.single_object_frequencies = single_object_frequencies
        self.single_object_gmms = single_object_gmms
        self.object_pair_frequencies = object_pair_frequencies
        self.object_pair_gmms = object_pair_gmms
        self.number_of_training_scenes = number_of_training_scenes

        self.single_object_sim_score = 0
        self.object_pair_sim_score = 0
        self.total_sim_score = 0


    def compute_single_object_sim_score(self, test_single_object_dfs):
        """
        Method that computes similarity scores of each individual object
        in the test scenes with respect to the training data.

        Parameters
        ----------
        test_single_object_dfs: Pandas DataFrame
        DataFrame that contains all the single object features from all the
        scenes in the test data.

        Returns
        -------
        test_single_object_dfs: Pandas DataFrame
        DataFrame that contains an additional column of 'sim_scores' compared to
        the 'test_single_object_dfs' DataFrame, indicating similarity score of the
        single object features of each scene with respect to the training data.
        """

        for obj in self.single_object_frequencies:
            object_frequency = self.single_object_frequencies[obj]
            object_gmm = self.single_object_gmms[obj]['gmm']

            test_single_object_dfs[obj]['sim_scores'] = (object_frequency * object_gmm.score_samples(test_single_object_dfs[obj]) / self.number_of_training_scenes)

        return test_single_object_dfs

    def compute_object_pair_sim_score(self, test_object_pair_dfs):
        """
        Method that computes similarity scores of each object pair
        in the test scenes with respect to the training data.

        Parameters
        ----------
        test_object_pair_dfs: Pandas DataFrame
        DataFrame that contains all the object pair features from all the
        scenes in the test data.

        Returns
        -------
        test_object_pair_dfs: Pandas DataFrame
        DataFrame that contains an additional column of 'sim_scores' compared to
        the 'test_object_pair_dfs' DataFrame, indicating similarity score of the
        object pair features of each scene with respect to the training data.
        """

        for object_pair in self.object_pair_frequencies:
            object_pair_frequency = self.object_pair_frequencies[object_pair]
            object_pair_gmm = self.object_pair_gmms[object_pair]['gmm']

            test_object_pair_dfs[object_pair]['sim_scores'] = (object_pair_frequency * object_pair_gmm.score_samples(test_object_pair_dfs[object_pair]) / self.number_of_training_scenes)

        return test_object_pair_dfs

    def compute_overall_sim_score(self, test_single_object_dfs, test_object_pair_dfs):
        """
        Method that computes similarity scores of each scene
        in the test dataset with respect to the training data.

        Parameters
        ----------
        test_single_object_dfs: Pandas DataFrame
        DataFrame that contains all the single object features from all the
        scenes in the test data.

        test_object_pair_dfs: Pandas DataFrame
        DataFrame that contains all the object pair features from all the
        scenes in the test data.

        Returns
        -------
        single_object_results: Pandas DataFrame
        DataFrame that contains an additional column of 'sim_scores' compared to
        the 'test_single_object_dfs' DataFrame, indicating similarity score of the
        single object features of each scene with respect to the training data.

        object_pair_results: Pandas DataFrame
        DataFrame that contains an additional column of 'sim_scores' compared to
        the 'test_object_pair_dfs' DataFrame, indicating similarity score of the
        object pair features of each scene with respect to the training data.

        overall_results: Pandas DataFrame
        DataFrame that combines the 'test_single_object_dfs' DataFrame and the
        'test_object_pair_dfs' DataFrame with an additional column of 'sim_scores',
        indicating similarity score of each scene with respect to the training data.
        """

        single_object_results = self.compute_single_object_sim_score(test_single_object_dfs)
        object_pair_results = self.compute_object_pair_sim_score(test_object_pair_dfs)

        overall_results = pd.DataFrame()
        overall_sim_scores = pd.Series()

        for obj in test_single_object_dfs:
            overall_results = pd.concat([overall_results, test_single_object_dfs[obj]], axis=1)
            if overall_sim_scores.empty:
                overall_sim_scores = single_object_results[obj]['sim_scores']
            else:
                overall_sim_scores = overall_sim_scores + single_object_results[obj]['sim_scores']

        for object_pair in test_object_pair_dfs:
            overall_results = pd.concat([overall_results, test_object_pair_dfs[object_pair]], axis=1)
            overall_sim_scores = overall_sim_scores + object_pair_results[object_pair]['sim_scores']

        overall_results['sim_scores'] = overall_sim_scores

        return single_object_results, object_pair_results, overall_results
