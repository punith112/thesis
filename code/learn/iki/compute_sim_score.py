import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
import pickle
import itertools
from collections import OrderedDict
import copy

THRESHOLD = 15

class SimScoreComputer:
    """
    A class implementation for computing similarity scores of
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

        # self.anomalous = 0

    # def check_for_objects(self, test_single_object_dfs):
    #
    #     training_set = self.single_object_frequencies.keys()
    #     test_set = test_single_object_dfs.keys()
    #     print(test_set)
    #
    #
    #     new_objects = test_set -(training_set & test_set)
    #     absent_objects = training_set - (training_set & test_set)
    #
    #     if new_objects:
    #         self.anomalous = 1
    #         print("Anomaly detected! New objects detected!")
    #         print("These objects from the test scene were not present in the training scenes: {}".format(new_objects))
    #
    #     if absent_objects:
    #         self.anomalous = 1
    #         print("Anomaly detected! Objects missing!")
    #         print("These objects from the training scene are absent in the test scene: {}".format(absent_objects))
    #
    #     return self.anomalous


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

        temp_single_object_dfs = copy.deepcopy(test_single_object_dfs)

        for obj in self.single_object_frequencies:
            object_frequency = self.single_object_frequencies[obj]
            object_gmm = self.single_object_gmms[obj]['gmm']

            temp_single_object_dfs[obj][obj + '_' + 'sim_scores'] = (object_frequency *
                                                         object_gmm.score_samples(test_single_object_dfs[obj])
                                                         / self.number_of_training_scenes)

        return temp_single_object_dfs

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

        temp_object_pair_dfs = copy.deepcopy(test_object_pair_dfs)

        for object_pair in self.object_pair_frequencies:
            object_pair_frequency = self.object_pair_frequencies[object_pair]
            object_pair_gmm = self.object_pair_gmms[object_pair]['gmm']

            temp_object_pair_dfs[object_pair][object_pair + '_' + 'sim_scores'] = \
                                                               (object_pair_frequency *
                                                               object_pair_gmm.score_samples(test_object_pair_dfs[object_pair])
                                                               / self.number_of_training_scenes)

        return temp_object_pair_dfs

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
            overall_results = pd.concat([overall_results, single_object_results[obj][obj + '_' + 'sim_scores']], axis=1)
            # import ipdb; ipdb.set_trace()
            if overall_sim_scores.empty:
                overall_sim_scores = single_object_results[obj][obj + '_' + 'sim_scores']
            else:
                overall_sim_scores = overall_sim_scores + single_object_results[obj][obj + '_' + 'sim_scores']

            # for index, row in single_object_results[obj].iterrows():
            #     if row[obj + '_' + 'sim_scores'] < THRESHOLD:
            #         print("Anomaly detected! There is something wrong with the {} in {}!".format(obj, index))

        for object_pair in test_object_pair_dfs:
            overall_results = pd.concat([overall_results, test_object_pair_dfs[object_pair]], axis=1)
            overall_results = pd.concat([overall_results, object_pair_results[object_pair][object_pair + '_' + 'sim_scores']], axis=1)
            overall_sim_scores = overall_sim_scores + object_pair_results[object_pair][object_pair + '_' + 'sim_scores']

            # for index, row in object_pair_results[object_pair].iterrows():
            #     if row[object_pair + '_' + 'sim_scores'] < THRESHOLD:
            #         print("Anomaly detected! There is something wrong with the {} "
            #               "relation in {}!".format(object_pair, index))

        overall_results['sim_scores'] = overall_sim_scores

        anomalous = 0

        for index, row in overall_results.iterrows():

            print(index)
            print("--------------")
            if row['sim_scores'] < 120:
                anomalous = 1
                print("There is something wrong with {}!".format(index))

            for obj in test_single_object_dfs:
                if row[obj + '_' + 'sim_scores'] < 20:
                    anomalous = 1
                    print("Anomaly detected! There is something wrong with the {} in {}!".format(obj, index))

            for object_pair in test_object_pair_dfs:
                if row[object_pair + '_' + 'sim_scores'] < 20:
                    anomalous = 1
                    print("Anomaly detected! There is something wrong with the {} "
                          "relation in {}!".format(object_pair, index))

            if anomalous == 0:
                print("{} looks okay!".format(index))

            anomalous = 0

            print("")

        return single_object_results, object_pair_results, overall_results
