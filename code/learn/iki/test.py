import pickle
import os
import xml.etree.ElementTree as ET
import pandas as pd
from single_object import SingleObjectWrapper
from object_pair import ObjectPairWrapper
from extract_data import DataExtractor

training_data_path = "/home/iki/catkin_ws/src/thesis/iki_dataset/training_data/"
training_dict_dump_file_name = "training_data/training_data.dict"
training_df_dump_file_name = "training_data/training_data.df"

test_data_path = "/home/iki/catkin_ws/src/thesis/iki_dataset/test_data/"
test_dict_dump_file_name = "test_data/test_data.dict"
test_df_dump_file_name = "test_data/test_data.df"

attributes = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'length', 'width', 'height']

training_data_extractor = DataExtractor(training_data_path, attributes, training_dict_dump_file_name, training_df_dump_file_name)
training_scenes_list = training_data_extractor.generate_scenes_list()
training_objects_in_scenes = training_data_extractor.get_objects_in_scenes()
training_scenes_df = training_data_extractor.generate_scenes_df()

test_data_extractor = DataExtractor(test_data_path, attributes, test_dict_dump_file_name, test_df_dump_file_name)
test_scenes_list = test_data_extractor.generate_scenes_list()
test_objects_in_scenes = test_data_extractor.get_objects_in_scenes()
test_scenes_df = test_data_extractor.generate_scenes_df()

training_data_dump_folder = "training_data/"

training_single_object_wrapper = SingleObjectWrapper(training_objects_in_scenes, training_df_dump_file_name, training_data_dump_folder)
training_single_object_gmms = training_single_object_wrapper.get_gmm_params()
training_single_object_dfs = training_single_object_wrapper.single_object_dfs
training_single_object_frequencies = training_single_object_wrapper.get_single_object_frequencies(training_scenes_list)

training_object_pair_wrapper = ObjectPairWrapper(training_objects_in_scenes, training_data_dump_folder)
training_object_pair_gmms = training_object_pair_wrapper.get_gmm_params()
training_object_pair_dfs = training_object_pair_wrapper.object_pair_dfs
training_object_pair_frequencies = training_object_pair_wrapper.get_object_pair_frequencies(training_scenes_list)

test_data_dump_folder = "test_data/"

test_single_object_wrapper = SingleObjectWrapper(test_objects_in_scenes, test_df_dump_file_name, test_data_dump_folder)
test_single_object_gmms = test_single_object_wrapper.get_gmm_params()
test_single_object_dfs = test_single_object_wrapper.single_object_dfs
test_single_object_frequencies = test_single_object_wrapper.get_single_object_frequencies(test_scenes_list)

test_object_pair_wrapper = ObjectPairWrapper(test_objects_in_scenes, test_data_dump_folder)
test_object_pair_gmms = test_object_pair_wrapper.get_gmm_params()
test_object_pair_dfs = test_object_pair_wrapper.object_pair_dfs
test_object_pair_frequencies = test_object_pair_wrapper.get_object_pair_frequencies(test_scenes_list)

# training_total_scenes = 10
#
# p_plate = training_single_object_frequencies['plate'] / training_total_scenes
# gmm_plate = training_single_object_gmms['plate']['gmm']

class SimScoreComputer:

    def __init__(self, single_object_frequencies, single_object_gmms, object_pair_frequencies,
                    object_pair_gmms, number_of_training_scenes):

        self.single_object_frequencies = single_object_frequencies
        self.single_object_gmms = single_object_gmms
        self.object_pair_frequencies = object_pair_frequencies
        self.object_pair_gmms = object_pair_gmms

        self.single_object_sim_score = 0
        self.object_pair_sim_score = 0
        self.total_sim_score = 0


    def compute_single_object_sim_score(self, test_single_object_dfs):

        for obj in self.single_object_frequencies:
            object_frequency = self.single_object_frequencies[obj]
            object_gmm = self.single_object_gmms[obj]['gmm']
            object_features_test = test_single_object_dfs[obj]

            self.single_object_sim_score = self.single_object_sim_score + (object_frequency * object_gmm.fit(object_features_test))


number_of_training_scenes = len(training_scenes_list)


sim_score_computer = SimScoreComputer(training_single_object_frequencies, training_single_object_gmms,
                                        training_object_pair_frequencies, training_object_pair_gmms,
                                        number_of_training_scenes)
sim_score_computer.compute_single_object_sim_score(test_single_object_dfs)
