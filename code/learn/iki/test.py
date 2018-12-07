import pickle
import os
import xml.etree.ElementTree as ET
import pandas as pd
from single_object import SingleObjectWrapper
from object_pair import ObjectPairWrapper
from extract_data import DataExtractor
from compute_sim_score import SimScoreComputer

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
training_single_object_dfs = training_single_object_wrapper.get_single_object_dfs()
training_single_object_gmms = training_single_object_wrapper.get_gmm_params()
training_single_object_frequencies = training_single_object_wrapper.get_single_object_frequencies(training_scenes_list)

training_object_pair_wrapper = ObjectPairWrapper(training_objects_in_scenes, training_data_dump_folder)
training_object_pair_dfs = training_object_pair_wrapper.get_object_pair_dfs()
training_object_pair_gmms = training_object_pair_wrapper.get_gmm_params()
training_object_pair_frequencies = training_object_pair_wrapper.get_object_pair_frequencies(training_scenes_list)

test_data_dump_folder = "test_data/"

test_single_object_wrapper = SingleObjectWrapper(test_objects_in_scenes, test_df_dump_file_name, test_data_dump_folder)
test_single_object_dfs = test_single_object_wrapper.get_single_object_dfs()
test_single_object_frequencies = test_single_object_wrapper.get_single_object_frequencies(test_scenes_list)

test_object_pair_wrapper = ObjectPairWrapper(test_objects_in_scenes, test_data_dump_folder)
test_object_pair_dfs = test_object_pair_wrapper.get_object_pair_dfs()
# test_object_pair_frequencies = test_object_pair_wrapper.get_object_pair_frequencies(test_scenes_list)

print("===================================")

number_of_training_scenes = len(training_scenes_list)
sim_score_computer = SimScoreComputer(training_single_object_frequencies, training_single_object_gmms,
                                        training_object_pair_frequencies, training_object_pair_gmms,
                                        number_of_training_scenes)
pre_check = sim_score_computer.check_for_objects(test_single_object_frequencies)
if pre_check == 0:
    single_object_results, object_pair_results, overall_results = sim_score_computer.compute_overall_sim_score(test_single_object_dfs, test_object_pair_dfs)
