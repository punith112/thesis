import pickle
import os
import xml.etree.ElementTree as ET
import pandas as pd
from single_object import SingleObjectWrapper
from object_pair import ObjectPairWrapper
from extract_data import DataExtractor

training_data_path = "/home/iki/catkin_ws/src/thesis/iki_dataset/training_data/"
training_dict_dump_file_name = "data/training_data.dict"
training_df_dump_file_name = "data/training_data.df"

test_data_path = "/home/iki/catkin_ws/src/thesis/iki_dataset/test_data/"
test_dict_dump_file_name = "data/test_data.dict"
test_df_dump_file_name = "data/test_data.df"

attributes = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'length', 'width', 'height']

training_data_extractor = DataExtractor(training_data_path, attributes, training_dict_dump_file_name, training_df_dump_file_name)
training_scenes_list = training_data_extractor.generate_scenes_list()
training_objects_in_scenes = training_data_extractor.get_objects_in_scenes()
training_scenes_df = training_data_extractor.generate_scenes_df()

test_data_extractor = DataExtractor(test_data_path, attributes, test_dict_dump_file_name, test_df_dump_file_name)
test_scenes_list = test_data_extractor.generate_scenes_list()
test_objects_in_scenes = test_data_extractor.get_objects_in_scenes()
test_scenes_df = test_data_extractor.generate_scenes_df()


single_object_wrapper = SingleObjectWrapper(training_objects_in_scenes, training_df_dump_file_name)
single_object_gmms = single_object_wrapper.get_gmm_params()
single_object_frequencies = single_object_wrapper.get_single_object_frequencies(training_scenes_list)

object_pair_wrapper = ObjectPairWrapper(training_objects_in_scenes)
object_pair_gmms = object_pair_wrapper.get_gmm_params()
object_pair_frequencies = object_pair_wrapper.get_object_pair_frequencies(training_scenes_list)
