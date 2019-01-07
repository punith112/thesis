import pickle
import os
import xml.etree.ElementTree as ET
import pandas as pd
from single_object import SingleObjectWrapper
from object_pair import ObjectPairWrapper
from extract_data import DataExtractor
from compute_sim_score import SimScoreComputer
import math
import matplotlib.pyplot as plt

# Attributes to be extracted from the training scenes
attributes = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'length', 'width', 'height']

# Path to the training scenes
training_data_path = "/home/iki/catkin_ws/src/thesis/iki_dataset/training_data/"

# Path to where the extracted data from the scenes is to be dumped
training_dict_dump_file_name = "training_data/training_data.dict"
training_df_dump_file_name = "training_data/training_data.df"

# Extracting the attributes data from the training scenes
training_data_extractor = DataExtractor(training_data_path, attributes, training_dict_dump_file_name,
                                        training_df_dump_file_name)
training_scenes_list = training_data_extractor.generate_scenes_list()
training_objects_in_scenes = training_data_extractor.get_objects_in_scenes()
training_scenes_df = training_data_extractor.generate_scenes_df()

# Path to where the custom features extracted from the scenes are to be dumped
training_data_dump_folder = "training_data/"

# Extracting single object and object pair features
training_single_object_wrapper = SingleObjectWrapper(training_objects_in_scenes, training_df_dump_file_name, training_data_dump_folder)
training_single_object_dfs = training_single_object_wrapper.get_single_object_dfs()
training_single_object_gmms = training_single_object_wrapper.get_gmm_params()
training_single_object_frequencies = training_single_object_wrapper.get_single_object_frequencies(training_scenes_list)

training_object_pair_wrapper = ObjectPairWrapper(training_objects_in_scenes, training_data_dump_folder)
training_object_pair_dfs = training_object_pair_wrapper.get_object_pair_dfs()
training_object_pair_gmms = training_object_pair_wrapper.get_gmm_params()
training_object_pair_frequencies = training_object_pair_wrapper.get_object_pair_frequencies(training_scenes_list)

number_of_training_scenes = len(training_scenes_list)

training_sim_score_computer = SimScoreComputer(training_single_object_frequencies, training_single_object_gmms,
                                        training_object_pair_frequencies, training_object_pair_gmms,
                                        number_of_training_scenes)

training_single_object_thresholds, training_object_pair_thresholds = \
training_sim_score_computer.compute_training_thresholds(training_single_object_dfs, training_object_pair_dfs)



# Path to the test scenes
test_data_path = "/home/iki/catkin_ws/src/thesis/iki_dataset/test_data/"

# Path to where the extracted data from the scenes is to be dumped
test_dict_dump_file_name = "test_data/test_data.dict"
test_df_dump_file_name = "test_data/test_data.df"

# Extracting the attributes data from the test scenes
test_data_extractor = DataExtractor(test_data_path, attributes, test_dict_dump_file_name, test_df_dump_file_name)
test_scenes_list = test_data_extractor.generate_scenes_list()
test_objects_in_scenes = test_data_extractor.get_objects_in_scenes()
test_scenes_df = test_data_extractor.generate_scenes_df()

# Path to where the custom features extracted from the scenes are to be dumped
test_data_dump_folder = "test_data/"

# Extracting single object and object pair features
test_single_object_wrapper = SingleObjectWrapper(test_objects_in_scenes, test_df_dump_file_name, test_data_dump_folder)
test_single_object_dfs = test_single_object_wrapper.get_single_object_dfs()
test_single_object_frequencies = test_single_object_wrapper.get_single_object_frequencies(test_scenes_list)

test_object_pair_wrapper = ObjectPairWrapper(test_objects_in_scenes, test_data_dump_folder)
test_object_pair_dfs = test_object_pair_wrapper.get_object_pair_dfs()
test_object_pair_frequencies = test_object_pair_wrapper.get_object_pair_frequencies(test_scenes_list)
#
# print("===================================")
#
number_of_training_scenes = len(training_scenes_list)
#
test_sim_score_computer = SimScoreComputer(training_single_object_frequencies, training_single_object_gmms,
                                        training_object_pair_frequencies, training_object_pair_gmms,
                                        number_of_training_scenes)

# # pre_check = sim_score_computer.check_for_objects(test_single_object_frequencies)
#
#
#
test_single_object_results, test_object_pair_results, test_overall_results = \
test_sim_score_computer.compute_overall_sim_score(test_single_object_dfs, test_object_pair_dfs,
                                                  training_single_object_thresholds, training_object_pair_thresholds)


# -------- Line Plots --------

plate = test_single_object_results['plate'].iloc[:, 6]
cup = test_single_object_results['cup'].iloc[:, 6]
spoon = test_single_object_results['spoon'].iloc[:, 6]

cup_plate = test_object_pair_results['cup_plate'].iloc[:, 6]
cup_spoon = test_object_pair_results['cup_spoon'].iloc[:, 6]
plate_spoon = test_object_pair_results['plate_spoon'].iloc[:, 6]

# print(plate)
# print(cup)
# print(spoon)
#
# print(cup_plate)
# print(cup_spoon)
# print(plate_spoon)

plate_threshold = pd.Series(training_single_object_thresholds['plate'], index=plate.index)
cup_threshold = pd.Series(training_single_object_thresholds['cup'], index=plate.index)
spoon_threshold = pd.Series(training_single_object_thresholds['spoon'], index=plate.index)

cup_plate_threshold = pd.Series(training_object_pair_thresholds['cup_plate'], index=plate.index)
cup_spoon_threshold = pd.Series(training_object_pair_thresholds['cup_spoon'], index=plate.index)
plate_spoon_threshold = pd.Series(training_object_pair_thresholds['plate_spoon'], index=plate.index)

# fig, ax = plt.subplots()
# ax.plot(s.index, s)

single_threshold1, = plt.plot(plate_threshold.index, plate_threshold, '--r', label='plate_th')
single_threshold2, = plt.plot(cup_threshold.index, cup_threshold, '--g', label='cup_th')
single_threshold3, = plt.plot(spoon_threshold.index, spoon_threshold, '--b', label='spoon_th')
# first_legend = plt.legend(handles=[single_threshold1, single_threshold2, single_threshold3], loc=4)

pair_threshold1, = plt.plot(cup_plate_threshold.index, cup_plate_threshold, ':r', label='cup_plate_th')
pair_threshold2, = plt.plot(cup_spoon_threshold.index, cup_spoon_threshold, ':g', label='cup_spoon_th')
pair_threshold3, = plt.plot(plate_spoon_threshold.index, plate_spoon_threshold, ':b', label='plate_spoon_th')
# plt.gca().add_artist(first_legend)
first_legend = plt.legend(handles=[single_threshold1, single_threshold2, single_threshold3, pair_threshold1, pair_threshold2, pair_threshold3], loc=3)

single_sim_score1, = plt.plot(plate.index, plate, '-r', label='plate_score')
single_sim_score2, = plt.plot(cup.index, cup, '-g', label='cup_score')
single_sim_score3, = plt.plot(spoon.index, spoon, '-b', label='spoon_score')
# plt.gca().add_artist(first_legend)
# third_legend = plt.legend(handles=[single_sim_score1, single_sim_score2, single_sim_score3], loc=1)

pair_sim_score1, = plt.plot(cup_plate.index, cup_plate, '-.r', label='cup_plate_score')
pair_sim_score2, = plt.plot(cup_spoon.index, cup_spoon, '-.g', label='cup_spoon_score')
pair_sim_score3, = plt.plot(plate_spoon.index, plate_spoon, '-.b', label='plate_spoon_score')
plt.gca().add_artist(first_legend)
third_legend = plt.legend(handles=[single_sim_score1, single_sim_score2, single_sim_score3, pair_sim_score1, pair_sim_score2, pair_sim_score3], loc=2)

# plt.ylim(-550, 35)
plt.ylim(0, 35)
plt.ylabel('similarity scores')
plt.xticks(rotation=60)
plt.savefig('test.png')
plt.show(block=False)



# --------- Scatter Plots -------

# plate = test_single_object_results['plate'].iloc[:, 6]
# cup = test_single_object_results['cup'].iloc[:, 6]
# spoon = test_single_object_results['spoon'].iloc[:, 6]
#
# plate_threshold = pd.Series(training_single_object_thresholds['plate'], index=plate.index)
# cup_threshold = pd.Series(training_single_object_thresholds['cup'], index=cup.index)
# spoon_threshold = pd.Series(training_single_object_thresholds['spoon'], index=plate.index)
#
# plt.figure(1)
#
# single_threshold1, = plt.plot(plate_threshold.index, plate_threshold, '--r', label='plate_th')
# single_threshold2, = plt.plot(cup_threshold.index, cup_threshold, '--g', label='cup_th')
# single_threshold3, = plt.plot(spoon_threshold.index, spoon_threshold, '--b', label='spoon_th')
# first_legend = plt.legend(handles=[single_threshold1, single_threshold2, single_threshold3], loc=4)
#
# single_sim_score1, = plt.plot(plate.index, plate, 'o', c='red', alpha=0.5, markeredgecolor='none', label='plate_score')
# single_sim_score2, = plt.plot(cup.index, cup, 'o', c='green', alpha=0.5, markeredgecolor='none', label='cup_score')
# single_sim_score3, = plt.plot(spoon.index, spoon, 'o', c='blue', alpha=0.5, markeredgecolor='none', label='spoon_score')
# plt.gca().add_artist(first_legend)
# second_legend = plt.legend(handles=[single_sim_score1, single_sim_score2, single_sim_score3], loc=1)
#
#
# # plt.ylim(0, 35)
# # plt.yscale('log')
# plt.axis('tight')
# plt.ylabel('similarity scores')
# plt.xticks(rotation=60)
# plt.title('Single Object Similarity Scores')
#
#
# cup_plate = test_object_pair_results['cup_plate'].iloc[:, 6]
# cup_spoon = test_object_pair_results['cup_spoon'].iloc[:, 6]
# plate_spoon = test_object_pair_results['plate_spoon'].iloc[:, 6]
#
# cup_plate_threshold = pd.Series(training_object_pair_thresholds['cup_plate'], index=plate.index)
# cup_spoon_threshold = pd.Series(training_object_pair_thresholds['cup_spoon'], index=plate.index)
# plate_spoon_threshold = pd.Series(training_object_pair_thresholds['plate_spoon'], index=plate.index)
#
# plt.figure(2)
#
# pair_threshold1, = plt.plot(cup_plate_threshold.index, cup_plate_threshold, ':r', label='cup_plate_th')
# pair_threshold2, = plt.plot(cup_spoon_threshold.index, cup_spoon_threshold, ':g', label='cup_spoon_th')
# pair_threshold3, = plt.plot(plate_spoon_threshold.index, plate_spoon_threshold, ':b', label='plate_spoon_th')
# # plt.gca().add_artist(first_legend)
# third_legend = plt.legend(handles=[pair_threshold1, pair_threshold2, pair_threshold3], loc=4)
#
# pair_sim_score1, = plt.plot(cup_plate.index, cup_plate, 'o', c='red', alpha=0.5, markeredgecolor='none', label='cup_plate_score')
# pair_sim_score2, = plt.plot(cup_spoon.index, cup_spoon, 'o', c='green', alpha=0.5, markeredgecolor='none', label='cup_spoon_score')
# pair_sim_score3, = plt.plot(plate_spoon.index, plate_spoon, 'o', c='blue', alpha=0.5, markeredgecolor='none', label='plate_spoon_score')
# plt.gca().add_artist(third_legend)
# fourth_legend = plt.legend(handles=[single_sim_score1, single_sim_score2, single_sim_score3, pair_sim_score1, pair_sim_score2, pair_sim_score3], loc=1)
#
# # plt.ylim(0, 35)
# # plt.yscale('log')
# plt.axis('tight')
# plt.ylabel('similarity scores')
# plt.xticks(rotation=60)
# plt.title('Object Pair Similarity Scores')
#
# plt.show(block=False)



# --------- Line plots, but separate ones -------
