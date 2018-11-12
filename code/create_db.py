import xml.etree.ElementTree as ET
import os
import csv
import pickle
import pandas as pd

def gen_pd_series(obj_name, feature, main_list):

    temp_dict = {}

    for index in range(len(main_list)):
        try:
            temp_dict[main_list[index]['file']] = main_list[index][obj_name][feature]
        except KeyError:
            temp_dict[main_list[index]['file']] = float('NaN')

    pd_series = pd.Series(temp_dict)

    return pd_series

with open("extracted_data.txt", "rb") as myFile:
    main_list = pickle.load(myFile)

# objects_in_scene = []
#
# for i in range(len(main_list)):
#     objects_in_scene = list(set().union(objects_in_scene, main_list[i].keys()))
#
# objects_in_scene.remove('file')

objects_in_scene = ['Mouse', 'Monitor']

features = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'length', 'width', 'height']

main_df = pd.DataFrame()

for obj in objects_in_scene:
    for feature in features:
        temp_pd_series = gen_pd_series(obj, feature, main_list)
        main_df[obj + '_' + feature] = temp_pd_series
        # print("")
        # print("======")
        # print(obj, feature)
        # print(temp_pd_series)

# mouse_x_series = gen_pd_series('Mouse', 'x', main_list)
# monitor_y_series = gen_pd_series('Monitor', 'y', main_list)

main_df.to_csv('database', sep = '\t')
