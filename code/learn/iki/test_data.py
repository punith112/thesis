import pandas as pd
from collections import OrderedDict
from single_object import SingleObjectWrapper
from object_pair import ObjectPairWrapper

# indices = ['table_1', 'table_2', 'table_3']
#
# monitor_x = pd.Series([1, 2, 3], index=indices)
# monitor_y = pd.Series([2, 4, 6], index=indices)
# monitor_z = pd.Series([3, 6, 9], index=indices)
# monitor_length = pd.Series([10, 20, 30], index=indices)
# monitor_width = pd.Series([10, 20, 30], index=indices)
# monitor_height = pd.Series([10.0, 20.0, 30.0], index=indices)
#
# monitor_df = pd.DataFrame({'monitor_x': monitor_x, 'monitor_y': monitor_y, 'monitor_z': monitor_z,
#                             'monitor_length': monitor_length, 'monitor_width': monitor_width, 'monitor_height': monitor_height})
#
# keyboard_x = pd.Series([1, 2, 3], index=indices)
# keyboard_y = pd.Series([-2.0, -4.0, -6.0], index=indices)
# keyboard_z = pd.Series([-3, -6, -9], index=indices)
# keyboard_length = pd.Series([10, 20, 30], index=indices)
# keyboard_width = pd.Series([0, 0, 0], index=indices)
# keyboard_height = pd.Series([10.1, 20.2, 30.3], index=indices)
#
# keyboard_df = pd.DataFrame({'keyboard_x': keyboard_x, 'keyboard_y': keyboard_y, 'keyboard_z': keyboard_z,
#                             'keyboard_length': keyboard_length, 'keyboard_width': keyboard_width, 'keyboard_height': keyboard_height})
#
# object_attributes_df = pd.concat([monitor_df, keyboard_df], axis=1)
# object_attributes_df.to_csv('test_database', sep = '\t')

objects_in_scene = ['monitor', 'keyboard']

single_object_wrapper = SingleObjectWrapper(objects_in_scene, "data/test_database")
single_object_gmms = single_object_wrapper.get_gmm_params()

object_pair_wrapper = ObjectPairWrapper(objects_in_scene)
object_pair_gmms = object_pair_wrapper.get_gmm_params()
