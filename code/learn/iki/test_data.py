import pandas as pd
from collections import OrderedDict
from obj_pair_features import compute_x_diff, compute_y_diff, compute_z_diff
from obj_pair_features import compute_length_ratio, compute_width_ratio, compute_height_ratio
from obj_pair_features import extract_obj_pair_features

indices = ['table_1', 'table_2', 'table_3']

monitor_x = pd.Series([1, 2, 3], index=indices)
monitor_y = pd.Series([2, 4, 6], index=indices)
monitor_z = pd.Series([3, 6, 9], index=indices)
monitor_length = pd.Series([10, 20, 30], index=indices)
monitor_width = pd.Series([10, 20, 30], index=indices)
monitor_height = pd.Series([10.0, 20.0, 30.0], index=indices)

monitor_df = pd.DataFrame({'monitor_x': monitor_x, 'monitor_y': monitor_y, 'monitor_z': monitor_z,
                            'monitor_length': monitor_length, 'monitor_width': monitor_width, 'monitor_height': monitor_height})

keyboard_x = pd.Series([1, 2, 3], index=indices)
keyboard_y = pd.Series([-2.0, -4.0, -6.0], index=indices)
keyboard_z = pd.Series([-3, -6, -9], index=indices)
keyboard_length = pd.Series([10, 20, 30], index=indices)
keyboard_width = pd.Series([0, 0, 0], index=indices)
keyboard_height = pd.Series([10.1, 20.2, 30.3], index=indices)

keyboard_df = pd.DataFrame({'keyboard_x': keyboard_x, 'keyboard_y': keyboard_y, 'keyboard_z': keyboard_z,
                            'keyboard_length': keyboard_length, 'keyboard_width': keyboard_width, 'keyboard_height': keyboard_height})

# x_diff_series = compute_x_diff('keyboard', 'monitor', keyboard_df, monitor_df)
# y_diff_series = compute_y_diff('keyboard', 'monitor', keyboard_df, monitor_df)
# z_diff_series = compute_z_diff('keyboard', 'monitor', keyboard_df, monitor_df)


features = OrderedDict()

features['x_diff'] = compute_x_diff
features['y_diff'] = compute_y_diff
features['z_diff'] = compute_z_diff
features['length_ratio'] = compute_length_ratio
features['width_ratio'] = compute_width_ratio
features['height_ratio'] = compute_height_ratio

keyboard_monitor_features = extract_obj_pair_features('keyboard', 'monitor', keyboard_df, monitor_df, features)
