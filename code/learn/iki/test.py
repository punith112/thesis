from collections import OrderedDict


def compute_x_diff(obj1, obj2, obj1_df, obj2_df):

    x_diff_series = pd.Series()
    x_diff_series = obj1_df[obj1 + '_x'] - obj2_df[obj2 + '_x']

    return x_diff_series

def compute_y_diff(obj1, obj2, obj1_df, obj2_df):

    y_diff_series = pd.Series()
    y_diff_series = obj1_df[obj1 + '_y'] - obj2_df[obj2 + '_y']

    return y_diff_series

def compute_z_diff(obj1, obj2, obj1_df, obj2_df):

    z_diff_series = pd.Series()
    z_diff_series = obj1_df[obj1 + '_z'] - obj2_df[obj2 + '_z']

    return z_diff_series

def compute_length_ratio(obj1, obj2, obj1_df, obj2_df):

    length_ratio_series = pd.Series()
    length_ratio_series = obj1_df[obj1 + '_length'] / obj2_df[obj2 + '_length']

    return length_ratio_series

def compute_width_ratio(obj1, obj2, obj1_df, obj2_df):

    width_ratio_series = pd.Series()
    width_ratio_series = obj1_df[obj1 + '_width'] / obj2_df[obj2 + '_width']

    return width_ratio_series

def compute_height_ratio(obj1, obj2, obj1_df, obj2_df):

    height_ratio_series = pd.Series()
    height_ratio_series = obj1_df[obj1 + '_height'] / obj2_df[obj2 + '_height']

    return height_ratio_series

features = OrderedDict()
features['x_diff'] = compute_x_diff
features['y_diff'] = compute_y_diff
features['z_diff'] = compute_z_diff
features['length_ratio'] = compute_length_ratio
features['width_ratio'] = compute_width_ratio
features['height_ratio'] = compute_height_ratio

for key, value in features.items():
    print(key, value)
