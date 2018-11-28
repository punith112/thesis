import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

main_df = pd.read_csv('database', sep = '\t', index_col=0)

def extract_single_obj_features(df, obj, features):

    feature_set = []
    columns = []

    for i in range(len(features)):
        columns.append(obj + '_' + features[i])

    df = df[columns]

    return df


features = ['x', 'y', 'z', 'length', 'width', 'height']
plate_feature_set = extract_single_obj_features(main_df, 'plate', features)
