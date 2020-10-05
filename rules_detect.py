###################################
# RULES BASED ANOMALY DETECTION #
###################################
# This script includes functionality for rules based anomaly detection and preprocessing.
# Functions include range check, persistence check, and linear interpolation for correction.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anomaly_utilities


def range_check(df, maximum, minimum):
    """
    range_check adds column to data frame with label if data are out of range.
    df is a data frame with a column 'raw' of raw data.
    maximum and minimum define the range outside of which a value is anomalous.
    Output is the dataframe with an added column 'anomoly' with boolean where 1 = True for anomalous.
    """
    # could do some sort of look up table with the values for each sensor
    # could also add seasonal checks
    df = df.eval('anomaly = raw > @maximum or raw < @minimum')

    return df


def persistence(df, length):
    """
    persistence adds flag in data frame if data repeat for specified length.
    df is a data frame with a column 'raw' of raw data and a boolean column 'anomaly'.
    length is the duration of persistent/repeated values to be flagged.
    Output is the dataframe with column 'anomaly' modified.
    """
    # temp = df.copy(deep=True)
    temp = df[['raw', 'anomaly']].copy(deep=True)
    temp['value_grp'] = (temp.raw.diff(1) == 0)
    temp['value_grp'] = anomaly_utilities.anomaly_events(temp['value_grp'], 0, 1)
    for i in range(1, max(temp['value_grp']) + 1):
        if(len(temp['value_grp'][temp['value_grp'] == i]) >= length):
            temp['anomaly'][temp['value_grp'] == i] = True
    df['anomaly'] = temp['anomaly']

    return df


def group_size(df):
    """
    group_size determines the size of the largest consecutive group of anomalous points.
    df is a data frame with column 'anomaly'.
    Output: size is the length of the largest consecutive group of anomalous points.
    """
    temp = df[['anomaly']].copy(deep=True)
    temp['value_grp'] = anomaly_utilities.anomaly_events(temp['anomaly'], 0, 1)
    size = 0
    if max(temp['value_grp']) > 0:
        size = len(temp['value_grp'][temp['value_grp'] == 1])
        for i in range(2, max(temp['value_grp']) + 1):
            if(len(temp['value_grp'][temp['value_grp'] == i]) > size):
                size = len(temp['value_grp'][temp['value_grp'] == i])

    return size


def interpolate(df, limit=10000):
    """
    interpolate performs linear interpolation on points identified as anomolous, typically by rules based approaches.
    df is a data frame with a column 'raw' of raw data and a boolean column 'anomaly'.
    limit is the maximum length/number of points of acceptable interpolation.
    Output is the dataframe with column 'observed' for determined corrections.
    """
    df['observed'] = (np.where(df['anomaly'], np.nan, df['raw']))
    df['observed'].interpolate(method='linear', inplace=True, limit=limit)

    return df
