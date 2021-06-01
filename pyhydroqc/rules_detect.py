###################################
# RULES BASED ANOMALY DETECTION #
###################################
# This file includes functionality for rules based anomaly detection and preprocessing.
# Functions include range check, persistence check, and linear interpolation for correction.

from pyhydroqc import anomaly_utilities
import numpy as np


def range_check(df, maximum, minimum):
    """
    range_check adds a column to data frame with label if data are out of range.
    Arguments:
        df: data frame with a column 'raw' of raw data.
        maximum: maximum acceptable value - above this value, data are anomalous
        minimum: minimum acceptable value - below this value, data are anomalous
    Returns:
        df: data frame with an added column 'anomaly' with boolean where 1 = True for anomalous
        range_count: total number of anomalies from this check
    """
    # could do some sort of look up table with the values for each sensor
    # could also add seasonal checks
    df = df.eval('anomaly = raw > @maximum or raw < @minimum')
    range_count = sum(df['anomaly'])

    return df, range_count


def persistence(df, length, output_grp=False):
    """
    persistence adds an anomalous label in the data frame if data repeat for specified length.
    Arguments:
        df: data frame with a column 'raw' of raw data and a boolean column 'anomaly' (typically output of range_check)
        length: duration of persistent/repeated values to be flagged
        output_grp: boolean to indicate whether the length of persistence should be output as a column in the original dataframe.
    Returns:
        df: dataframe with column 'anomaly' modified and added column 'persist_grp' that indexes points as part of persistent groups
        persist_count: total number of persistent points in the data frame
    """
    temp = df[['raw', 'anomaly']].copy(deep=True)
    temp['persist_grp'] = (temp.raw.diff(1) == 0)
    temp['persist_grp'] = anomaly_utilities.anomaly_events(temp['persist_grp'], 0, 1)
    for i in range(1, max(temp['persist_grp']) + 1):
        if(len(temp['persist_grp'][temp['persist_grp'] == i]) >= length):
            temp['anomaly'][temp['persist_grp'] == i] = True
    persist_count = sum(temp['persist_grp'] != 0)
    df['anomaly'] = temp['anomaly']
    if output_grp:
        df['persist_grp'] = temp['persist_grp']

    return df, persist_count


def group_size(df):
    """
    group_size determines the size of the largest consecutive group of anomalous points.
    Arguments:
        df: data frame with column 'anomaly'.
    Returns:
        size: length of the largest consecutive group of anomalous points.
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
    interpolate performs linear interpolation on points identified as anomalous (typically by rules based approaches).
    Arguments:
        df: data frame with a column 'raw' of raw data and a boolean column 'anomaly'.
        limit: maximum length/number of points of acceptable interpolation. If an event is exceeds this length, it will not be interpolated.
    Returns:
        df: data frame with added column 'observed' for interpolated data.
    """
    df['observed'] = np.where(df['anomaly'], np.nan, df['raw'])
    df['observed'].interpolate(method='linear', inplace=True, limit=limit, limit_direction='both')

    return df


def add_labels(df, value=-9999):
    """
    add_labels adds an indicator that there is an anomalous value that should have been labeled by the expert but was not. Considers a specified 'no data value' (default is -9999) as well as null values. Only relevant if comparing to technician/expert labeled data.
    Arguments:
        df: data frame with columns:
            'raw' of raw data
            'cor' of corrected data
            'labeled_anomaly' booleans where True=1 corresponds to anomalies
        value: the 'no data value' in the data for which the function checks.
    Returns:
        df: data frame with column 'labeled_anomaly' modified.
    """
    df['labeled_anomaly'] = np.where((df['raw'] == value) | (df['cor'] == value) | (df['cor'].isnull()), True, df['labeled_anomaly'])

    return df
