###################################
# RULES BASED ANOMALY DETECTION #
###################################
# This script includes functionality for rules based anomaly detection and preprocessing.
# Functions include range check, persistence check, and linear interpolation for correction.

import numpy as np
import anomaly_utilities
import pandas as pd


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
    range_count = sum(df['anomaly'])

    return df, range_count


def persistence(df, length, output_grp=False):
    """
    persistence adds flag in data frame if data repeat for specified length.
    df is a data frame with a column 'raw' of raw data and a boolean column 'anomaly'.
    length is the duration of persistent/repeated values to be flagged.
    Output is the dataframe with column 'anomaly' modified.
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
    df['observed'] = np.where(df['anomaly'], np.nan, df['raw'])
    df['observed'].interpolate(method='linear', inplace=True, limit=limit)

    return df

def add_labels(df, value = -9999):
    """
    add_labels adds an indicator that there is an anomalous value that should have been labeled by the expert but was not.
    df is a data frame with columns 'raw' of raw data, 'cor' of corrected data, and a boolean column 'labeled_anomaly'.
    Considers a specified 'no data value' (default is -9999) as well as nan values.
    """
    df['labeled_anomaly'] = np.where((df['raw'] == value) | (df['cor'] == value) | (df['cor'].isnull()), True, df['labeled_anomaly'])

    return df


def calib_detect(df, calib_params):
    """
    """
    if 'persist_grp' in df:
        temp = df[['observed', 'anomaly', 'persist_grp']].copy(deep=True)
        for i in range(1, max(temp['persist_grp']) + 1):
            temp['persist_grp'][temp.loc[temp['persist_grp'].shift(-1) == i].index[0]] = i
            if ((len(temp['persist_grp'][temp['persist_grp'] == i]) >= calib_params['persist_low']) and
                    (len(temp['persist_grp'][temp['persist_grp'] == i]) <= calib_params['persist_high'])):
                temp['anomaly'][temp['persist_grp'] == i] = True
    else:
        temp = df[['observed', 'anomaly']].copy(deep=True)
        temp['persist_grp'] = (temp.observed.diff(1) == 0)
        temp['persist_grp'] = anomaly_utilities.anomaly_events(temp['persist_grp'], 0, 1)
        for i in range(1, max(temp['persist_grp']) + 1):
            temp['persist_grp'][temp.loc[temp['persist_grp'].shift(-1) == i].index[0]] = i
            if ((len(temp['persist_grp'][temp['persist_grp'] == i]) >= calib_params['persist_low']) and
                    (len(temp['persist_grp'][temp['persist_grp'] == i]) <= calib_params['persist_high'])):
                temp['anomaly'][temp['persist_grp'] == i] = True

    dayofweek = temp.index.dayofweek
    hour = temp.index.hour
    business = temp.iloc[((dayofweek == 0) | (dayofweek == 1) | (dayofweek ==2) | (dayofweek == 3) | (dayofweek == 4))
                         & (hour >= calib_params['hour_low']) & (hour <= calib_params['hour_high'])]
    calib = pd.DataFrame(index=temp.index)
    calib['anomaly'] = False
    calib['anomaly'].loc[business[business['anomaly']].index] = True
    calib_dates = calib[calib['anomaly']].index

    return calib, calib_dates


def calib_overlap(sensor_names, input_array, calib_params):
    """
    """
    all_calib = dict()
    all_calib_dates = dict()
    df_all_calib = pd.DataFrame(index = input_array[sensor_names[0]].index)
    df_all_calib['all_calib'] = True
    for snsr in sensor_names:
        calib, calib_dates = calib_detect(input_array[snsr], calib_params)
        all_calib[snsr] = calib
        all_calib_dates[snsr] = calib_dates
        df_all_calib[snsr] = input_array[snsr]['observed']
        df_all_calib[snsr + '_calib'] = all_calib[snsr]['anomaly']
        df_all_calib[snsr + '_event'] = anomaly_utilities.anomaly_events(df_all_calib[snsr + '_calib'], wf=1, sf=1)
        df_all_calib['all_calib'] = np.where(df_all_calib['all_calib'] & (df_all_calib[snsr + '_event'] != 0), True, False)
    calib_dates_overlap = df_all_calib[df_all_calib['all_calib']].index

    return all_calib, all_calib_dates, df_all_calib, calib_dates_overlap


def lin_drift_cor(observed, start, end, gap, replace=True):
    """
    todo: ignore -9999 values
    """
    result = pd.DataFrame(index=observed.loc[start:end].index)
    result['ldc'] = pd.Series(dtype=float)
    for i in range(len(observed.loc[start:end])):
        #y = original_value[i] - i * gap / total_num_points
        y = observed.iloc[observed.index.get_loc(start) + i] + gap/(len(observed.loc[start:end])-1) * i
        result['ldc'].iloc[i] = y
    if replace:
        observed.loc[start:end] = result['ldc']

    return result, observed

