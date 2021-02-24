###################################
# RULES BASED ANOMALY DETECTION #
###################################
# This script includes functionality for rules based anomaly detection and preprocessing.
# Functions include range check, persistence check, and linear interpolation for correction.

from PyHydroQC import anomaly_utilities
import pandas as pd
import numpy as np


def range_check(df, maximum, minimum):
    """
    range_check adds a column to data frame with label if data are out of range.
    Input:
    : param df: data frame with a column 'raw' of raw data.
    : param maximum: maximum acceptable value - above this value, data are anomalous
    : param minimum: minimum acceptable value - below this value, data are anomalous
    Output:
    : param df: data frame with an added column 'anomoly' with boolean where 1 = True for anomalous
    """
    # could do some sort of look up table with the values for each sensor
    # could also add seasonal checks
    df = df.eval('anomaly = raw > @maximum or raw < @minimum')
    range_count = sum(df['anomaly'])

    return df, range_count


def persistence(df, length, output_grp=False):
    """
    persistence adds an anomoulous label in the data frame if data repeat for specified length.
    Input:
    : param df: data frame with a column 'raw' of raw data and a boolean column 'anomaly' (typically output of range_check)
    : param length: duration of persistent/repeated values to be flagged
    Output
    : param df: dataframe with column 'anomaly' modified and added column 'persist_grp' that indexes points as part of persistent groups
    : param persist_count: total number of persistent points in the data frame
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
    Input:
    : param df: data frame with column 'anomaly'.
    Output:
    : param size: length of the largest consecutive group of anomalous points.
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
    interpolate performs linear interpolation on points identified as anomolous (typically by rules based approaches).
    Input:
    : param df: data frame with a column 'raw' of raw data and a boolean column 'anomaly'.
    : param limit: maximum length/number of points of acceptable interpolation. If an event is exceeds this length, it will not be interpolated.
    Output:
    : param df: data frame with added column 'observed' for interpolated data.
    """
    df['observed'] = np.where(df['anomaly'], np.nan, df['raw'])
    df['observed'].interpolate(method='linear', inplace=True, limit=limit)

    return df


def add_labels(df, value=-9999):
    """
    add_labels adds an indicator that there is an anomalous value that should have been labeled by the expert but was not. Considers a specified 'no data value' (default is -9999) as well as nan values. Only relevant if comparing to technician/expert labeled data.
    Input:
    : param df: data frame with columns:
        'raw' of raw data
        'cor' of corrected data
        'labeled_anomaly' booleans where True=1 corresponds to anomalies
    : param value: the 'no data value' in the data for which the function checks.
    Output:
    :param df: data frame with column 'labeled_anomaly' modified.
    """
    df['labeled_anomaly'] = np.where((df['raw'] == value) | (df['cor'] == value) | (df['cor'].isnull()), True, df['labeled_anomaly'])

    return df


def calib_detect(df, calib_params):
    """
    calib_detect seeks to find calibration events based on 2 conditions: persistence of a defined length, which often occurs when sensors are out of the water, during certain days of the week (M-F) and times of the day.
    Input:
    : param df: data frame with columns:
        'observed' of observed data
        'anomaly' booleans where True=1 corresponds to anomalies
        'persist_grp' with indices of peristent groups (output of the persist function)
    : param calib_params: parameters defined in the parameters file
        persist_high: longest length of the persistent group
        perist_low: shortest length of the persistent group
        hour_low: earliest hour for calibrations to have occurred
        hour_high: latest hour for calibrations to have occurred
    Output:
    : param calib: data frame of booleans indicating whether the conditions were met for a possible calibration event.
    : param calib_dates: datetimes for which the conditions were met for a possible calibration event.
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
    calib_overlap seeks to identify calibration events by identifying where overlaps occur between multiple sensors. Calls the calib_detect function to identify events with a defined persistence length during certain days of the week (M-F) and hours of the day.
    Input:
    : param sensor_names: list of sensors to be considered for overlap.
    : param input_array: array of data frames each with columns:
        'observed' of observed data
        'anomaly' booleans where True=1 corresponds to anomalies
        'persist_grp' with indices of peristent groups (output of the persist function)
    : param calib_params: parameters defined in the parameters file
        persist_high: longest length of the persistent group
        perist_low: shortest length of the persistent group
        hour_low: earliest hour for calibrations to have occurred
        hour_high: latest hour for calibrations to have occurred
    Output:
    : param all_calib: array of data frames (one for each sensor) of booleans indicating whether the conditions were met for a possible calibration event.
    : param all_calib_dates: array of datetimes (one for each sensor) for which the conditions were met for a possible calibration event.
    : param df_all_calib: data frame with columns for each sensor observations, columns of booleans for each sensor indicating whether a calibration event may have occurred, and a column 'all_calib' that indicates if the conditions were met for all sensors.
    : param calib_dates_overlap: datetimes for which the conditions were met for a possible calibration event for all sensors.
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
   lin_drift_cor performs linear drift correction on data. Typical correction for calibration events. this function operates on the basis of a single event
   Input:
   : param observed: time series of observations.
   : param start: datetime for the beginning of the correction
   : param end: datetime for the end of the correction
   : param gap: gap value that determines the degree of the shift, which occurs at the end date.
   : param replce: indicates whether the values of the correction should replace the associated values in the data frame.
   Output:
   : param result: data frame of corrected values
   : param observed: time series of observations with corrected values if replace was selected.
    """
    # todo: ignore - 9999 values

    result = pd.DataFrame(index=observed.loc[start:end].index)
    result['ldc'] = pd.Series(dtype=float)
    for i in range(len(observed.loc[start:end])):
        #y = original_value[i] - i * gap / total_num_points
        y = observed.iloc[observed.index.get_loc(start) + i] + gap/(len(observed.loc[start:end])-1) * i
        result['ldc'].iloc[i] = y
    if replace:
        observed.loc[start:end] = result['ldc']

    return result, observed

