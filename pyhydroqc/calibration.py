###################################
# CALIBRATION DETECTION AND CORRECTION #
###################################
# This file includes functionality for identification and correction of calibration events.
# Functions include detection based on edges or persistence restricted by day of week and hour of day, identification
# of gap values as input to correction, and linear drift correction.

from pyhydroqc import anomaly_utilities
import pandas as pd
import numpy as np


def calib_edge_detect(observed, width, calib_params, threshold=float("nan"), num_events=1, alpha=float("nan")):
    """
   calib_edge_detect seeks to find likely calibration event candidates by using edge filtering
   Arguments:
       observed: time series of observations
       width: the width of the edge detection filter
       calib_params: parameters defined in the parameters file
            hour_low: earliest hour for calibrations to have occurred
            hour_high: latest hour for calibrations to have occurred
       threshold: used for determining candidates from edge filter results
       num_events: the number of calibration event candidates to return
       alpha: used for determining a threshold from the data
   Returns:
       candidates: datetimes of the most likely calibration event candidates
       edge_diff: differences indicating degree of edges
    """
    # TODO: add functionality for num_events and alpha

    candidates = []
    edge_diff = pd.DataFrame(index=observed.index)  # diff['val'] is the filter output
    edge_diff['val'] = 0
    for i in range(width, len(observed) - width):  # loop over every possible difference calculation
        # implement the edge detection filter - difference of the sums of before and after data
        edge_diff.iloc[i] = (sum(observed[i - width:i]) - sum(observed[i:i + width])) / width

    if not np.isnan(threshold):  # if the function is being called with a threshold

        # iterate over each day, this assumes that a sensor will not be calibrated twice in one day
        for idx, day in edge_diff.groupby(edge_diff.index.date):
            if max(abs((day['val']))) > threshold:  # if any value is above the threshold in that day
                candidates.append(pd.to_datetime(day.idxmax()['val']))  # add it to the list of calibration candidates

    # specify that calibrations would only occur on work days and between specified hours of the day
    candidates = np.array(candidates)
    candidates = candidates[(pd.to_datetime(candidates).dayofweek <= 4) &
                            (pd.to_datetime(candidates).hour >= calib_params.hour_low) &
                            (pd.to_datetime(candidates).hour <= calib_params.hour_high)]

    return candidates, edge_diff


def calib_persist_detect(df, calib_params):
    """
    calib_detect seeks to find calibration events based on 2 conditions: persistence of a defined length, which often occurs when sensors are out of the water, during certain days of the week (M-F) and times of the day.
    Arguments:
        df: data frame with columns:
            'observed' of observed data
            'anomaly' booleans where True=1 corresponds to anomalies
            'persist_grp' with indices of peristent groups (output of the persist function)
        calib_params: parameters defined in the parameters file
            persist_high: longest length of the persistent group
            perist_low: shortest length of the persistent group
            hour_low: earliest hour for calibrations to have occurred
            hour_high: latest hour for calibrations to have occurred
    Returns:
        calib: data frame of booleans indicating whether the conditions were met for a possible calibration event.
        calib_dates: datetimes for which the conditions were met for a possible calibration event.
    """
    if 'persist_grp' in df:
        temp = df[['observed', 'anomaly', 'persist_grp']].copy(deep=True)
        for i in range(1, max(temp['persist_grp']) + 1):
            temp['persist_grp'][temp.loc[temp['persist_grp'].shift(-1) == i].index[0]] = i
            if ((len(temp['persist_grp'][temp['persist_grp'] == i]) >= calib_params.persist_low) and
                    (len(temp['persist_grp'][temp['persist_grp'] == i]) <= calib_params.persist_high)):
                temp['anomaly'][temp['persist_grp'] == i] = True
    else:
        temp = df[['observed', 'anomaly']].copy(deep=True)
        temp['persist_grp'] = (temp.observed.diff(1) == 0)
        temp['persist_grp'] = anomaly_utilities.anomaly_events(temp['persist_grp'], 0, 1)
        for i in range(1, max(temp['persist_grp']) + 1):
            temp['persist_grp'][temp.loc[temp['persist_grp'].shift(-1) == i].index[0]] = i
            if ((len(temp['persist_grp'][temp['persist_grp'] == i]) >= calib_params.persist_low) and
                    (len(temp['persist_grp'][temp['persist_grp'] == i]) <= calib_params.persist_high)):
                temp['anomaly'][temp['persist_grp'] == i] = True

    dayofweek = temp.index.dayofweek
    hour = temp.index.hour
    business = temp.iloc[((dayofweek == 0) | (dayofweek == 1) | (dayofweek == 2) | (dayofweek == 3) | (dayofweek == 4))
                         & (hour >= calib_params.hour_low) & (hour <= calib_params.hour_high)]
    calib = pd.DataFrame(index=temp.index)
    calib['anomaly'] = False
    calib['anomaly'].loc[business[business['anomaly']].index] = True
    calib_dates = calib[calib['anomaly']].index

    return calib, calib_dates


def calib_overlap(sensor_names, input_array, calib_params):
    """
    calib_overlap seeks to identify calibration events by identifying where overlaps occur between multiple sensors.
    Calls the calib_detect function to identify events with a defined persistence length during certain days of the
    week (M-F) and hours of the day.
    Arguments:
        sensor_names: list of sensors to be considered for overlap.
        input_array: array of data frames each with columns:
            'observed' of observed data
            'anomaly' booleans where True=1 corresponds to anomalies
            'persist_grp' with indices of persistent groups (output of the persist function)
        calib_params: parameters defined in the parameters file
            persist_high: longest length of the persistent group
            perist_low: shortest length of the persistent group
            hour_low: earliest hour for calibrations to have occurred
            hour_high: latest hour for calibrations to have occurred
    Returns:
        all_calib: array of data frames (one for each sensor) of booleans indicating whether the conditions were met
        for a possible calibration event.
        all_calib_dates: array of datetimes (one for each sensor) for which the conditions were met for a possible
        calibration event.
        df_all_calib: data frame with columns for each sensor observations, columns of booleans for each sensor
        indicating whether a calibration event may have occurred, and a column 'all_calib' that indicates if the
        conditions were met for all sensors.
        calib_dates_overlap: datetimes for which the conditions were met for a possible calibration event for
        all sensors.
    """
    all_calib = dict()
    all_calib_dates = dict()
    df_all_calib = pd.DataFrame(index=input_array[sensor_names[0]].index)
    df_all_calib['all_calib'] = True
    for snsr in sensor_names:
        calib, calib_dates = calib_persist_detect(input_array[snsr], calib_params)
        all_calib[snsr] = calib
        all_calib_dates[snsr] = calib_dates
        df_all_calib[snsr] = input_array[snsr]['observed']
        df_all_calib[snsr + '_calib'] = all_calib[snsr]['anomaly']
        df_all_calib[snsr + '_event'] = anomaly_utilities.anomaly_events(df_all_calib[snsr + '_calib'], wf=1, sf=1)
        df_all_calib['all_calib'] = np.where(df_all_calib['all_calib'] & (df_all_calib[snsr + '_event'] != 0), True, False)
    calib_dates_overlap = df_all_calib[df_all_calib['all_calib']].index

    return all_calib, all_calib_dates, df_all_calib, calib_dates_overlap


def find_gap(observed, calib_date, hours=2, show_shift=False):
    """
    find_gap determines the gap value of a calibration event based on the largest single difference.
    Uses a given time stamp and searches within a designated window. Accounts for large spikes
    immediately following the difference.
    Args:
        observed: time series of observations
        calib_date: datetime for performing the correction
        hours: window on each side of the calib_date to consider for finding the greatest difference. To use the exact datetime and not consider a window, use hours=0.
        show_shift: boolean indicating if subset used to determine the gap value should be output
    Returns:
        gap: the resulting value of the gap
        end: the ending timestamp corresponding to applying the gap. Used as input to linear drift correction.
        shifted: the subset of data used to determine the gap value with the gap applied

    """
    # time window to consider
    subset = observed.loc[
             pd.to_datetime(calib_date) - pd.Timedelta(hours=hours):
             pd.to_datetime(calib_date) + pd.Timedelta(hours=hours)
             ]
    # shift index by 1
    shifted = subset.shift(-1)
    # timestamp of greatest difference
    maxtime = abs(subset.diff()).idxmax()
    # if the two subsequent signs are different, then add them together for the gap. This should address/eliminate
    #   spikes following the calibration.
    if subset.diff().loc[maxtime] * shifted.diff().loc[maxtime] < 0:
        gap = subset.diff().loc[maxtime] + shifted.diff().loc[maxtime]
    else:
        gap = subset.diff().loc[maxtime]
    # the last timestamp for the shift to occur
    end = abs(shifted.diff()).idxmax()

    if show_shift:
        # shift subset to compare
        shifted = subset.loc[subset.index[0]:end] + gap
        return gap, end, shifted
    else:
        return gap, end


def lin_drift_cor(observed, start, end, gap, replace=True):
    """
   lin_drift_cor performs linear drift correction on data. Typical correction for calibration events. This function operates on the basis of a single event.
   Arguments:
       observed: time series of observations
       start: datetime for the beginning of the correction
       end: datetime for the end of the correction
       gap: gap value that determines the degree of the shift, which occurs at the end date.
       replace: indicates whether the values of the correction should replace the associated values in the data frame
   Returns:
       result: data frame of corrected values
       observed: time series of observations with corrected values if replace was selected
    """
    # todo: ignore - 9999 values

    result = pd.DataFrame(index=observed.loc[start:end].index)
    result['ldc'] = pd.Series(dtype=float)
    for i in range(len(observed.loc[start:end])):
        # y = original_value[i] - i * gap / total_num_points
        y = observed.iloc[observed.index.get_loc(start) + i] + gap/(len(observed.loc[start:end])-1) * i
        result['ldc'].iloc[i] = y
    if replace:
        observed.loc[start:end] = result['ldc']

    return result, observed
