################################
# ANOMALY EVENT UTILITIES #
################################
# This code includes utilities for performing anomaly detection.
# A function is defined for accessing data.
# Functions are defined for determining events and comparing them to events in labeled data.
# A windowing function widens the window of anomaly detection.
# A metrics function compares detection compared to labeled data and outputs perfomance metrics.

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
pd.options.mode.chained_assignment = None


def get_data(site, sensor, year, path=""):
    """
    get_data imports a single year of data based on files named by site, sensor/variable, and year.
        Includes labeling of data as anomalous.
    site (string): name of the data collection site
    sensor (list): name(s) of the sensor/variable data of interest
    year (integer): the year of interest
    path (string): path to .csv file containing the data of interest
    df_full (pandas DataFrame): has 3 columns for each variable/sensor in the .csv file
        column names are in the format: '<sensor>', '<sensor>_cor', '<sensor>_qual'
    sensor_array (array of pandas DataFrames): each data frame has 3 columns for the 
        variable/sensor of interest: 'raw', 'cor', 'labeled_anomaly'.
    """
    if path == "":
        path = os.getcwd() + "/"
    df_full = pd.read_csv(path + site + str(year) + ".csv",
                          skipinitialspace=True,
                          engine='python',
                          header=0,
                          index_col=0,
                          parse_dates=True,
                          infer_datetime_format=True)
    # create data frames with raw, corrected, and labeled data
    sensor_array = []
    for i in range(0, len(sensor)):
        df = []
        df = pd.DataFrame(index=df_full.index)
        df['raw'] = df_full[[sensor[i]]]
        df['cor'] = df_full[[sensor[i] + "_cor"]]
        df['labeled_anomaly'] = ~df_full[sensor[i] + "_qual"].isnull()
        sensor_array.append(df)
    sensor_array = dict(zip(sensor, sensor_array))

    return df_full, sensor_array


def anomaly_events(anomaly, wf=1, sf=0.05):
    """
    anomaly_events function searches through data and counts groups of immediately consecutively labeled data points
        as anomalous events. Input to the windowing function.
    anomaly (boolean array): labeled or detected anomalies where True (1) = anomalous data point.
        e.g., 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 1 0 0 0 1 1 1 1
    wf (assumed to be a positive integer): a widening factor that is used to determine how much to widen each event
        before and after the true values.
    sf (assumed to be a ratio between 0.0-1.0): a significance factor - used to warn user when an event size is greater
        than this ratio compared to the entire data set
    event (integer array): enumerated event labels corresponding to each widened group of consecutive anomalous points.
        e.g., 0 0 0 1 1 1 1 1 1 0 0 0 0 0 2 2 2 2 2 0 3 3 3 0 4 4 4 4 4
    """
    # initialize event variables
    event_count = 0
    event = []
    # handle the first wf data points in the entire time series
    for i in range(0, wf):
        event.append(0)

    # search through data assigning each point an event (positive integer) or 0 for points not belonging to an event
    for i in range(wf, len(anomaly) - wf):
        if (sum(anomaly[i-wf:i+wf+1]) != 0):  # if there are anomalies within the wf window
            if (event[-1] == 0):  # if the last event value is 0, then a new event has been encountered
                event_count += 1  # increment the event counter
            event.append(event_count)  # append the event array with the current event number
        else:  # no anomalies are within the wf window
            event.append(0)  # this data point does not belong to an event, append 0

    # handle the last wf data points in the entire time series
    for i in range(0, wf):
        event.append(0)

    # determine if an event is greater than the significance factor
    event_values = pd.Series(event).value_counts()
    for i in range(1, len(event_values)):
        if event_values[i] > (sf * len(event)):
            print("WARNING: an event was found to be greater than the significance factor!")

    return event


class CompareDetectionsContainer:
    pass


def compare_labeled_detected(df):
    """ df is a data frame with required columns:
    'labeled_event': array of numbered events based on expert labeled anomalies.
    'detected_event': array of numbered events based on machine detected anomalies.
    Outputs:
    labeled_in_detected
    detected_in_labeled
    valid_detections
    invalid_detections
    Compares the widened/windowed events between labels and detections."""

    # generate lists of detected anomalies and valid detections
    compare = CompareDetectionsContainer()
    labeled_in_detected = [0]
    detected_in_labeled = [0]
    valid_detections = [0]
    invalid_detections = [0]
    for i in range(0, len(df['detected_event'])):
        if (0 != df['detected_event'][i]):  # anomaly detected
            if (0 != df['labeled_event'][i]):  # labeled as anomaly
                if (labeled_in_detected[-1] != df['labeled_event'][i]):  # if not already in list of detected anomalies
                    labeled_in_detected.append(df['labeled_event'][i])
                if (detected_in_labeled[-1] != df['detected_event'][i]):
                    detected_in_labeled.append(df['detected_event'][i])
                if (valid_detections[-1] != df['detected_event'][i]):  # if not already in list of valid detections
                    valid_detections.append(df['detected_event'][i])

    det_ind = 0
    for i in range(1, max(df['detected_event'])):
        if (det_ind < len(valid_detections)):
            if i == valid_detections[det_ind]:
                det_ind += 1
            else:
                invalid_detections.append(i)

    labeled_in_detected.pop(0)
    valid_detections.pop(0)
    invalid_detections.pop(0)

    compare.labeled_in_detected = labeled_in_detected
    compare.valid_detections = valid_detections
    compare.invalid_detections = invalid_detections
    compare.detected_in_labeled = detected_in_labeled

    return compare


class MetricsContainer:
    pass


def metrics(df, valid_detections, invalid_detections):
    metrics = MetricsContainer()
    metrics.TruePositives = sum(df['detected_event'].value_counts()[valid_detections])
    metrics.FalseNegatives = sum(df['labeled_event'].value_counts()[1:]) - metrics.TruePositives
    metrics.FalsePositives = sum(df['detected_event'].value_counts()[invalid_detections])
    metrics.TrueNegatives = \
        len(df['detected_event']) - metrics.TruePositives - metrics.FalseNegatives - metrics.FalsePositives

    metrics.PRC = metrics.PPV = metrics.TruePositives / (metrics.TruePositives + metrics.FalsePositives)
    metrics.NPV = metrics.TrueNegatives / (metrics.TrueNegatives + metrics.FalseNegatives)
    metrics.ACC = (metrics.TruePositives + metrics.TrueNegatives) / len(df['detected_anomaly'])
    metrics.RCL = metrics.TruePositives / (metrics.TruePositives + metrics.FalseNegatives)
    metrics.f1 = 2.0 * (metrics.PRC * metrics.RCL) / (metrics.PRC + metrics.RCL)
    metrics.f2 = \
        5.0 * metrics.TruePositives / (5.0 * metrics.TruePositives + 4.0 * metrics.FalseNegatives + metrics.FalsePositives)

    return metrics


def group_bools(df):
    """ group_bools is used for anomaly correction, indexing each grouping of anomalies and normal points as numbered sets.
    df is a data frame with required column:
    'detected_anomaly': boolean array of classified data points
    Outputs:
    df with additional column: 'group' of boolean groupings
    """

    # initialize the 'group' column to zeros
    df['group'] = 0
    # initialize placeholder for boolean state of previous group
    last = df.iloc[0]['detected_event']
    # initialize the group index to zero
    gi = 0

    # loop over every row in dataframe
    for i in range(0, len(df['group'])):

        # if the anomaly bool has changed
        if last != df.iloc[i]['detected_event']:
            gi += 1  # increment the group index
        # assign this row to the group index
        df.iloc[i, df.columns.get_loc('group')] = gi

        # update last boolean state
        last = df.iloc[i]['detected_event']

    return df


def xfade(xfor, xbac):
    """ xfade ("cross-fade") blends two data sets of matching length with a ramp function (weighted average).
    xfor is the data to be more weighted at the front
    xbac is the data to be more weighted at the back
    Outputs:
    x is the blended data
    """
    # if arrays are not matching in length
    if (len(xfor) != len(xbac)):
        # send error message
        print("ERROR in xfade() call: mismatched array lengths!")
    else:
        # initialize a weighting function
        fader = []

        # loop over the length of data
        for i in range(0, len(xfor)):
            # calculate the weights at each index
            fader.append((i + 1) / (len(xfor)+1))
        # fader should be a ramp with positive slope between 0.0 and 1.0
        # use this to fade the back data
        xbac_faded = np.multiply(xbac, fader)

        # now flip the ramp to negative slope and fade the front data
        fader = np.flip(fader)
        xfor_faded = np.multiply(xfor, fader)

        # add the results
        x = xfor_faded + xbac_faded
    return x


def set_dynamic_threshold(data, alpha, window_sz):
    """Determines a threshold based on the local confidence interval, 
    considering the data looking forward and backward window_sz steps.
    data is a series like object
    alpha is a scalar between 0 and 1 representing the acceptable uncertainty
    window_sz is an integer representing how many data points to use in both directions
    the return value is an array of pairs
    """
    threshold = []  # initialize empty list to hold thresholds
    
    # if the window size parameter is too big for this data set
    if (window_sz > len(data)):
        print("WARNING: in set_dynamic_threshold(), window_sz > len(data)! Reducing window_sz.")
        window_sz = len(data)  # reduce the window to the max allowable

    # loop through data and add each threshold pair
    for i in range(0,len(data)):
        if(window_sz > i):  # index is closer than window size to left edge of data
            lo = 0
        else:  # look back as far as the window size
            lo = i - window_sz
        if (i + window_sz > len(data)):  # index is close to right edge of data
            hi = len(data)
        else:  # look forward as far as the window size
            hi = i + window_sz

        # calculate the range of probable values using given alpha
        mean = data[lo:hi].mean()
        sigma = data[lo:hi].std()
        z = norm.ppf(1-alpha/2)
        n = len(data[lo:hi])
        # append pair of upper and lower thresholds
        threshold.append([mean - z*sigma/np.sqrt(n), mean + z*sigma/np.sqrt(n)])
    return threshold

