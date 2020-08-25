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
pd.options.mode.chained_assignment = None


def get_data(site, sensor, year, path=""):
    """
    get_data imports a single year of data based on files named by site, sensor/variable, and year.
        Labels data as anomalous. Generates a series from the data frame.
    site (string): name of the data collection site
    sensor (string): name of the sensor/variable data of interest
    year (integer): the year of interest
    path (string): path to .csv file containing the data of interest
    df_full (pandas DataFrame): has 3 columns for each variable/sensor in the .csv file
        column names are in the format: '<sensor>', '<sensor>_cor', '<sensor>_qual'
    df (pandas DataFrame): has 3 columns for the variable/sensor of interest: 'raw', 'cor', 'labeled_anomaly'
    """
    # TODO: make sensors input argument a list and output df with multiple normal_lbl columns.
    if path == "":
        path = os.getcwd() + "/"
    df_full = pd.read_csv(path + site + str(year) + ".csv",
                          skipinitialspace=True,
                          engine='python',
                          header=0,
                          index_col=0,
                          parse_dates=True,
                          infer_datetime_format=True)
    # create data frame with raw, corrected, and labeled data
    df = pd.DataFrame(index = df_full.index)
    df['raw'] = df_full[[sensor]]
    df['cor'] = df_full[[sensor + "_cor"]]
    df['labeled_anomaly'] = ~df_full[sensor + "_qual"].isnull()

    return df_full, df


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

    return labeled_in_detected, detected_in_labeled, valid_detections, invalid_detections


def metrics(df, valid_detections, invalid_detections):
    """Calculates metrics for anomaly detection.
    Requires a dataframe with columns for detected_events and labeled_events.
    Requires lists of valid_detections and invalid_detections."""
    TruePositives = sum(df['detected_event'].value_counts()[valid_detections])
    FalseNegatives = sum(df['labeled_event'].value_counts()[1:]) - TruePositives
    FalsePositives = sum(df['detected_event'].value_counts()[invalid_detections])
    TrueNegatives = len(df['detected_event']) - TruePositives - FalseNegatives - FalsePositives

    PRC = PPV = TruePositives / (TruePositives + FalsePositives)
    NPV = TrueNegatives / (TrueNegatives + FalseNegatives)
    ACC = (TruePositives + TrueNegatives) / len(df['detected_anomaly'])
    RCL = TruePositives / (TruePositives + FalseNegatives)
    f1 = 2.0 * (PRC * RCL) / (PRC + RCL)
    f2 = 5.0 * TruePositives / (5.0 * TruePositives + 4.0 * FalseNegatives + FalsePositives)
    # ACC = (TruePositives+TrueNegatives)/(TruePositives+TrueNegatives+FalsePositives+FalseNegatives)

    return TruePositives, FalseNegatives, FalsePositives, TrueNegatives, PRC, PPV, NPV, ACC, RCL, f1, f2


def group_bools(df):
    """ group_bools is used for anomaly correction, indexing each grouping of anomalies/normal points
    df is a data frame with required column:
    'detected_anomaly': boolean array of classified data points
    Outputs:
    df with additional column: 'groups' of boolean groupings
    """

    # initialize the 'groups' column to zeros
    df['groups'] = 0
    # initialize placeholder for boolean state of previous group
    last = df.iloc[0]['detected_anomaly']
    # initialize the group index to zero
    gi = 0

    # loop over every row in dataframe
    for i in range(0, len(df['groups'])):

        # if the anomaly bool has changed
        if last != df.iloc[i]['detected_anomaly']:
            gi += 1  # increment the group index
        # assign this row to the group index
        df.iloc[i, df.columns.get_loc('groups')] = gi

        # update last boolean state
        last = df.iloc[i]['detected_anomaly']

    return df


def xfade(xfor, xbac):
    """ the xfade ("cross-fade") function blends two data sets of matching length
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

