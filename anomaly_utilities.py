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
    """Imports a single year of data based on files named by site, sensor/variable, and year.
    Labels data as anomalous. Generates a series from the data frame."""
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

    return df_full, df #df_raw, df_cor, normal_lbl, srs


def anomaly_events(anomaly):
    """anomaly is a boolean array of labeled or detected anomalies where True (1) = anomalous data point.
    e.g., 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 1 0 0 0 1 1 1 1
    event is an array of enumerations corresponding to each group of consecutive anomalous points.
    e.g., 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 2 2 2 0 0 0 3 0 0 0 4 4 4 4
    Function searches through data and counts groups of immediately consecutively labeled data points
    as anomalous events. Input to the windowing function. """
    # TODO: uses +- 1 to widen window before and after the event. Should make this a parameter.
    # TODO: if an event is > some percent of the data, output a warning.
    event_count = 0
    event = []
    event.append(0)
    for i in range(1, len(anomaly)):
        if anomaly[i]:
            if ~anomaly[i-1]:
                event_count += 1
                event[i-1] = event_count
            event.append(event_count)
        else:
            if anomaly[i-1]:
                event.append(event_count)
            else:
                event.append(0)

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
    Input to the windowing function. Compares the widened/windowed events between labels and detections."""

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

