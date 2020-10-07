################################
# ANOMALY UTILITIES #
################################
# This code includes utilities for performing anomaly detection.
# A function is defined for accessing data.
# Functions are defined for determining events and comparing them to events in labeled data.
# A windowing function widens the window of anomaly detection.
# A metrics function compares detection compared to labeled data and outputs performance metrics.
# Threshold functions dvelop either a constant threshold or a dynamic threshold based on model residuals.

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings
pd.options.mode.chained_assignment = None


def get_data(site, sensor, year, path=""):
    """
    get_data imports a single year of data based on files named by site, sensor/variable, and year.
        Includes labeling of data as anomalous.
    site (string): name of the data collection site
    sensor (list): name(s) of the sensor/variable data of interest
    year (list): the year(s) of interest
    path (string): path to .csv file containing the data of interest
    Outputs:
    df_full (pandas DataFrame): has 3 columns for each variable/sensor in the .csv file
        column names are in the format: '<sensor>', '<sensor>_cor', '<sensor>_qual'
    sensor_array (array of pandas DataFrames): each data frame has 3 columns for the 
        variable/sensor of interest: 'raw', 'cor', 'labeled_anomaly'.
    """
    if path == "":
        path = os.getcwd() + "/"
    df_full = pd.DataFrame()
    for i in range(0, len(year)):
        df_year = pd.read_csv(path + site + str(year[i]) + ".csv",
                         skipinitialspace=True,
                         engine='python',
                         header=0,
                         index_col=0,
                         parse_dates=True,
                         infer_datetime_format=True)
        df_full = pd.concat([df_full, df_year], axis=0)
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
    anomaly_events searches through data and counts groups of immediately consecutively labeled data points
        as anomalous events. Input to the windowing function.
    anomaly (boolean array): labeled or detected anomalies where True (1) = anomalous data point.
        e.g., 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 1 0 0 0 1 1 1 1
    wf (assumed to be a positive integer): a widening factor that is used to determine how much to widen each event
        before and after the true values. Default =1 adds a single point to be anomalous before/after the labeled point.
    sf (assumed to be a ratio between 0.0-1.0): a significance factor - used to warn user when an event size is greater
        than this ratio compared to the entire data set. Default = 0.05 = 5%.
    Outputs:
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
    """
    compare_labeled_detected compares anomalous events that are technician labeled and machine detected.
    Labeled and detected data may be widened to increase the window of overlap.
    df is a data frame with required columns:
    'labeled_event': array of numbered events based on expert labeled anomalies.
    'detected_event': array of numbered events based on machine detected anomalies.
    Outputs:
    labeled_in_detected: boolean indicating the labeled points that appear in the detections.
    detected_in_labeled: boolean indicating the detected point that appear in the labeled.
    valid_detections: boolean indicating whether the detections were also present in the labeled.
    invalid_detections: boolean indicating detections that were not present in the labeled.
    """
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
    """
    metrics evaluates detector performance by comparing machine detected anomalies to technician labeled anomalies.
    df is a data frame with required columns:
    'detected_event': boolean corresponding to machine detected anomalies where True (1) = anomalous data point.
    'labeled_event': boolean corresponding to technician labeled anomalies where True (1) = anomalous data point.
    valid_detections: output from compare function.
    invalid_detections: output from compare function.
    Outputs:
    true_positives is the count of valid detections.
    false_negatives is the count of missed events.
    false_positives is the count of incorrect detections.
    true_negatives is the count of valid undetected data.
    prc is the precision of detections.
    npv is negative predicted value.
    acc is accuracy of detections.
    rcl is recall of detections.
    f1 is a statistic that balances true positives and false negatives.
    f2 is a statistic that gives more weight to true positives.
    """
    metrics = MetricsContainer()
    metrics.true_positives = sum(df['detected_event'].value_counts()[valid_detections])
    metrics.false_negatives = sum(df['labeled_event'].value_counts()[1:]) - metrics.true_positives
    metrics.false_positives = sum(df['detected_event'].value_counts()[invalid_detections])
    metrics.true_negatives = \
        len(df['detected_event']) - metrics.true_positives - metrics.false_negatives - metrics.false_positives

    metrics.prc = metrics.PPV = metrics.true_positives / (metrics.true_positives + metrics.false_positives)
    metrics.npv = metrics.true_negatives / (metrics.true_negatives + metrics.false_negatives)
    metrics.acc = (metrics.true_positives + metrics.true_negatives) / len(df['detected_anomaly'])
    metrics.rcl = metrics.true_positives / (metrics.true_positives + metrics.false_negatives)
    metrics.f1 = 2.0 * (metrics.prc * metrics.rcl) / (metrics.prc + metrics.rcl)
    metrics.f2 = 5.0 * metrics.true_positives / \
                 (5.0 * metrics.true_positives + 4.0 * metrics.false_negatives + metrics.false_positives)

    return metrics


def group_bools(df):
    """
    group_bools indexes each grouping of anomalies (1) and normal points (0) as numbered sets.
    Used for anomaly correction.
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
    """
    xfade ("cross-fade") blends two data sets of matching length with a ramp function (weighted average).
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


def set_dynamic_threshold(residuals, window_sz=96, alpha=0.01, min_range=0.0):
    """
    set_dynamic_threshold determines a threshold for each point based on the local confidence interval
    considering the model residuals looking forward and backward window_sz steps.
    residuals is a series like object or a data frame.
    alpha is a scalar between 0 and 1 representing the acceptable uncertainty.
    window_sz is an integer representing how many data points to use in both directions.
        default = 96 for one day for 15-minute data.
    Outputs:
    threshold is data frame of pairs of low and high values.
    """
    threshold = []  # initialize empty list to hold thresholds
    z = norm.ppf(1 - alpha / 2)

    # if the window size parameter is too big for this data set
    if (window_sz > len(residuals)):
        print("WARNING: in set_dynamic_threshold(), window_sz > len(data)! Reducing window_sz.")
        window_sz = len(residuals)  # reduce the window to the max allowable

    # loop through data and add each threshold pair
    for i in range(0, len(residuals)):
        if (window_sz > i):  # index is closer than window size to left edge of data
            lo = 0
        else:  # look back as far as the window size
            lo = i - window_sz
        if (i + window_sz > len(residuals)):  # index is close to right edge of data
            hi = len(residuals)
        else:  # look forward as far as the window size
            hi = i + window_sz

        # calculate the range of probable values using given alpha
        mean = residuals[lo:(hi+1)].mean()
        sigma = residuals[lo:(hi+1)].std()
        th_range = z*sigma
        if (th_range < min_range):
            th_range = min_range
        # append pair of upper and lower thresholds
        threshold.append([mean - th_range, mean + th_range])

    threshold = pd.DataFrame(threshold, columns=['low', 'high'])

    return threshold


def set_cons_threshold(model_fit, alpha_in):
    """
    set_cons_threshold determines a threshold based on confidence interval and specified alpha for an ARIMA model.
    model_fit is a SARIMAX model object.
    alpha_in is a scalar between 0 and 1 representing the acceptable uncertainty.
    Outputs:
    threshold is a single value.
    """
    predict = model_fit.get_prediction()
    predict_ci = predict.conf_int(alpha=alpha_in)
    predict_ci.columns = ["lower", "upper"]
    predict_ci["lower"][0] = predict_ci["lower"][1]

    # This gives a constant interval for all points.
    # Could also try a threshold to maximize F2, but that requires having labeled data. Could base on a portion of data?
    thresholds = predict[0] - predict_ci["lower"]
    threshold = thresholds[-1]

    return threshold


def detect_anomalies(observed, predictions, residuals, threshold, summary):
    """
    detect_anomalies compares model residuals to thresholds to determine which points are anomalous.
    observed is a data frame or series of the observed data.
    predictions are a series of model predictions.
    residuals are a series of model residuals.
    threshold is a data frame with the columns 'lower' and 'upper' corresponding to the acceptable range of the residual.
    Outputs:
    detections is a data frame with observations, predictions, residuals, anomalies,
        a boolean where True (1) = anomalous data point
    """
    detections = pd.DataFrame(observed)
    detections['prediction'] = np.array(predictions)
    detections['residual'] = np.array(residuals)
    detections['anomaly'] = (detections['residual'] < threshold['low']) | (threshold['high'] < detections['residual'])
    # anomalies = test_score_df[test_score_df.anomaly == True]

    # output summary
    if summary:
        print('\n\n\nratio of detections: %f' % ((sum(detections.anomaly) / len(detections.anomaly)) * 100), '%')

    return detections


def detect_dyn_anomalies(residuals, threshold, summary=True):
    """Compares residuals to threshold to identify anomalies. Can use set threshold level or threshold
    determined by set_threshold function."""
    # DETERMINE ANOMALIES
    detected_anomaly = (residuals[0] < threshold['low']) | (threshold['high'] < residuals[0])  # gives bools
    # output summary
    if summary:
        print('\n\n\nratio of detections: %f' % ((sum(detected_anomaly)/len(detected_anomaly))*100), '%')

    return detected_anomaly
  