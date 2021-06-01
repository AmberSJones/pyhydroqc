################################
# ANOMALY UTILITIES #
################################
# This file includes utilities for performing anomaly detection.
# A function is defined for accessing data.
# Functions are defined for determining events and comparing them to events in labeled data.
# A windowing function widens the window of anomaly detection.
# A metrics function compares detection compared to labeled data and outputs performance metrics.
# Threshold functions develop either a constant threshold or a dynamic threshold based on model residuals.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

pd.options.mode.chained_assignment = None


def get_data(sensors, filename="", site="", years="", path=""):
    """
    get_data imports time series data from csv files. Files may specified explicitly by file name, or a series of files
    may be imported that follow a naming pattern with site and year (e.g. "MainStreet2014.csv").
    Files should have columns corresponding to each sensor. If technician labels and corrections exist, they may be
    imported by naming columns sensor_cor and labeled_anomaly.
    Arguments:
        sensors: list of name(s) of the sensor/variable data of interest. These must be the column names in data file(s).
        filename: string of the file name containing input data
        site: string of name of the data collection site
        years: list of the year(s) of interest
        path: path to .csv files containing the data of interest
    Returns:
        sensor_array: array of pandas DataFrames, each with 3 columns for the variable/sensor of interest:
        'raw', 'cor', 'labeled_anomaly'.
    """
    if path == "":  # use default directory when none is provided
        path = os.getcwd() + "/"  # default directory is ./
    df_full = pd.DataFrame()  # start with empty dataframe and concatenate each file

    if filename:
        df_full = pd.read_csv(path + filename,
                              skipinitialspace=True,
                              engine='python',
                              header=0,
                              index_col=0,
                              parse_dates=True,
                              infer_datetime_format=True)
    if years:
        for yr in years:  # loop over each file
            df_year = pd.read_csv(path + site + str(yr) + ".csv",
                                  skipinitialspace=True,
                                  engine='python',
                                  header=0,
                                  index_col=0,
                                  parse_dates=True,
                                  infer_datetime_format=True)
            df_full = pd.concat([df_full, df_year], axis=0)

    # create data frames with raw, corrected, and labeled data (if the corrected and labeled data exist)
    sensor_array = dict()
    for snsr in sensors:
        df = []
        df = pd.DataFrame(index=df_full.index)
        df['raw'] = df_full[snsr]

        # if corrected data is available in dataset
        if snsr + '_cor' in df_full.columns:
            df['cor'] = df_full[snsr + '_cor']
        if snsr + "_qual" in df_full.columns:
            df['labeled_anomaly'] = ~df_full[snsr + '_qual'].isnull()
        sensor_array[snsr] = df

    return sensor_array


def anomaly_events(anomaly, wf=1, sf=0.05):
    """
    anomaly_events groups consecutively labeled data points as anomalous events by adding an index.
    Events may also be widened.
    Arguments:
        anomaly: boolean series of labeled or detected anomalies where True (1) = anomalous data point.
            e.g., 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 1 1 1 0 0 0 1 0 0 0 1 1 1 1
        wf: positive integer, a widening factor that is used to determine how much to widen each event
            before and after the true values. Default = 1 adds a single anomalous point before/after the labeled point.
        sf: ratio between 0.0-1.0. a significance factor used to warn user when an event size is greater
            than this ratio compared to the entire data set. Default = 0.05 = 5%.
    Returns:
        event: integer series of enumerated event labels corresponding to each widened group of consecutive anomalous points.
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
        if (sum(anomaly[i - wf:i + wf + 1]) != 0):  # if there are anomalies within the wf window
            if (len(event) == 0):  # if event is empty
                event_count += 1  # increment the event counter
            elif (event[-1] == 0):  # if the last event value is 0, then a new event has been encountered
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


def assign_cm(val, len, wf):
    """
    assign_cm is a simple helper function used in compare_events
    Arguments:
        val: string value to specify which area of the confusion matrix this point belongs to: 'tp', 'fp', or 'fn'
        len: how long the total array should be
        wf: integer widening factor that determines how many points should be turned into 'tn' on both edges
    Returns:
        cm: series of length len, with wf 'tn' at the beginning and the end filed with val in between
    """
    cm = ['tn' for i in range(len)]
    for i in range(wf, len - wf):
        cm[i] = val
    return cm


def compare_events(df, wf=1):
    """
    compare_events compares anomalous events that are technician labeled and machine detected.
    Labeled and detected data may have been widened to increase the window of overlap.
    Arguments:
        wf: integer widening factor used when generating events
        df: data frame with required columns:
            'labeled_anomaly': series of booleans based on expert labeled anomalies.
            'detected_anomaly': series of machine detected anomaly booleans based on modeling.
            'labeled_event': series of numbered events based on expert labeled anomalies
            (output of anomaly_events function).
            'detected_event': series of numbered events based on machine detected anomalies
            (output of anomaly_events function).
    Returns:
        df: orginal data frame with additional columns:
            'grp': a new column representing the indices of event groups.
            'conf_mtx': a new column that gives the confusion matrix value for each data point.
    """

    # initialize variables
    grp_idx = 0
    df['grp'] = -1  # initialize to error flag
    df['conf_mtx'] = 'tn'
    prev_la = df['labeled_event'][0]
    prev_da = df['detected_event'][0]
    prev_gi = 0

    for i in range(0, len(df['labeled_event'])):  # for every row of data

        # if this row is an event transition case
        if (prev_la != df['labeled_event'][i] or prev_da != df['detected_event'][i]):

            # if coming from a true negative case
            if (prev_la == 0 and prev_da == 0):
                grp_idx += 1

            # if entering a true negative case
            elif (df['labeled_event'][i] == 0 and df['detected_event'][i] == 0):
                grp_idx += 1

            # if it's a complete flip-flop case
            elif (prev_la != df['labeled_event'][i] and prev_da != df['detected_event'][i]):
                grp_idx += 1

        df['grp'][i] = grp_idx  # update the group index for this row

        # if this row is a group transition
        if (grp_idx != prev_gi):
            # add confusion matrix category to previous group

            # if this event group is both labeled and detected as anomalous case
            if (any(df['detected_event'][df['grp'] == prev_gi]) and any(df['labeled_event'][df['grp'] == prev_gi])):
                # True Positive group
                df['conf_mtx'][df['grp'] == prev_gi] = assign_cm('tp', len(df[df['grp'] == prev_gi]), wf)
            # if this event group is detected as anomalous but not labeled
            elif (any(df['detected_event'][df['grp'] == prev_gi])):
                # False Positive group
                df['conf_mtx'][df['grp'] == prev_gi] = assign_cm('fp', len(df[df['grp'] == prev_gi]), wf)
            # if this event group is labeled as anomalous but not detected
            elif (any(df['labeled_event'][df['grp'] == prev_gi])):
                # False Negative group
                df['conf_mtx'][df['grp'] == prev_gi] = assign_cm('fn', len(df[df['grp'] == prev_gi]), wf)

        # update previous state variables
        prev_la = df['labeled_event'][i]
        prev_da = df['detected_event'][i]
        prev_gi = grp_idx

    df = df.drop(columns='grp')  # delete group index column

    return df


class MetricsContainer:
    pass


def metrics(df):
    """
    metrics evaluates the performance of anomaly detection comparing detected anomalies to technician labeled anomalies.
    Output is contained in an object of the class MetricsContainer.
    Arguments:
        df: data frame with required column:
            'conf_mtx': strings corresponding to confusion matrix categories: tp, tn, fp, fn
    Returns:
        true_positives: count of data points from valid detections.
        false_negatives: count of data points from missed events.
        false_positives: count of data points from incorrect detections.
        true_negatives: count of valid undetected data.
        prc: is the precision of detections.
        npv: negative predicted value.
        acc: accuracy of detections.
        rcl: recall of detections.
        f1: statistic that balances true positives and false negatives.
        f2: statistic that gives more weight to true positives.
    """
    metrics = MetricsContainer()
    metrics.true_positives = len(df['conf_mtx'][df['conf_mtx'] == 'tp'])
    metrics.false_negatives = len(df['conf_mtx'][df['conf_mtx'] == 'fn'])
    metrics.false_positives = len(df['conf_mtx'][df['conf_mtx'] == 'fp'])
    metrics.true_negatives = len(df['conf_mtx'][df['conf_mtx'] == 'tn'])
    metrics.prc = metrics.ppv = metrics.true_positives / (metrics.true_positives + metrics.false_positives)
    metrics.npv = metrics.true_negatives / (metrics.true_negatives + metrics.false_negatives)
    metrics.acc = (metrics.true_positives + metrics.true_negatives) / len(df['conf_mtx'])
    metrics.rcl = metrics.true_positives / (metrics.true_positives + metrics.false_negatives)
    metrics.f1 = 2.0 * (metrics.prc * metrics.rcl) / (metrics.prc + metrics.rcl)
    metrics.f2 = 5.0 * metrics.true_positives / \
                 (5.0 * metrics.true_positives + 4.0 * metrics.false_negatives + metrics.false_positives)

    return metrics


def event_metrics(df):
    """
    event_metrics calculates an alternative set of metrics where every event is treated with equal weight
    regardless of size.
    Arguments:
        df: data frame with required columns:
            'conf_mtx': strings corresponding to confusion matrix categories: tp, tn, fp, fn
    Returns:
        true_positives: count of valid detection events.
        false_negatives: count of missed events.
        false_positives: count of incorrect detections.
        prc: precision of detections.
        npv: negative predicted value.
        acc: accuracy of detections.
        rcl: recall of detections.
        f1: statistic that balances true positives and false negatives.
        f2: statistic that gives more weight to true positives.
    """
    metrics = MetricsContainer()
    tp_events = 0
    fp_events = 0
    fn_events = 0
    prev_cm = 'tn'
    for i in range(0, len(df['conf_mtx'])):  # for every row of data

        # if the confusion matrix class has changed
        if (df['conf_mtx'][i] != prev_cm):

            if (df['conf_mtx'][i] == 'tp'):  # true positive case
                tp_events += 1
            elif (df['conf_mtx'][i] == 'fp'):  # false positive case
                fp_events += 1
            elif (df['conf_mtx'][i] == 'fn'):  # false negative case
                fn_events += 1
            prev_cm = df['conf_mtx'][i]

    # calculate metrics
    metrics.true_positives = tp_events
    metrics.false_positives = fp_events
    metrics.false_negatives = fn_events
    metrics.prc = metrics.ppv = tp_events / (tp_events + fp_events)
    metrics.rcl = tp_events / (tp_events + fn_events)
    metrics.f1 = 2.0 * (metrics.prc * metrics.rcl) / (metrics.prc + metrics.rcl)
    metrics.f2 = 5.0 * tp_events / \
                 (5.0 * tp_events + 4.0 * fn_events + fp_events)
    return metrics


def print_metrics(metrics):
    print('PPV = %f' % metrics.prc)
    if hasattr(metrics, 'npv'):
        print('NPV = %f' % metrics.npv)
    if hasattr(metrics, 'acc'):
        print('Acc = %f' % metrics.acc)
    print('TP  = %i' % metrics.true_positives)
    if hasattr(metrics, 'true_negatives'):
        print('TN  = %i' % metrics.true_negatives)
    print('FP  = %i' % metrics.false_positives)
    print('FN  = %i' % metrics.false_negatives)
    print('F1 = %f' % metrics.f1)
    print('F2 = %f' % metrics.f2)


def group_bools(df, column_in, column_out):
    """
    group_bools indexes each grouping of anomalies (1) and valid points (0) as numbered sets.
    Used for anomaly correction.
    Arguments:
        df: data frame with required columns:
            'detected_event': boolean array of classified data points
    Returns:
        df: original data frame with additional column:
            'group' containing an index of boolean groupings
    """
    # initialize the 'group' column to zeros
    df[column_out] = 0
    # initialize placeholder for boolean state of previous group
    last = df.iloc[0][column_in]
    # initialize the group index to zero
    gi = 0

    # loop over every row in dataframe
    for i in range(0, len(df[column_out])):

        # if the anomaly bool has changed
        if last != df.iloc[i][column_in]:
            gi += 1  # increment the group index
        # assign this row to the group index
        df.iloc[i, df.columns.get_loc(column_out)] = gi

        # update last boolean state
        last = df.iloc[i][column_in]

    return df


def xfade(xfor, xbac):
    """
    xfade ("cross-fade") blends two data sets of matching length with a ramp function (weighted average).
    Arguments:
        xfor: forecasted data to be more weighted at the front
        xbac: backcasted data to be more weighted at the back
    Returns:
        x: the blended data
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
            fader.append((i + 1) / (len(xfor) + 1))
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
    considering the model residuals looking forward and backward a specified number of steps.
    Arguments:
        residuals: series like object or a data frame of model residuals.
        alpha: scalar between 0 and 1 representing the acceptable uncertainty.
        window_sz: integer representing how many data points to use in both directions.
            default = 96 for one day for 15-minute data.
    Returns:
        threshold: data frame of columns of low and high threshold values.
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
        mean = residuals[lo:(hi + 1)].mean()
        sigma = residuals[lo:(hi + 1)].std()
        th_range = z * sigma
        if (th_range < min_range):
            th_range = min_range
        # append pair of upper and lower thresholds
        threshold.append([mean - th_range, mean + th_range])

    threshold = pd.DataFrame(threshold, columns=['low', 'high'])

    return threshold


def set_cons_threshold(model_fit, alpha_in):
    """
    set_cons_threshold determines a threshold based on confidence interval and specified alpha for an ARIMA model.
    Arguments:
        model_fit: SARIMAX model object.
        alpha_in: scalar between 0 and 1 representing the acceptable uncertainty.
    Returns:
        threshold: single value.
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


def detect_anomalies(observed, predictions, residuals, threshold, summary=True):
    """
    detect_anomalies compares model residuals to thresholds to determine which points are anomalous.
    Arguments:
        observed: data frame or series of observed data.
        predictions: series of model predictions.
        residuals: series of model residuals.
        threshold: data frame with the columns 'lower' and 'upper' corresponding to the acceptable range of the residual.
        summary: if True, will print the ratio of detections.
    Returns:
        detections: data frame with columns for observations, predictions, residuals, anomalies
        (boolean where True (1) = anomalous data point)
    """
    detections = pd.DataFrame(observed)
    detections['prediction'] = np.array(predictions)
    detections['residual'] = np.array(residuals)
    detections['anomaly'] = (detections['residual'] < threshold['low']) | (threshold['high'] < detections['residual'])
    # anomalies = test_score_df[test_score_df.anomaly == True]

    # output summary
    if summary:
        print('ratio of detections: %f' % ((sum(detections.anomaly) / len(detections.anomaly)) * 100), '%')

    return detections


def detect_anomalies_cons(residuals, threshold, summary=True):
    """
    Compares residuals to a constant threshold to identify anomalies. Can use set threshold level or threshold
    determined by set_cons_threshold function.
    Arguments:
        residuals: series of model residuals.
        threshold: constant threshold value.
        summary: if True, will print the ratio of detections.
    Returns:
        detected_anomaly: boolean series where True (1) = anomalous data point
    """
    # DETERMINE ANOMALIES
    detected_anomaly = (residuals[0] < threshold['low']) | (threshold['high'] < residuals[0])  # gives bools
    # output summary
    if summary:
        print('ratio of detections: %f' % ((sum(detected_anomaly) / len(detected_anomaly)) * 100), '%')

    return detected_anomaly


def aggregate_results(df, models, verbose=False, compare=False):
    """
    aggregate_results combines the assessments of detections from multiple models to give a single output of anomalies.
    If any model detects an anomaly, the point is labeled as anomalous.
    Arguments:
        df: data frame with required column 'observed' of observed data values.
        models: dictionary of model outputs consisting of dataframes with the required column 'detected_event' of booleans indicating anomalies.
        verbose: if True, includes columns for each model type in the output.
        compare: if True, includes columns for technician labeled anomalies and labeled events in the output (as gathered from the input df) for determination of metrics.
    Returns:
        results_all: data frame containing columns 'detected_event' of booleans representing anomalies aggregated
        from all of the models and 'observed' of observed values.
            Additional columns are added if verbose and compare options are selected.
        metrics_all: if compare is selected, then metrics are output for the aggregate anomalies.
    """
    results_all = pd.DataFrame(index=df.index)
    for model in models:
        results_all[model] = models[model]['detected_event'] > 0
    results_all['detected_event'] = results_all.any(axis=1)
    results_all['observed'] = df['observed']

    if not verbose:
        for model in models:
            results_all = results_all.drop(model, 1)

    if compare:
        results_all['labeled_anomaly'] = df['labeled_anomaly']
        results_all['labeled_event'] = anomaly_events(results_all['labeled_anomaly'], wf=1)
        compare_events(results_all, wf=1)
        metrics_all = metrics(results_all)
        return results_all, metrics_all
    else:
        return results_all


def plt_threshold(residuals, threshold, sensor):
    plt.plot(residuals, 'b', label='residuals')
    plt.plot(threshold['low'], 'c', label='thresh_low')
    plt.plot(threshold['high'], 'm', mfc='none', label='thresh_high')
    plt.legend()
    plt.ylabel(sensor)


def plt_results(raw, predictions, labels, detections, sensor):
    plt.plot(raw, 'b', label='original data')
    plt.plot(predictions, 'c', label='predicted values')
    plt.plot(raw[labels > 0], 'mo', mfc='none', label='technician labeled anomalies')
    plt.plot(predictions[detections > 0], 'r+', label='machine detected anomalies')
    plt.legend()
    plt.ylabel(sensor)

