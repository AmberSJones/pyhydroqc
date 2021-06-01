################################
# ARIMA CORRECT #
################################
# This script includes functionality for making corrections using ARIMA regression.

import numpy as np
from pyhydroqc import anomaly_utilities
import pmdarima as pm
import warnings
import pandas as pd


def arima_group(df, anomalies, group, min_group_len=20):
    """
    arima_group examines detected events and performs conditional widening (marks some valid points as anomalous) to ensure that widened event is sufficient for forecasting/backcasting.
    Arguments:
        df: data frame with a column for detected events.
        anomalies: string of the column name for detected events.
        group: string of column name containing an ascending index for each group of valid or anomolous data points (output of the group_bools function).
        min_group_len: the minimum group length.
    Returns:
        df: data frame with new columns: 'ARIMA_event' and 'arima_group'
    """
    arima_group = []
    df['ARIMA_event'] = df[anomalies]
    new_gi = 0
    # for each group
    for i in range(0, (max(df[group]) + 1)):
        # determine the length of this group
        group_len = len(df.loc[df[group] == i][group])
        # if this group is not an anomaly event and is too small to support an ARIMA model
        if ((df.loc[df[group] == i][anomalies][0] == 0) and (group_len < min_group_len)):
            # this group needs to be added to previous group
            df.loc[df[group] == i, 'ARIMA_event'] = True
            if (new_gi > 0):
                new_gi -= 1
            arima_group.extend(np.full([1, group_len], new_gi, dtype=int)[0])

        else:  # this group does not need to be altered
            arima_group.extend(np.full([1, group_len], new_gi, dtype=int)[0])
            new_gi += 1

    if (new_gi < (max(df[group])/2)):
        print("WARNING: more than half of the anomaly events have been merged!")
    df['arima_group'] = arima_group

    return df


def arima_forecast(x, l, suppress_warnings=True):
    """
    arima_forecast creates predictions of data where anomalies occur. Creates ARIMA model and outputs forecasts of specified length.
    Arguments:
        x: array of values from which to predict corrections. corresponds to non-anomalous data.
        l: number of predicted data points to be forecasted/corrected.
        suppress_warnings: indicates whether warnings associated with ARIMA model development and fitting should be suppressed.
    Returns:
        y: array of length l of the corrected values as predicted by the model
    """
    model = pm.auto_arima(x, error_action='ignore', suppress_warnings=True)
    if suppress_warnings:
        warnings.filterwarnings('ignore', message='Non-stationary starting autoregressive parameters')
        warnings.filterwarnings('ignore', message='Non-invertible starting MA parameters found.')
        warnings.filterwarnings('ignore', message='ConvergenceWarning: Maximum Likelihood optimization failed to converge.')
    y = model.predict(l)
    return y


def generate_corrections(df, observed, anomalies, model_limit=6, savecasts=False, suppress_warnings=True):
    """
    generate_corrections passes through data with identified anomalies and determines corrections using ARIMA models.
    Corrections are determined by combining both a forecast and a backcast in a weighted average that is informed by
    non-anamolous data before and after anomalies. Corrections are generated for anomalies by order of the shortest to
    longest and those corrected values from the shorter anomalies are used with non-anomalous values to generate
    corrections for longer anomalies.
    Arguments:
        df: data frame with columns for observations and anomalies as defined by the user.
        observed: string that names the column in the data frame containing observed values.
        anomalies: string that names the column in the data frame containing booleans corresponding to anomalies where True = anomalous.
        model_limit: int for number of days used to limit the amount of data from which to generate forecasts and backcasts
        savecasts: boolean used for saving the forecast as backcast data which can be used for analysis or plotting.
        suppress_warnings: indicates whether warnings associated with ARIMA model development and fitting should be suppressed.
    Returns:
        df with additional columns:
            'det_cor' - determined correction
            'corrected' - boolean indicating whether the data was corrected
            'forecasts' - forecasted values used in correction (only created if savecasts=True)
            'backcasts' - backcasted values used in correction (only created if savecasts=True)
    """

    # assign group index numbers to each set of consecutiveTrue/False data points
    df = anomaly_utilities.group_bools(df, column_in=anomalies, column_out='group')
    df = arima_group(df, anomalies, 'group')

    # create new output columns
    df['det_cor'] = df[observed]
    df['corrected'] = df['ARIMA_event']
    if (savecasts):
        df['forecasts'] = np.nan
        df['backcasts'] = np.nan

    # while there are anomalous groups of points left to correct
    while len(df[df['ARIMA_event'] != 0]) > 0:
        # find an index for an anomalous arima_group having the smallest number of points
        i = df[df['ARIMA_event'] != 0]['arima_group'].value_counts().index.values[-1]

        # reset the conditionals
        forecasted = False
        backcasted = False
        # perform forecasting to generate corrected data points
        if (i != 0):  # if not at the beginning
            # forecast in forward direction
            # create an array of corrected data for current anomalous group
            # i-1 is the index of the previous group being used to forecast
            pre_data = df.loc[df['arima_group'] == (i - 1)][observed]  # save off data for modeling
            pre_data = pre_data[pre_data.index[-1] - pd.Timedelta(days=model_limit):pre_data.index[-1]]  # limit data
            # generate the forecast data
            yfor = arima_forecast(np.array(pre_data),
                                  len(df.loc[df['arima_group'] == i]),
                                  suppress_warnings)

            forecasted = True
            if (savecasts):
                df.loc[df['arima_group'] == i, 'forecasts'] = yfor
                df.loc[df[df['arima_group'] == i - 1].index[-1], 'forecasts'] = \
                    df.loc[df[df['arima_group'] == i - 1].index[-1], 'observed']

        # perform backcasting to generate corrected data points
        if (i != max(df['arima_group'])): # if not at the end
            # forecast in reverse direction
            # data associated with group i+1 gets flipped for making a forecast
            post_data = df.loc[df['arima_group'] == (i + 1)][observed]  # save off data for modeling
            post_data = post_data[post_data.index[0]:post_data.index[0] + pd.Timedelta(days=model_limit)]  # limit data
            # create backcast
            yrev = arima_forecast(np.flip(np.array(post_data)),
                                  len(df.loc[df['arima_group'] == i]),
                                  suppress_warnings)
            # output is reversed, making what was forecast into a backcast
            ybac = np.flip(yrev)
            backcasted = True
            if (savecasts):
                df.loc[df['arima_group'] == i, 'backcasts'] = ybac
                df.loc[df[df['arima_group'] == i + 1].index[0], 'backcasts'] = \
                    df.loc[df[df['arima_group'] == i + 1].index[0], 'observed']

        # fill the det_cor column using forecasted and backcasted conditionals
        if ((not forecasted) and (not backcasted)):
            print("ERROR: all data points are anomalous!")
        elif (not forecasted):  # if there is no forecast
            # add the correction to the detected event
            df.loc[df['arima_group'] == i, 'det_cor'] = ybac

            # remove the ARIMA_event
            df.loc[df['arima_group'] == i, 'ARIMA_event'] = 0

            # decrement the following ARIMA_groups (to merge 0 and 1)
            df.loc[df['arima_group'] > i, 'arima_group'] -= 1

        elif (not backcasted):  # if there is no backcast
            # add the correction to the detected event
            df.loc[df['arima_group'] == i, 'det_cor'] = yfor

            # remove the ARIMA_event
            df.loc[df['arima_group'] == i, 'ARIMA_event'] = 0

            # merge the last arima_group after correction
            df.loc[df['arima_group'] == i, 'arima_group'] = i - 1

        else:  # both a forecast and a backcast exist
            # add the correction to the detected event
            df.loc[df['arima_group'] == i, 'det_cor'] = anomaly_utilities.xfade(yfor, ybac)

            # remove the ARIMA_event
            df.loc[df['arima_group'] == i, 'ARIMA_event'] = 0

            # merge the ARIMA_groups after correction
            df.loc[df['arima_group'] == i, 'arima_group'] = i - 1
            df.loc[df['arima_group'] == i + 1, 'arima_group'] = i - 1

            # decrement the following ARIMA_groups
            df.loc[df['arima_group'] > i, 'arima_group'] -= 2

    # delete unused columns
    df = df.drop('group', 1)
    df = df.drop('ARIMA_event', 1)
    df = df.drop('arima_group', 1)

    return df
