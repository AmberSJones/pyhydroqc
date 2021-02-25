################################
# ARIMA CORRECT #
################################
# This script includes functionality for making corrections using ARIMA regression.

import numpy as np
from PyHydroQC import anomaly_utilities
import pmdarima as pm
import warnings


def ARIMA_group(df, anomalies, group, min_group_len=20):
    """
    ARIMA_group examines detected events and performs conditional widening (marks some valid points as anomalous) to ensure that widened event is sufficient for forecasting/backcasting.
    Arguments:
        df: data frame with a column for detected events.
        anomalies: string of the column name for detected events.
        group: string of column name containing an ascending index for each group of valid or anomolous data points (output of the group_bools function).
        min_group_len: the minimum group length.
    Returns:
        df: data frame with new columns: 'ARIMA_event' and 'ARIMA_group'
    """
    ARIMA_group = []
    df['ARIMA_event'] = df[anomalies]
    new_gi = 0
    merging = False
    # for each group
    for i in range(0, (max(df[group]) + 1)):
        # determine the length of this group
        group_len = len(df.loc[df[group] == i][group])
        # if this group is not an anomaly event and is too small to support an ARIMA model
        if ((df.loc[df[group] == i][anomalies][0] == 0) and
            (group_len < min_group_len)):
            # this group needs to be added to previous group
            df.loc[df[group] == i, 'ARIMA_event'] = 1
            if (new_gi > 0):
                new_gi -= 1
            ARIMA_group.extend(np.full([1, group_len], new_gi, dtype=int)[0])
            merging = True

        else:  # this group does not need to be altered

            ARIMA_group.extend(np.full([1, group_len], new_gi, dtype=int)[0])

            # if not merging last group to current group
            if not merging:
                merging = False
                new_gi += 1

    if (new_gi < (max(df[group])/2)):
        print("WARNING: more than half of the anomaly events have been merged!")
    df['ARIMA_group'] = ARIMA_group

    return df


def ARIMA_forecast(x, l):
    """
    ARIMA_forecast creates predictions of data where anomalies occur. Creates ARIMA model and outputs forecasts of specified length.
    Arguments:
        x: array of values from which to predict corrections. corresponds to non-anomalous data.
        l: number of predicted data points to be forecasted/corrected.
    Returns:
        y: array of length l of the corrected values as predicted by the model
    """
    model = pm.auto_arima(x, error_action='ignore', suppress_warnings=True)
    warnings.filterwarnings('ignore', message='Non-stationary starting autoregressive parameters')
    warnings.filterwarnings('ignore', message='Non-invertible starting MA parameters found.')
    warnings.filterwarnings('ignore', message='ConvergenceWarning: Maximum Likelihood optimization failed to converge.')
    y = model.predict(l)
    return y


def generate_corrections(df, observed, anomalies):
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
    Returns:
        df with additional columns:
            'det_cor' - determined correction
            'corrected' - boolean indicating whether the data was corrected
    """

    # assign group index numbers to each set of consecutiveTrue/False data points
    df = anomaly_utilities.group_bools(df, column_in=anomalies, column_out='group')
    df = ARIMA_group(df, anomalies, 'group')

    # create new output columns
    df['det_cor'] = df[observed]
    df['corrected'] = df['ARIMA_event']

    # while there are anomalous groups of points left to correct
    while len(df[df['ARIMA_event'] != 0]) > 0:
        # find an index for an anomalous ARIMA_group having the smallest number of points
        i = df[df['ARIMA_event'] != 0]['ARIMA_group'].value_counts().index.values[-1]

        # # if this is an anomalous group of points (event index not equal to 0)
        # if(df.loc[df['ARIMA_group'] == i]['detected_event'][0] != 0):
        # reset the conditionals
        forecasted = False
        backcasted = False
        # perform forecasting to generate corrected data points
        if (i != 0):  # if not at the beginning
            # forecast in forward direction
            # create an array of corrected data for current anomalous group
            # i-1 is the index of the previous group being used to forecast
            yfor = ARIMA_forecast(np.array(df.loc[df['ARIMA_group'] == (i - 1)][observed]),
                                       len(df.loc[df['ARIMA_group'] == i]))
            forecasted = True
        # perform backcasting to generate corrected data points
        if (i != max(df['ARIMA_group'])): # if not at the end
            # forecast in reverse direction
            # data associated with group i+1 gets flipped for making a forecast
            yrev = ARIMA_forecast(np.flip(np.array(df.loc[df['ARIMA_group'] == (i + 1)][observed])),
                                       len(df.loc[df['ARIMA_group'] == i]))
            # output is reversed, making what was forecast into a backcast
            ybac = np.flip(yrev)
            backcasted = True

        # fill the det_cor column using forecasted and backcasted conditionals
        if ((not forecasted) and (not backcasted)):
            print("ERROR: all data points are anomalous!")
        elif (not forecasted):  # if there is no forecast
            # add the correction to the detected event
            df.loc[df['ARIMA_group'] == i, 'det_cor'] = ybac

            # remove the ARIMA_event
            df.loc[df['ARIMA_group'] == i, 'ARIMA_event'] = 0

            # decrement the following ARIMA_groups (to merge 0 and 1)
            df.loc[df['ARIMA_group'] > i, 'ARIMA_group'] -= 1

        elif (not backcasted):  # if there is no backcast
            # add the correction to the detected event
            df.loc[df['ARIMA_group'] == i, 'det_cor'] = yfor

            # remove the ARIMA_event
            df.loc[df['ARIMA_group'] == i, 'ARIMA_event'] = 0

            # merge the last ARIMA_group after correction
            df.loc[df['ARIMA_group'] == i, 'ARIMA_group'] = i - 1

        else:  # both a forecast and a backcast exist
            # add the correction to the detected event
            df.loc[df['ARIMA_group'] == i, 'det_cor'] = anomaly_utilities.xfade(yfor, ybac)

            # remove the ARIMA_event
            df.loc[df['ARIMA_group'] == i, 'ARIMA_event'] = 0

            # merge the ARIMA_groups after correction
            df.loc[df['ARIMA_group'] == i, 'ARIMA_group'] = i - 1
            df.loc[df['ARIMA_group'] == i + 1, 'ARIMA_group'] = i - 1

            # decrement the following ARIMA_groups
            df.loc[df['ARIMA_group'] > i, 'ARIMA_group'] -= 2

    # delete unused columns
    df = df.drop('group', 1)
    df = df.drop('ARIMA_event', 1)
    df = df.drop('ARIMA_group', 1)

    return df
