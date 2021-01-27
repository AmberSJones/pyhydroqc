################################
# ARIMA CORRECT #
################################
# This script includes functionality for making corrections using ARIMA regression.

import numpy as np
import pandas as pd
import anomaly_utilities
import pmdarima as pm


def ARIMA_group(df, min_group_len=20):
    """Examines detected events and performs conditional widening to ensure
    that widened event is sufficient for forecasting/backcasting.
    df is a data frame with the required columns: 'group' and 'detected_event'
    and returns with new columns: 'ARIMA_event' and 'ARIMA_group'"""
    ARIMA_group = []
    df['ARIMA_event'] = df['detected_event']
    new_gi = 0
    merging = False
    # for each group
    for i in range(0, (max(df['group']) + 1)):
        # determine the length of this group
        group_len = len(df.loc[df['group'] == i]['group'])
        # if this group is not an anomaly event and is too small to support an ARIMA model
        if ((df.loc[df['group'] == i]['detected_event'][0] == 0) and
            (group_len < min_group_len)):
            # this group needs to be added to previous group
            df.loc[df['group'] == i, 'ARIMA_event'] = 1
            if (new_gi > 0):
                new_gi -= 1
            ARIMA_group.extend(np.full([1, group_len], new_gi, dtype=int)[0])
            merging = True

        else:  # this group does not need to be altered

            ARIMA_group.extend(np.full([1, group_len], new_gi, dtype=int)[0])

            # if not merging last group to current group
            if ~merging:
                merging = False
                new_gi += 1

    if (new_gi < (max(df['group'])/2)):
        print("WARNING: more than half of the anomaly events have been merged!")
    df['ARIMA_group'] = ARIMA_group
    return df


def ARIMA_forecast(x, l):
    """ ARIMA_forecast is used to create predictions of data where anomalies occur.
    Creates ARIMA model and outputs forecasts of specified length.
    x is an array of values from which to predict corrections. corresponds to non-anomalous data.
    l is how many predicted data points are to be forecasted/corrected.
    Outputs:
    y is an array of length l of the corrected values as predicted by the model
    """
    model = pm.auto_arima(x)
    y = model.predict(l)
    return y


def generate_corrections(df):
    """generate_corrections uses passes through data with identified anomalies and determines corrections
    using an ARIMA model. Corrections are determined by combining both a forecast and a backcast in a weighted
    average to be informed by non-anamolous data before and after anomalies.
    df is a data frame with required columns:
    'raw': raw data
    'detected_event': boolean array corresponding to classified anomalies where True = anomalous
    Outputs:
    df with additional column: 'det_cor' determined correction.
    """

    # assign group index numbers to each set of consecutiveTrue/False data points
    df = anomaly_utilities.group_bools(df)
    df = ARIMA_group(df,5)

    # initialize new column of corrected data
    df['det_cor'] = df['raw']

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
            yfor = ARIMA_forecast(np.array(df.loc[df['ARIMA_group'] == (i - 1)]['raw']),
                                       len(df.loc[df['ARIMA_group'] == i]))
            forecasted = True
        # perform backcasting to generate corrected data points
        if (i != max(df['ARIMA_group'])): # if not at the end
            # forecast in reverse direction
            # data associated with group i+1 gets flipped for making a forecast
            yrev = ARIMA_forecast(np.flip(np.array(df.loc[df['ARIMA_group'] == (i + 1)]['raw'])),
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

            # decrement the following ARIMA_groups
            df.loc[df['ARIMA_group'] > i, 'ARIMA_group'] -= 1

        elif (not backcasted):  # if there is no backcast
            # add the correction to the detected event
            df.loc[df['ARIMA_group'] == i, 'det_cor'] = yfor

            # remove the ARIMA_event
            df.loc[df['ARIMA_group'] == i, 'ARIMA_event'] = 0

            # merge the last ARIMA_group
            df.loc[df['ARIMA_group'] == i, 'ARIMA_group'] = i - 1

        else:  # both a forecast and a backcast exist
            # add the correction to the detected event
            df.loc[df['ARIMA_group'] == i, 'det_cor'] = anomaly_utilities.xfade(yfor, ybac)

            # remove the ARIMA_event
            df.loc[df['ARIMA_group'] == i, 'ARIMA_event'] = 0

            # merge the ARIMA_groups
            df.loc[df['ARIMA_group'] == i, 'ARIMA_group'] = i - 1
            df.loc[df['ARIMA_group'] == i + 1, 'ARIMA_group'] = i - 1

            # decrement the following ARIMA_groups
            df.loc[df['ARIMA_group'] > i, 'ARIMA_group'] -= 2

    # delete unused columns
    df = df.drop('group', 1)
    df = df.drop('ARIMA_event', 1)
    df = df.drop('ARIMA_group', 1)
    return df


df = generate_corrections(df)


############ PLOTTING ##############

import matplotlib.pyplot as plt
plt.figure()
plt.plot(df['raw'], 'b', label='original data')
plt.plot(df['cor'], 'c', label='technician corrected')
plt.plot(df['raw'][df['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
plt.plot(df['raw'][df['detected_anomaly']], 'r+', mfc='none', label='machine detected anomalies')
plt.plot(df['det_cor'], 'm', label='determined_corrected')
plt.legend()
plt.ylabel(sensor[0])
plt.show()

