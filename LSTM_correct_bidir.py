################################
# LSTM CORRECT #
################################
# This script includes functionality for making corrections using LSTM regression. There are two separate functions
# for  univariate LSTM and multivariate LSTM.
# Note that the LSTM model needs to be trained on relatively clean data. If drift corrections have been applied to
#   training data, drift corrections need to have been applied to the data to the model inputs else unexpected
#   predictions will result.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def LSTM_correct(df, model, time_steps, scaler):
    """Makes corrections for detected events.
    'df': data frame with columns 'raw_scaled' corresponding to raw data with a scaler applied and 'detected_event'
    corresponding to a list of numbered anomalous events.
    'model': LSTM model developed and used to detect anomalies in this dataset.
    'time_steps': number of previous time steps being considered in the model.
    'scaler': standard scaler object for re-scaling the data back to the original units.
    Outputs: df with additional columns: 'det_cor_scaled' and 'det_cor' corresponding to the determined correction."""

    # Add column and populate the det_cor_scaled data
    df['det_cor_scaled'] = df['raw_scaled']

    # Loop through each time step with a detected event. Loop starts based on the time steps to account for
    # the necessary inputs to the model.
    for i in range(time_steps, len(df['raw_scaled'] - time_steps)):

        if (df.iloc[i]['detected_event'] != 0):              # enter loop for observations associated with a detected event
            v = df['det_cor_scaled'].iloc[(i-time_steps):i]  # reshape data to include past data to input to model
            w = np.resize(v, (1, v.shape[0], 1))             # get in from (observations, time_steps, features)
            y = model.predict(w)                             # apply the model to each step
            pred = pd.DataFrame(y[:, 0])[0][0]               # get prediction for that step
            df['det_cor_scaled'].iloc[i] = pred              # replace the raw with

    # Add column to unscale/transform back to original units
    df['det_cor'] = scaler.inverse_transform(df['det_cor_scaled'])

    return df

# Apply function to df_anomalies from LSTM_detect script.
df_anomalies = LSTM_correct(df_anomalies, model, time_steps, scaler)

############ PLOTTING ##############

plt.figure()
plt.plot(df_anomalies['raw'], 'b', label='original data')
plt.plot(df_anomalies['cor'], 'c', label='technician corrected')
plt.plot(df_anomalies['raw'][df_anomalies['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
plt.plot(df_anomalies['raw'][df_anomalies['detected_anomaly']], 'r+', mfc='none', label='machine detected anomalies')
plt.plot(df_anomalies['det_cor'], 'm', label='determined_corrected')
plt.legend()
plt.ylabel(sensor[0])
plt.show()

################ MULTI ######################


def LSTM_multi_correct(df_array, df_raw_scaled, time_steps, model, scaler):
    """Makes corrections for detected events on multi variate data. At this point, it is assumed that there are 4 variables.
        0: temp, 1: cond, 2: ph, 3: do
        # TODO: make list of sensors variable
    'df_array': array containing data frames for each variable with column 'detected_event'
        corresponding to a list of numbered anomalous events.
    'df_raw_scaled': data frame containing columns each corresponding to a variable with the raw data scaled.
    'model': LSTM model developed and used to detect anomalies in this dataset. Gives multivariate output
    'time_steps': number of previous time steps being considered in the model.
    'scaler': standard scaler object for re-scaling the data back to the original units.
    Outputs: df_array of data frames each with additional columns: 'det_cor_scaled' and 'det_cor' corresponding to the determined correction."""

    df_array[0]['det_cor_scaled'] = df_raw_scaled['temp']
    df_array[1]['det_cor_scaled'] = df_raw_scaled['cond']
    df_array[2]['det_cor_scaled'] = df_raw_scaled['ph']
    df_array[3]['det_cor_scaled'] = df_raw_scaled['do']

    for i in range(time_steps, len(df_array[0])):
        if (df_array[0].iloc[i]['detected_event'] != 0
            or df_array[1].iloc[i]['detected_event'] != 0
            or df_array[2].iloc[i]['detected_event'] != 0
            or df_array[3].iloc[i]['detected_event'] != 0):  # enter loop for observations associated with a detected event
                v = df_raw_scaled.iloc[(i - time_steps):i]  # reshape data to include past data to input to model
                w = np.resize(v, (1, v.shape[0], len(df_array)))  # get in form (observations, time_steps, features)
                y = model.predict(w)  # apply the model to each step
                for j in range(0, y.shape[1]):
                    if (df_array[j].iloc[i]['detected_event'] != 0):
                        pred = y[:, j][0]  # get prediction for that step
                        df_array[j]['det_cor_scaled'].iloc[i] = pred  # replace the raw with

    det_cor_scaled = pd.DataFrame(index=df_array[0].index)
    det_cor_scaled['temp_cor_scaled'] = df_array[0]['det_cor_scaled']
    det_cor_scaled['cond_cor_scaled'] = df_array[1]['det_cor_scaled']
    det_cor_scaled['ph_cor_scaled'] = df_array[2]['det_cor_scaled']
    det_cor_scaled['do_cor_scaled'] = df_array[3]['det_cor_scaled']
    det_cor = pd.DataFrame(scaler.inverse_transform(det_cor_scaled), columns=df_raw_scaled.columns, index=det_cor_scaled.index)

    return det_cor, det_cor_scaled

# To test with smaller arrays
df_array[0] = df_array[0].loc['2014-01-01 00:00':'2014-01-15 00:00']
df_array[1] = df_array[1].loc['2014-01-01 00:00':'2014-01-15 00:00']
df_array[2] = df_array[2].loc['2014-01-01 00:00':'2014-01-15 00:00']
df_array[3] = df_array[3].loc['2014-01-01 00:00':'2014-01-15 00:00']

# Apply function to df_anomalies from LSTM_detect script.
det_cor, det_cor_scaled = LSTM_multi_correct(df_array, df_raw_scaled, time_steps, model, scaler)

############ PLOTTING ##############

for i in range(0, len(sensor)):
    plt.figure()
    plt.plot(df_raw[df_raw.columns[i]], 'b', label='original data')
    plt.plot(df_cor[df_cor.columns[i]], 'c', label='technician corrected data' )
    plt.plot(test_score_array[i]['pred_unscaled'], 'c', label='predicted values')
    plt.plot(sensor_array[sensor[i]]['raw'][sensor_array[sensor[i]]['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
    plt.plot(test_score_array[i]['pred_unscaled'][test_score_array[i]['anomaly']], 'r+', label='machine detected anomalies')
    plt.plot(det_cor[det_cor.columns[i]], 'm', label='determined_corrected')
    plt.legend()
    plt.ylabel(sensor[i])
    plt.show()
