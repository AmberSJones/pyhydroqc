################################
# LSTM CORRECT #
################################
# This script includes functionality for making corrections using LSTM regression.
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
    for i in range(time_steps, len(df['raw_scaled'])):

        if (df.iloc[i]['detected_event'] != 0):              # enter loop for observations associated with a detected event
            v = df['det_cor_scaled'].iloc[(i-time_steps):i]  # reshape data to include past data to input to model
            w = np.resize(v, (1, v.shape[0], 1))             # get in from (observations, time_steps, features)
            y = model.predict(w)                             # apply the model to each step
            pred = pd.DataFrame(y[:, 0])[0][0]               # get prediction for that step
            df['det_cor_scaled'].iloc[i] = pred              # replace the raw with

    # Add column to unscale/transform back to original units
    df['det_cor'] = scaler.inverse_transform(df['det_cor_scaled'])

# Apply function to df_anomalies from LSTM_detect script
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