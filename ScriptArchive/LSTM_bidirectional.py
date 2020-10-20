################################
# LSTM BIDIRECTIONAL #
################################
# This code takes raw data and corrected data, applies an LSTM model, and identifies anomalies.

print("LSTM exploration script begin.")

import rules_detect
import anomaly_utilities
import modeling_utilities
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
import plotly.io as pio
from random import randint

pio.renderers.default = "browser"
pd.options.mode.chained_assignment = None

sns.set(style='whitegrid', palette='muted')
rcParams['figure.figsize'] = 14, 8
np.random.seed(1)
# tf.random.set_seed(1)

print('Tensorflow version:', tf.__version__)


# DEFINE SITE and VARIABLE #
#########################################
# site = "BlackSmithFork"
# site = "FranklinBasin"
# site = "MainStreet"
site = "Mendon"
# site = "TonyGrove"
# site = "WaterLab"
# sensor = "temp"
sensor = ['cond']
# sensor = "ph"
# sensor = "do"
# sensor = "turb"
# sensor = "stage"
year = 2017

# EXECUTE FUNCTIONS #
#########################################
# Get data
df_full, sensor_array = anomaly_utilities.get_data(site, sensor, year, path="./LRO_data/")
df = sensor_array[sensor[0]]

# Valid data must be used to train the detector. Options include:
#   - Use corrected data to train the detector. This is problematic due to pervasive drift corrections throughout.
#       Also problematic because of -9999 values in the data.
#   - Use a subset of raw data that is in decent shape without NaNs, -9999 values, or data gaps.
#   - Use raw data that have been preprocessed to filter out extreme values and have drift correction applied.
#   - Either with raw or corrected data for training, use data that are not labeled as anomalous/corrected.
#   df_cor = df_cor.replace(-9999, np.NaN)


# Rules based Detection
# General sensor ranges for LRO data:
# Temp min: -5, max: 30
# SpCond min: 100, max: 900
# pH min: 7.5, max: 9.0
# do min: 2, max: 16

maximum = 900
minimum = 150
df = rules_detect.range_check(df, maximum, minimum)
length = 6
df = rules_detect.persistence(df, length)
size = rules_detect.group_size(df)
df = rules_detect.interpolate(df)

# Model creation
# scales data, reshapes data, builds and trains model, evaluates model results
time_steps = 10
samples = 5000
cells = 128
dropout = 0.2
patience = 6

X_train, y_train, model, history, X_test, y_test, model_eval, predictions, train_residuals, test_residuals = modeling_utilities.bidir_LSTM_model(df, time_steps, samples, cells, dropout, patience)

# Plot Metrics and Evaluate the Model
# plot training loss and validation loss with matplotlib and pyplot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# look at the distribution of the errors using a distribution plot
# could find a way to do a 95% percentile or some other actual value to automatically select the threshold.
# However, that is a set number of anomalies.
sns.distplot(train_residuals, bins=50, kde=True)
plt.show()
# choose a threshold to use for anomalies based on x-axis. try where error is greater than 0.75, it's anomalous.
threshold = [18]
sns.distplot(test_residuals, bins=50, kde=True)
plt.show()

# Detect anomalies
test_data = df[['det_cor']]
test_score_array = modeling_utilities.detect_anomalies_bidir(test_data, predictions, time_steps, test_residuals, threshold)

# Use events function to widen and number anomalous events
df_anomalies = df.iloc[time_steps:]
df_anomalies['labeled_event'] = anomaly_utilities.anomaly_events(df_anomalies['labeled_anomaly'])
df_anomalies['detected_anomaly'] = test_score_array[0]['anomaly']
df_anomalies['detected_event'] = anomaly_utilities.anomaly_events(df_anomalies['detected_anomaly'])

# Determine Metrics
anomaly_utilities.compare_events(df_anomalies, 0)
metrics = anomaly_utilities.metrics(df_anomalies)

# OUTPUT RESULTS #
#########################################
print('\n\n\nScript report:\n')
print('Sensor: ' + sensor[0])
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
anomaly_utilities.print_metrics(metrics)
print("\n LSTM script end.")


# GENERATE PLOTS #
#########################################

plt.figure()
plt.plot(df['raw'], 'b', label='original data')
plt.plot(test_score_array[0]['prediction'], 'c', label='predicted values')
plt.plot(df['raw'][df['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
plt.plot(test_score_array[0]['prediction'][test_score_array[0]['anomaly']], 'r+', label='machine detected anomalies')
plt.legend()
plt.ylabel(sensor)
plt.show()


print("\n LSTM script end.")
