################################
# LSTM DEVELOP AND DIAGNOSTIC #
################################
# This code takes raw data and corrected data, applies an LSTM model, and identifies anomalies.

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
pio.renderers.default = "browser"
pd.options.mode.chained_assignment = None
sns.set(style='whitegrid', palette='muted')
rcParams['figure.figsize'] = 14, 8
np.random.seed(1)
print('Tensorflow version:', tf.__version__)

print("LSTM exploration script begin.")

################################################
# LSTM Univariate Retrieve and Preprocess Data #
################################################

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

# GET DATA #
#########################################
df_full, sensor_array = anomaly_utilities.get_data(site, sensor, year, path="./LRO_data/")
df = sensor_array[sensor[0]]

# Valid data must be used to train the detector. Options include:
#   - Use corrected data to train the detector. This is problematic due to pervasive drift corrections throughout.
#       Also problematic because of -9999 values in the data.
#   - Use a subset of raw data that is in decent shape without NaNs, -9999 values, or data gaps.
#       df_sub = df.loc['2017-01-01 00:00':'2017-07-01 00:00']
#   - Use raw data that have been preprocessed to filter out extreme values and have drift correction applied.
#   - Either with raw or corrected data for training, use data that are not labeled as anomalous/corrected.
#   df_cor = df_cor.replace(-9999, np.NaN)

# RULES BASED DETECTION #
#########################################
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

#########################################
# LSTM Univariate Vanilla Model #
#########################################

# MODEL CREATION #
#########################################
# scales data, reshapes data, builds and trains model, evaluates model results
time_steps = 10
samples = 5000
cells = 128
dropout = 0.2
patience = 6

lstm_univar = modeling_utilities.lstm_univar(df, time_steps, samples, cells, dropout, patience)

plt.plot(lstm_univar.history.history['loss'], label='Training Loss')
plt.plot(lstm_univar.history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# DETERMINE THRESHOLD AND DETECT ANOMALIES #
############################################
threshold = anomaly_utilities.set_dynamic_threshold(lstm_univar.test_residuals[0], 75, 0.01, 4)
threshold.index = df[time_steps:].index

residuals = pd.DataFrame(lstm_univar.test_residuals)
residuals.index = threshold.index

plt.figure()
# plt.plot(df['raw'], 'b', label='original data')
plt.plot(residuals, 'b', label='residuals')
plt.plot(threshold['low'], 'c', label='thresh_low')
plt.plot(threshold['high'], 'm', mfc='none', label='thresh_high')
plt.legend()
plt.ylabel(sensor)
plt.show()

observed = df[['observed']][time_steps:]
detections = anomaly_utilities.detect_anomalies(observed, lstm_univar.predictions, lstm_univar.test_residuals, threshold, summary=True)

# Use events function to widen and number anomalous events
df_anomalies = df.iloc[time_steps:]
df_anomalies['labeled_event'] = anomaly_utilities.anomaly_events(df_anomalies['labeled_anomaly'])
df_anomalies['detected_anomaly'] = detections['anomaly']
df_anomalies['all_anomalies'] = df_anomalies.eval('detected_anomaly or anomaly')
df_anomalies['detected_event'] = anomaly_utilities.anomaly_events(df_anomalies['all_anomalies'])

# DETERMINE METRICS #
#########################################
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
plt.plot(detections['prediction'], 'c', label='predicted values')
plt.plot(df['raw'][df['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
plt.plot(detections['prediction'][df_anomalies[df_anomalies['detected_event'] > 0]], 'r+', label='machine detected anomalies')
plt.legend()
plt.ylabel(sensor)
plt.show()

#########################################
# LSTM Univariate Bidirectional Model #
#########################################

# MODEL CREATION #
#########################################
# scales data, reshapes data, builds and trains model, evaluates model results
time_steps = 10
samples = 5000
cells = 128
dropout = 0.2
patience = 6

lstm_univar_bidir = modeling_utilities.lstm_univar_bidir(df, time_steps, samples, cells, dropout, patience)

# Plot Metrics and Evaluate the Model
# plot training loss and validation loss with matplotlib and pyplot
plt.plot(lstm_univar_bidir.history.history['loss'], label='Training Loss')
plt.plot(lstm_univar_bidir.history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# DETERMINE THRESHOLD AND DETECT ANOMALIES #
#########################################
threshold = anomaly_utilities.set_dynamic_threshold(lstm_univar_bidir.test_residuals[0], 75, 0.01, 4)
threshold.index = df[time_steps:-time_steps].index

residuals = pd.DataFrame(lstm_univar_bidir.test_residuals)
residuals.index = threshold.index

plt.figure()
# plt.plot(df['raw'], 'b', label='original data')
plt.plot(residuals, 'b', label='residuals')
plt.plot(threshold['low'], 'c', label='thresh_low')
plt.plot(threshold['high'], 'm', mfc='none', label='thresh_high')
plt.legend()
plt.ylabel(sensor)
plt.show()

observed = df[['observed']][time_steps:-time_steps]
detections = anomaly_utilities.detect_anomalies(observed, lstm_univar_bidir.predictions, lstm_univar_bidir.test_residuals, threshold, summary=True)

# Use events function to widen and number anomalous events
df_anomalies = df.iloc[time_steps:]
df_anomalies['labeled_event'] = anomaly_utilities.anomaly_events(df_anomalies['labeled_anomaly'])
df_anomalies['detected_anomaly'] = detections['anomaly']
df_anomalies['all_anomalies'] = df_anomalies.eval('detected_anomaly or anomaly')
df_anomalies['detected_event'] = anomaly_utilities.anomaly_events(df_anomalies['all_anomalies'])

# DETERMINE METRICS #
#########################################
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
plt.plot(detections['prediction'], 'c', label='predicted values')
plt.plot(df['raw'][df['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
plt.plot(detections['prediction'][df_anomalies[df_anomalies['detected_event'] > 0]], 'r+', label='machine detected anomalies')
plt.legend()
plt.ylabel(sensor)
plt.show()

##################################################
# LSTM Multivariate Retrieve and Preprocess Data #
##################################################

# DEFINE SITE and VARIABLE #
#########################################
# site = "BlackSmithFork"
# site = "FranklinBasin"
site = "MainStreet"
# site = "Mendon"
# site = "TonyGrove"
# site = "WaterLab"
sensor = ['temp', 'cond', 'ph', 'do']
year = 2014

# GET DATA #
#########################################
df_full, sensor_array = anomaly_utilities.get_data(site, sensor, year, path="./LRO_data/")

# RULES BASED DETECTION #
#########################################
# General sensor ranges for LRO data:
# Temp min: -5, max: 30
# SpCond min: 100, max: 900
# pH min: 7.5, max: 9.0
# do min: 2, max: 16

maximum = [30, 900, 9.0, 16]
minimum = [-5, 100, 7.5, 2]
length = 6

size = []
for i in range(0, len(sensor_array)):
    sensor_array[sensor[i]] = rules_detect.range_check(sensor_array[sensor[i]], maximum[i], minimum[i])
    sensor_array[sensor[i]] = rules_detect.persistence(sensor_array[sensor[i]], length)
    s = rules_detect.group_size(sensor_array[sensor[i]])
    size.append(s)
    sensor_array[sensor[i]] = rules_detect.interpolate(sensor_array[sensor[i]])

# Create new data frame with raw and corrected data for variables of interest
df_observed = pd.DataFrame(index=df_full.index)
df_observed['temp_obs'] = sensor_array['temp']['observed']
df_observed['cond_obs'] = sensor_array['cond']['observed']
df_observed['ph_obs'] = sensor_array['ph']['observed']
df_observed['do_obs'] = sensor_array['do']['observed']

df_raw = pd.DataFrame(index=df_full.index)
df_raw['temp'] = df_full['temp']
df_raw['cond'] = df_full['cond']
df_raw['ph'] = df_full['ph']
df_raw['do'] = df_full['do']

df_anomaly = pd.DataFrame(index=df_full.index)
df_anomaly['temp_anom'] = sensor_array['temp']['anomaly']
df_anomaly['cond_anom'] = sensor_array['cond']['anomaly']
df_anomaly['ph_anom'] = sensor_array['ph']['anomaly']
df_anomaly['do_anom'] = sensor_array['do']['anomaly']

print(df_observed.shape)
print(df_raw.shape)
print(df_anomaly.shape)

#########################################
# LSTM Multivariate Vanilla Model #
#########################################

# MODEL CREATION #
#########################################
# scales data, reshapes data, builds and trains model, evaluates model results
time_steps = 10
samples = 5000
cells = 128
dropout = 0.2
patience = 6

lstm_multivar = modeling_utilities.lstm_multivar(df_observed, df_anomaly, df_raw, time_steps, samples, cells, dropout, patience)

# Plot Metrics and Evaluate the Model
# plot training loss and validation loss with matplotlib and pyplot
plt.plot(lstm_multivar.history.history['loss'], label='Training Loss')
plt.plot(lstm_multivar.history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# DETERMINE THRESHOLD AND DETECT ANOMALIES #
#########################################
residuals = pd.DataFrame(lstm_multivar.test_residuals)
predictions = pd.DataFrame(lstm_multivar.predictions)
residuals.index = df_observed[time_steps:].index
predictions.index = df_observed[time_steps:].index

window_sz = [40, 40, 40, 40]
alpha = [0.01, 0.01, 0.01, 0.01]
min_range = [0.2, 4, 0.02, 0.04]

threshold = []
for i in range(0, lstm_multivar.test_residuals.shape[1]):
     threshold_df = anomaly_utilities.set_dynamic_threshold(residuals.iloc[:, i], window_sz[i], alpha[i], min_range[i])
     threshold_df.index = residuals.index
     threshold.append(threshold_df)

     plt.figure()
     # plt.plot(df['raw'], 'b', label='original data')
     plt.plot(residuals.iloc[:, i], 'b', label='residuals')
     plt.plot(threshold[i]['low'], 'c', label='thresh_low')
     plt.plot(threshold[i]['high'], 'm', mfc='none', label='thresh_high')
     plt.legend()
     plt.ylabel(sensor[i])
     plt.show()

observed = df_observed[time_steps:]
detections_array = []
for i in range(0, observed.shape[1]):
    detections_df = anomaly_utilities.detect_anomalies(observed.iloc[:, i], lstm_multivar.predictions.iloc[:, i], lstm_multivar.test_residuals.iloc[:, i], threshold[i], summary=True)
    detections_array.append(detections_df)

# Use events function to widen and number anomalous events
df_array = []
for i in range(0, len(detections_array)):
    all_data = []
    all_data = sensor_array[sensor[i]].iloc[time_steps:]
    all_data['labeled_event'] = anomaly_utilities.anomaly_events(all_data['labeled_anomaly'])
    all_data['detected_anomaly'] = detections_array[i]['anomaly']
    all_data['all_anomalies'] = all_data.eval('detected_anomaly or anomaly')
    all_data['detected_event'] = anomaly_utilities.anomaly_events(all_data['all_anomalies'])
    df_array.append(all_data)

# DETERMINE METRICS #
#########################################
anomaly_utilities.compare_events(df_array[0], 0)
temp_metrics = anomaly_utilities.metrics(df_array[0])

anomaly_utilities.compare_events(df_array[1], 0)
cond_metrics = anomaly_utilities.metrics(df_array[1])

anomaly_utilities.compare_events(df_array[2], 0)
ph_metrics = anomaly_utilities.metrics(df_array[2])

anomaly_utilities.compare_events(df_array[3], 0)
do_metrics = anomaly_utilities.metrics(df_array[3])

# OUTPUT RESULTS #
#########################################
print('\n\n\nScript report:\n')
print('Sensor: temp')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
anomaly_utilities.print_metrics(temp_metrics)

print('\n\n\nScript report:\n')
print('Sensor: cond')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
anomaly_utilities.print_metrics(cond_metrics)

print('\n\n\nScript report:\n')
print('Sensor: ph')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
anomaly_utilities.print_metrics(ph_metrics)

print('\n\n\nScript report:\n')
print('Sensor: do')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
anomaly_utilities.print_metrics(do_metrics)


# GENERATE PLOTS #
#########################################
for i in range(0, len(sensor)):
    plt.figure()
    plt.plot(df_raw[df_raw.columns[i]], 'b', label='original data')
    #plt.plot(df_observed[df_observed.columns[i]], 'm', label='corrected data' )
    plt.plot(detections_array[i]['prediction'], 'c', label='predicted values')
    plt.plot(sensor_array[sensor[i]]['raw'][sensor_array[sensor[i]]['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
    plt.plot(detections_array[i]['prediction'][detections_array[i]['anomaly']], 'r+', label='machine detected anomalies')
    plt.plot(detections_array[i]['prediction'][df_array[i][df_array[i]['detected_event'] > 0]], 'r+', label='machine detected anomalies')
    plt.legend()
    plt.ylabel(sensor[i])
    plt.show()

#########################################
# LSTM Multivariate Bidirectional Model #
#########################################

# Model creation #
#########################################
# scales data, reshapes data, builds and trains model, evaluates model results
time_steps = 10
samples = 5000
cells = 128
dropout = 0.2
patience = 6

lstm_multivar_bidir = modeling_utilities.lstm_multivar_bidir(df_observed, df_anomaly, df_raw, time_steps, samples, cells, dropout, patience)

# Plot Metrics and Evaluate the Model
# plot training loss and validation loss with matplotlib and pyplot
plt.plot(lstm_multivar_bidir.history.history['loss'], label='Training Loss')
plt.plot(lstm_multivar_bidir.history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# DETERMINE THRESHOLD AND DETECT ANOMALIES #
#########################################
residuals = pd.DataFrame(lstm_multivar_bidir.test_residuals)
residuals.index = df_observed[time_steps:-time_steps].index

window_sz = [40, 40, 40, 40]
alpha = [0.01, 0.01, 0.01, 0.01]
min_range = [0.2, 4, 0.02, 0.04]

threshold = []
for i in range(0, lstm_multivar_bidir.test_residuals.shape[1]):
     threshold_df = anomaly_utilities.set_dynamic_threshold(residuals.iloc[:, i], window_sz[i], alpha[i], min_range[i])
     threshold_df.index = residuals.index
     threshold.append(threshold_df)

     plt.figure()
     # plt.plot(df['raw'], 'b', label='original data')
     plt.plot(residuals.iloc[:, i], 'b', label='residuals')
     plt.plot(threshold[i]['low'], 'c', label='thresh_low')
     plt.plot(threshold[i]['high'], 'm', mfc='none', label='thresh_high')
     plt.legend()
     plt.ylabel(sensor[i])
     plt.show()

observed = df_observed[time_steps:-time_steps]
detections_array = []
for i in range(0, observed.shape[1]):
    detections_df = anomaly_utilities.detect_anomalies(observed.iloc[:, i], lstm_multivar_bidir.predictions.iloc[:, i], lstm_multivar_bidir.test_residuals.iloc[:, i], threshold[i], summary=True)
    detections_array.append(detections_df)

# Use events function to widen and number anomalous events
df_array = []
for i in range(0, len(detections_array)):
    all_data = []
    all_data = sensor_array[sensor[i]].iloc[time_steps:]
    all_data['labeled_event'] = anomaly_utilities.anomaly_events(all_data['labeled_anomaly'])
    all_data['detected_anomaly'] = detections_array[i]['anomaly']
    all_data['detected_event'] = anomaly_utilities.anomaly_events(all_data['detected_anomaly'])
    df_array.append(all_data)

# DETERMINE METRICS #
#########################################
anomaly_utilities.compare_events(df_array[0], 0)
temp_metrics = anomaly_utilities.metrics(df_array[0])

anomaly_utilities.compare_events(df_array[1], 0)
cond_metrics = anomaly_utilities.metrics(df_array[1])

anomaly_utilities.compare_events(df_array[2], 0)
ph_metrics = anomaly_utilities.metrics(df_array[2])

anomaly_utilities.compare_events(df_array[3], 0)
do_metrics = anomaly_utilities.metrics(df_array[3], compare_temp.valid_detections, compare_temp.invalid_detections)

# OUTPUT RESULTS #
#########################################
print('\n\n\nScript report:\n')
print('Sensor: temp')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
anomaly_utilities.print_metrics(temp_metrics)

print('\n\n\nScript report:\n')
print('Sensor: cond')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
anomaly_utilities.print_metrics(cond_metrics)

print('\n\n\nScript report:\n')
print('Sensor: ph')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
anomaly_utilities.print_metrics(ph_metrics)

print('\n\n\nScript report:\n')
print('Sensor: do')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
anomaly_utilities.print_metrics(do_metrics)


# GENERATE PLOTS #
#########################################
for i in range(0, len(sensor)):
    plt.figure()
    plt.plot(df_raw[df_raw.columns[i]], 'b', label='original data')
    #plt.plot(df_observed[df_observed.columns[i]], 'm', label='corrected data' )
    plt.plot(detections_array[i]['prediction'], 'c', label='predicted values')
    plt.plot(sensor_array[sensor[i]]['raw'][sensor_array[sensor[i]]['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
    plt.plot(detections_array[i]['prediction'][df_array[i][df_array[i]['detected_event'] > 0]], 'r+', label='machine detected anomalies')
    plt.legend()
    plt.ylabel(sensor[i])
    plt.show()

print("\n LSTM script end.")
