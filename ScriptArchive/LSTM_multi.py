################################
# LSTM MULTIVARIATE DEVELOP AND DIAGNOSTIC #
################################
# This code takes raw data and corrected data, applies an LSTM model, and identifies anomalies.

print("LSTM multivariate script begin.")

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

# EXECUTE FUNCTIONS #
#########################################
# Get data
df_full, sensor_array = anomaly_utilities.get_data(site, sensor, year, path="./LRO_data/")

# Rules based detection
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
df_det_cor = pd.DataFrame(index=df_full.index)
df_det_cor['temp_cor'] = sensor_array['temp']['det_cor']
df_det_cor['cond_cor'] = sensor_array['cond']['det_cor']
df_det_cor['ph_cor'] = sensor_array['ph']['det_cor']
df_det_cor['do_cor'] = sensor_array['do']['det_cor']

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

print(df_det_cor.shape)
print(df_raw.shape)

# Model creation
# scales data, reshapes data, builds and trains model, evaluates model results
time_steps = 10
samples = 5000
cells = 128
dropout = 0.2
patience = 6

X_train, y_train, model, history, X_test, y_test, model_eval, predictions, train_residuals, test_residuals = modeling_utilities.multi_vanilla_LSTM_model(df_det_cor, df_anomaly, df_raw, time_steps, samples, cells, dropout, patience)

# Plot Metrics and Evaluate the Model
# plot training loss and validation loss with matplotlib and pyplot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Look at the distribution of the errors using distribution plots
for i in range(0, train_residuals.shape[1]):
    plt.figure()
    sns.distplot(train_residuals[i], bins=50, kde=True)
    plt.show()

# Choose thresholds to use for anomalies based on the x-axes.
threshold = [0.5, 50, 0.1, 2.0]
# Examine errors in the test data
for i in range(0, test_residuals.shape[1]):
    plt.figure()
    sns.distplot(test_residuals[i], bins=50, kde=True)
    plt.show()

# Detect anomalies
test_score_array = modeling_utilities.detect_anomalies(df_det_cor, predictions, time_steps, test_residuals, threshold)

# Use events function to widen and number anomalous events
df_array = []
for i in range(0, len(test_score_array)):
    all_data = []
    all_data = sensor_array[sensor[i]].iloc[time_steps:]
    all_data['labeled_event'] = anomaly_utilities.anomaly_events(all_data['labeled_anomaly'])
    all_data['detected_anomaly'] = test_score_array[i]['anomaly']
    all_data['detected_event'] = anomaly_utilities.anomaly_events(all_data['detected_anomaly'])
    df_array.append(all_data)

# Determine Metrics
compare_temp = anomaly_utilities.compare_labeled_detected(df_array[0])
temp_metrics = anomaly_utilities.metrics(df_array[0], compare_temp.valid_detections, compare_temp.invalid_detections)

compare_cond = anomaly_utilities.compare_labeled_detected(df_array[1])
cond_metrics = anomaly_utilities.metrics(df_array[1], compare_cond.valid_detections, compare_cond.invalid_detections)

compare_ph = anomaly_utilities.compare_labeled_detected(df_array[2])
ph_metrics = anomaly_utilities.metrics(df_array[2], compare_ph.valid_detections, compare_ph.invalid_detections)

compare_do = anomaly_utilities.compare_labeled_detected(df_array[3])
do_metrics = anomaly_utilities.metrics(df_array[3], compare_temp.valid_detections, compare_temp.invalid_detections)


# OUTPUT RESULTS #
#########################################
print('\n\n\nScript report:\n')
print('Sensor: temp')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
anomaly_utilities.print_metrics(temp_metrics)

print('Sensor: cond')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
anomaly_utilities.print_metrics(cond_metrics)

print('Sensor: ph')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
anomaly_utilities.print_metrics(ph_metrics)

print('Sensor: do')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
anomaly_utilities.print_metrics(do_metrics)


# GENERATE PLOTS #
#########################################

for i in range(0, len(sensor)):
    plt.figure()
    plt.plot(df_raw[df_raw.columns[i]], 'b', label='original data')
    plt.plot(df_det_cor[df_det_cor.columns[i]], 'm', label='corrected data' )
    plt.plot(test_score_array[i]['prediction'], 'c', label='predicted values')
    plt.plot(sensor_array[sensor[i]]['raw'][sensor_array[sensor[i]]['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
    plt.plot(test_score_array[i]['prediction'][test_score_array[i]['anomaly']], 'r+', label='machine detected anomalies')
    plt.legend()
    plt.ylabel(sensor[i])
    plt.show()


print("\n LSTM script end.")