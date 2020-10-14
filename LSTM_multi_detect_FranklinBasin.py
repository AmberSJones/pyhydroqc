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

##################################################
# LSTM Multivariate Retrieve and Preprocess Data #
##################################################

# DEFINE SITE and VARIABLE #
#########################################
# site = "BlackSmithFork"
site = "FranklinBasin"
# site = "MainStreet"
# site = "Mendon"
# site = "TonyGrove"
# site = "WaterLab"
sensor = ['temp', 'cond', 'ph', 'do']
year = [2014, 2015, 2016, 2017, 2018, 2019]

# GET DATA #
#########################################
df_full, sensor_array = anomaly_utilities.get_data(site, sensor, year, path="/Users/amber/PycharmProjects/LRO-anomaly-detection/LRO_data/")

# RULES BASED DETECTION #
#########################################
maximum = [13, 380, 9.2, 13]
minimum = [-2, 120, 7.5, 8]
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
samples = 10000
cells = 128
dropout = 0.2
patience = 6

LSTM_multivar = modeling_utilities.LSTM_multivar(df_observed, df_anomaly, df_raw, time_steps, samples, cells, dropout, patience)

# Plot Metrics and Evaluate the Model
# plot training loss and validation loss with matplotlib and pyplot
plt.plot(LSTM_multivar.history.history['loss'], label='Training Loss')
plt.plot(LSTM_multivar.history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# DETERMINE THRESHOLD AND DETECT ANOMALIES #
#########################################
residuals = pd.DataFrame(LSTM_multivar.test_residuals)
predictions = pd.DataFrame(LSTM_multivar.predictions)
residuals.index = df_observed[time_steps:].index
predictions.index = df_observed[time_steps:].index

window_sz = [40, 40, 40, 40]
alpha = [0.0001, 0.0001, 0.0001, 0.001]
min_range = [0.25, 5, 0.01, 0.15]

threshold = []
for i in range(0, LSTM_multivar.test_residuals.shape[1]):
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
    detections_df = anomaly_utilities.detect_anomalies(observed.iloc[:, i], LSTM_multivar.predictions.iloc[:, i], LSTM_multivar.test_residuals.iloc[:, i], threshold[i], summary=True)
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
print('PPV = %f' % temp_metrics.prc)
print('NPV = %f' % temp_metrics.npv)
print('Acc = %f' % temp_metrics.acc)
print('TP  = %i' % temp_metrics.true_positives)
print('TN  = %i' % temp_metrics.true_negatives)
print('FP  = %i' % temp_metrics.false_positives)
print('FN  = %i' % temp_metrics.false_negatives)
print('F1 = %f' % temp_metrics.f1)
print('F2 = %f' % temp_metrics.f2)

print('\n\n\nScript report:\n')
print('Sensor: cond')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
print('PPV = %f' % cond_metrics.prc)
print('NPV = %f' % cond_metrics.npv)
print('Acc = %f' % cond_metrics.acc)
print('TP  = %i' % cond_metrics.true_positives)
print('TN  = %i' % cond_metrics.true_negatives)
print('FP  = %i' % cond_metrics.false_positives)
print('FN  = %i' % cond_metrics.false_negatives)
print('F1 = %f' % cond_metrics.f1)
print('F2 = %f' % cond_metrics.f2)

print('\n\n\nScript report:\n')
print('Sensor: ph')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
print('PPV = %f' % ph_metrics.prc)
print('NPV = %f' % ph_metrics.npv)
print('Acc = %f' % ph_metrics.acc)
print('TP  = %i' % ph_metrics.true_positives)
print('TN  = %i' % ph_metrics.true_negatives)
print('FP  = %i' % ph_metrics.false_positives)
print('FN  = %i' % ph_metrics.false_negatives)
print('F1 = %f' % ph_metrics.f1)
print('F2 = %f' % ph_metrics.f2)

print('\n\n\nScript report:\n')
print('Sensor: do')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
print('PPV = %f' % do_metrics.prc)
print('NPV = %f' % do_metrics.npv)
print('Acc = %f' % do_metrics.acc)
print('TP  = %i' % do_metrics.true_positives)
print('TN  = %i' % do_metrics.true_negatives)
print('FP  = %i' % do_metrics.false_positives)
print('FN  = %i' % do_metrics.false_negatives)
print('F1 = %f' % do_metrics.f1)
print('F2 = %f' % do_metrics.f2)

# GENERATE PLOTS #
#########################################
for i in range(0, len(sensor)):
    plt.figure()
    plt.plot(df_raw[df_raw.columns[i]], 'b', label='original data')
    #plt.plot(df_observed[df_observed.columns[i]], 'm', label='corrected data' )
    plt.plot(detections_array[i]['prediction'], 'c', label='predicted values')
    plt.plot(sensor_array[sensor[i]]['raw'][sensor_array[sensor[i]]['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
    plt.plot(detections_array[i]['prediction'][detections_array[i]['anomaly']], 'r+', label='machine detected anomalies')
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
samples = 10000
cells = 128
dropout = 0.2
patience = 6

LSTM_multivar_bidir = modeling_utilities.LSTM_multivar_bidir(df_observed, df_anomaly, df_raw, time_steps, samples, cells, dropout, patience)

# Plot Metrics and Evaluate the Model
# plot training loss and validation loss with matplotlib and pyplot
plt.plot(LSTM_multivar_bidir.history.history['loss'], label='Training Loss')
plt.plot(LSTM_multivar_bidir.history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# DETERMINE THRESHOLD AND DETECT ANOMALIES #
#########################################
residuals = pd.DataFrame(LSTM_multivar_bidir.test_residuals)
residuals.index = df_observed[time_steps:-time_steps].index

window_sz = [40, 40, 40, 40]
alpha = [0.0001, 0.0001, 0.0001, 0.001]
min_range = [0.25, 5, 0.01, 0.15]

threshold = []
for i in range(0, LSTM_multivar_bidir.test_residuals.shape[1]):
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
    detections_df = anomaly_utilities.detect_anomalies(observed.iloc[:, i], LSTM_multivar_bidir.predictions.iloc[:, i], LSTM_multivar_bidir.test_residuals.iloc[:, i], threshold[i], summary=True)
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
print('PPV = %f' % temp_metrics.prc)
print('NPV = %f' % temp_metrics.npv)
print('Acc = %f' % temp_metrics.acc)
print('TP  = %i' % temp_metrics.true_positives)
print('TN  = %i' % temp_metrics.true_negatives)
print('FP  = %i' % temp_metrics.false_positives)
print('FN  = %i' % temp_metrics.false_negatives)
print('F1 = %f' % temp_metrics.f1)
print('F2 = %f' % temp_metrics.f2)

print('\n\n\nScript report:\n')
print('Sensor: cond')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
print('PPV = %f' % cond_metrics.prc)
print('NPV = %f' % cond_metrics.npv)
print('Acc = %f' % cond_metrics.acc)
print('TP  = %i' % cond_metrics.true_positives)
print('TN  = %i' % cond_metrics.true_negatives)
print('FP  = %i' % cond_metrics.false_positives)
print('FN  = %i' % cond_metrics.false_negatives)
print('F1 = %f' % cond_metrics.f1)
print('F2 = %f' % cond_metrics.f2)

print('\n\n\nScript report:\n')
print('Sensor: ph')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
print('PPV = %f' % ph_metrics.prc)
print('NPV = %f' % ph_metrics.npv)
print('Acc = %f' % ph_metrics.acc)
print('TP  = %i' % ph_metrics.true_positives)
print('TN  = %i' % ph_metrics.true_negatives)
print('FP  = %i' % ph_metrics.false_positives)
print('FN  = %i' % ph_metrics.false_negatives)
print('F1 = %f' % ph_metrics.f1)
print('F2 = %f' % ph_metrics.f2)

print('\n\n\nScript report:\n')
print('Sensor: do')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
print('PPV = %f' % do_metrics.prc)
print('NPV = %f' % do_metrics.npv)
print('Acc = %f' % do_metrics.acc)
print('TP  = %i' % do_metrics.true_positives)
print('TN  = %i' % do_metrics.true_negatives)
print('FP  = %i' % do_metrics.false_positives)
print('FN  = %i' % do_metrics.false_negatives)
print('F1 = %f' % do_metrics.f1)
print('F2 = %f' % do_metrics.f2)

# GENERATE PLOTS #
#########################################
for i in range(0, len(sensor)):
    plt.figure()
    plt.plot(df_raw[df_raw.columns[i]], 'b', label='original data')
    # plt.plot(df_observed[df_observed.columns[i]], 'm', label='corrected data' )
    plt.plot(detections_array[i]['prediction'], 'c', label='predicted values')
    plt.plot(sensor_array[sensor[i]]['raw'][sensor_array[sensor[i]]['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
    plt.plot(detections_array[i]['prediction'][detections_array[i]['anomaly']], 'r+', label='machine detected anomalies')
    plt.legend()
    plt.ylabel(sensor[i])
    plt.show()

print("\n LSTM script end.")
