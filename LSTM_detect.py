################################
# LSTM DEVELOP AND DIAGNOSTIC #
################################
# This code takes raw data and corrected data, applies an LSTM model, and identifies anomalies.

print("LSTM exploration script begin.")

import rules_detect
import anomaly_utilities
import LSTM_utilities
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
# tf.random.set_seed(1)
print('Tensorflow version:', tf.__version__)

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
df_full, sensor_array = anomaly_utilities.get_data(site, sensor, year, path="/Users/amber/PycharmProjects/LRO-anomaly-detection/LRO_data/")
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

LSTM_univar = LSTM_utilities.LSTM_univar(df, time_steps, samples, cells, dropout, patience)

plt.plot(LSTM_univar.history.history['loss'], label='Training Loss')
plt.plot(LSTM_univar.history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# look at the distribution of the errors using a distribution plot
# could find a way to do a 95% percentile or some other actual value to automatically select the threshold.
# However, that is a set number of anomalies.
sns.distplot(LSTM_univar.train_residuals, bins=50, kde=True)
plt.show()
# choose a threshold to use for anomalies based on x-axis.
threshold = [18]
sns.distplot(LSTM_univar.test_residuals, bins=50, kde=True)
plt.show()

# DETECT ANOMALIES #
#########################################
test_data = df[['det_cor']]
test_score_array = LSTM_utilities.detect_anomalies(test_data, LSTM_univar.predictions, time_steps, LSTM_univar.test_residuals, threshold)

# Use events function to widen and number anomalous events
df_anomalies = df.iloc[time_steps:]
df_anomalies['labeled_event'] = anomaly_utilities.anomaly_events(df_anomalies['labeled_anomaly'])
df_anomalies['detected_anomaly'] = test_score_array[0]['anomaly']
df_anomalies['detected_event'] = anomaly_utilities.anomaly_events(df_anomalies['detected_anomaly'])

# DETERMINE METRICS #
#########################################
compare = anomaly_utilities.compare_labeled_detected(df_anomalies)
metrics = anomaly_utilities.metrics(df_anomalies, compare.valid_detections, compare.invalid_detections)

# OUTPUT RESULTS #
#########################################
print('\n\n\nScript report:\n')
print('Sensor: ' + sensor[0])
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
print('PPV = %f' % metrics.PPV)
print('NPV = %f' % metrics.NPV)
print('Acc = %f' % metrics.ACC)
print('TP  = %i' % metrics.TruePositives)
print('TN  = %i' % metrics.TrueNegatives)
print('FP  = %i' % metrics.FalsePositives)
print('FN  = %i' % metrics.FalseNegatives)
print('F1 = %f' % metrics.f1)
print('F2 = %f' % metrics.f2)
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

LSTM_univar_bidir = LSTM_utilities.LSTM_univar_bidir(df, time_steps, samples, cells, dropout, patience)

# Plot Metrics and Evaluate the Model
# plot training loss and validation loss with matplotlib and pyplot
plt.plot(LSTM_univar_bidir.history.history['loss'], label='Training Loss')
plt.plot(LSTM_univar_bidir.history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# look at the distribution of the errors using a distribution plot
# could find a way to do a 95% percentile or some other actual value to automatically select the threshold.
# However, that is a set number of anomalies.
sns.distplot(LSTM_univar_bidir.train_residuals, bins=50, kde=True)
plt.show()
# choose a threshold to use for anomalies based on x-axis. try where error is greater than 0.75, it's anomalous.
threshold = [18]
sns.distplot(LSTM_univar_bidir.test_residuals, bins=50, kde=True)
plt.show()

# DETECT ANOMALIES #
#########################################
test_data = df[['det_cor']]
test_score_array = LSTM_utilities.detect_anomalies_bidir(test_data, LSTM_univar_bidir.predictions, time_steps, LSTM_univar_bidir.test_residuals, threshold)

# Use events function to widen and number anomalous events
df_anomalies = df.iloc[time_steps:]
df_anomalies['labeled_event'] = anomaly_utilities.anomaly_events(df_anomalies['labeled_anomaly'])
df_anomalies['detected_anomaly'] = test_score_array[0]['anomaly']
df_anomalies['detected_event'] = anomaly_utilities.anomaly_events(df_anomalies['detected_anomaly'])

# DETERMINE METRICS #
#########################################
compare = anomaly_utilities.compare_labeled_detected(df_anomalies)
metrics = anomaly_utilities.metrics(df_anomalies, compare.valid_detections, compare.invalid_detections)

# OUTPUT RESULTS #
#########################################
print('\n\n\nScript report:\n')
print('Sensor: ' + sensor[0])
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
print('PPV = %f' % metrics.PPV)
print('NPV = %f' % metrics.NPV)
print('Acc = %f' % metrics.ACC)
print('TP  = %i' % metrics.TruePositives)
print('TN  = %i' % metrics.TrueNegatives)
print('FP  = %i' % metrics.FalsePositives)
print('FN  = %i' % metrics.FalseNegatives)
print('F1 = %f' % metrics.f1)
print('F2 = %f' % metrics.f2)
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
df_full, sensor_array = anomaly_utilities.get_data(site, sensor, year, path="/Users/amber/PycharmProjects/LRO-anomaly-detection/LRO_data/")

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

LSTM_multivar = LSTM_utilities.LSTM_multivar(df_det_cor, df_anomaly, df_raw, time_steps, samples, cells, dropout, patience)

# Plot Metrics and Evaluate the Model
# plot training loss and validation loss with matplotlib and pyplot
plt.plot(LSTM_multivar.history.history['loss'], label='Training Loss')
plt.plot(LSTM_multivar.history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Look at the distribution of the errors using distribution plots
for i in range(0, LSTM_multivar.train_residuals.shape[1]):
    plt.figure()
    sns.distplot(LSTM_multivar.train_residuals[i], bins=50, kde=True)
    plt.show()

# Choose thresholds to use for anomalies based on the x-axes.
threshold = [0.5, 50, 0.1, 2.0]
# Examine errors in the test data
for i in range(0, LSTM_multivar.test_residuals.shape[1]):
    plt.figure()
    sns.distplot(LSTM_multivar.test_residuals[i], bins=50, kde=True)
    plt.show()

# DETECT ANOMALIES #
#########################################
test_score_array = LSTM_utilities.detect_anomalies(df_det_cor, LSTM_multivar.predictions, time_steps, LSTM_multivar.test_residuals, threshold)

# Use events function to widen and number anomalous events
df_array = []
for i in range(0, len(test_score_array)):
    all_data = []
    all_data = sensor_array[sensor[i]].iloc[time_steps:]
    all_data['labeled_event'] = anomaly_utilities.anomaly_events(all_data['labeled_anomaly'])
    all_data['detected_anomaly'] = test_score_array[i]['anomaly']
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
print('PPV = %f' % temp_metrics.PPV)
print('NPV = %f' % temp_metrics.NPV)
print('Acc = %f' % temp_metrics.ACC)
print('TP  = %i' % temp_metrics.TruePositives)
print('TN  = %i' % temp_metrics.TrueNegatives)
print('FP  = %i' % temp_metrics.FalsePositives)
print('FN  = %i' % temp_metrics.FalseNegatives)
print('F1 = %f' % temp_metrics.f1)
print('F2 = %f' % temp_metrics.f2)

print('\n\n\nScript report:\n')
print('Sensor: cond')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
print('PPV = %f' % cond_metrics.PPV)
print('NPV = %f' % cond_metrics.NPV)
print('Acc = %f' % cond_metrics.ACC)
print('TP  = %i' % cond_metrics.TruePositives)
print('TN  = %i' % cond_metrics.TrueNegatives)
print('FP  = %i' % cond_metrics.FalsePositives)
print('FN  = %i' % cond_metrics.FalseNegatives)
print('F1 = %f' % cond_metrics.f1)
print('F2 = %f' % cond_metrics.f2)

print('\n\n\nScript report:\n')
print('Sensor: ph')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
print('PPV = %f' % ph_metrics.PPV)
print('NPV = %f' % ph_metrics.NPV)
print('Acc = %f' % ph_metrics.ACC)
print('TP  = %i' % ph_metrics.TruePositives)
print('TN  = %i' % ph_metrics.TrueNegatives)
print('FP  = %i' % ph_metrics.FalsePositives)
print('FN  = %i' % ph_metrics.FalseNegatives)
print('F1 = %f' % ph_metrics.f1)
print('F2 = %f' % ph_metrics.f2)

print('\n\n\nScript report:\n')
print('Sensor: do')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
print('PPV = %f' % do_metrics.PPV)
print('NPV = %f' % do_metrics.NPV)
print('Acc = %f' % do_metrics.ACC)
print('TP  = %i' % do_metrics.TruePositives)
print('TN  = %i' % do_metrics.TrueNegatives)
print('FP  = %i' % do_metrics.FalsePositives)
print('FN  = %i' % do_metrics.FalseNegatives)
print('F1 = %f' % do_metrics.f1)
print('F2 = %f' % do_metrics.f2)

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

LSTM_multivar_bidir = LSTM_utilities.LSTM_multivar_bidir(df_det_cor, df_anomaly, df_raw, time_steps, samples, cells, dropout, patience)

# Plot Metrics and Evaluate the Model
# plot training loss and validation loss with matplotlib and pyplot
plt.plot(LSTM_multivar_bidir.history.history['loss'], label='Training Loss')
plt.plot(LSTM_multivar_bidir.history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Look at the distribution of the errors using distribution plots
for i in range(0, LSTM_multivar_bidir.train_residuals.shape[1]):
    plt.figure()
    sns.distplot(LSTM_multivar_bidir.train_residuals[i], bins=50, kde=True)
    plt.show()

# Choose thresholds to use for anomalies based on the x-axes.
threshold = [0.5, 50, 0.1, 2.0]
# Examine errors in the test data
for i in range(0, LSTM_multivar_bidir.test_residuals.shape[1]):
    plt.figure()
    sns.distplot(LSTM_multivar_bidir.test_residuals[i], bins=50, kde=True)
    plt.show()

# DETECT ANOMALIES #
#########################################
test_score_array = LSTM_utilities.detect_anomalies_bidir(df_det_cor, LSTM_multivar_bidir.predictions, time_steps, LSTM_multivar_bidir.test_residuals, threshold)

# Use events function to widen and number anomalous events
df_array = []
for i in range(0, len(test_score_array)):
    all_data = []
    all_data = sensor_array[sensor[i]].iloc[time_steps:]
    all_data['labeled_event'] = anomaly_utilities.anomaly_events(all_data['labeled_anomaly'])
    all_data['detected_anomaly'] = test_score_array[i]['anomaly']
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
print('PPV = %f' % temp_metrics.PPV)
print('NPV = %f' % temp_metrics.NPV)
print('Acc = %f' % temp_metrics.ACC)
print('TP  = %i' % temp_metrics.TruePositives)
print('TN  = %i' % temp_metrics.TrueNegatives)
print('FP  = %i' % temp_metrics.FalsePositives)
print('FN  = %i' % temp_metrics.FalseNegatives)
print('F1 = %f' % temp_metrics.f1)
print('F2 = %f' % temp_metrics.f2)

print('\n\n\nScript report:\n')
print('Sensor: cond')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
print('PPV = %f' % cond_metrics.PPV)
print('NPV = %f' % cond_metrics.NPV)
print('Acc = %f' % cond_metrics.ACC)
print('TP  = %i' % cond_metrics.TruePositives)
print('TN  = %i' % cond_metrics.TrueNegatives)
print('FP  = %i' % cond_metrics.FalsePositives)
print('FN  = %i' % cond_metrics.FalseNegatives)
print('F1 = %f' % cond_metrics.f1)
print('F2 = %f' % cond_metrics.f2)

print('\n\n\nScript report:\n')
print('Sensor: ph')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
print('PPV = %f' % ph_metrics.PPV)
print('NPV = %f' % ph_metrics.NPV)
print('Acc = %f' % ph_metrics.ACC)
print('TP  = %i' % ph_metrics.TruePositives)
print('TN  = %i' % ph_metrics.TrueNegatives)
print('FP  = %i' % ph_metrics.FalsePositives)
print('FN  = %i' % ph_metrics.FalseNegatives)
print('F1 = %f' % ph_metrics.f1)
print('F2 = %f' % ph_metrics.f2)

print('\n\n\nScript report:\n')
print('Sensor: do')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
print('PPV = %f' % do_metrics.PPV)
print('NPV = %f' % do_metrics.NPV)
print('Acc = %f' % do_metrics.ACC)
print('TP  = %i' % do_metrics.TruePositives)
print('TN  = %i' % do_metrics.TrueNegatives)
print('FP  = %i' % do_metrics.FalsePositives)
print('FN  = %i' % do_metrics.FalseNegatives)
print('F1 = %f' % do_metrics.f1)
print('F2 = %f' % do_metrics.f2)


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
