################################
# LSTM MULTIVARIATE DEVELOP AND DIAGNOSTIC #
################################
# This code takes raw data and corrected data, applies an LSTM model, and identifies anomalies.

print("LSTM multivariate script begin.")

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
df_full, sensor_array = anomaly_utilities.get_data(site, sensor, year, path="/Users/amber/PycharmProjects/LRO-anomaly-detection/LRO_data/")

# Create new data frame with raw and corrected data for variables of interest
df_cor = pd.DataFrame(index=df_full.index)
df_cor['temp_cor'] = df_full['temp_cor']
df_cor['cond_cor'] = df_full['cond_cor']
df_cor['ph_cor'] = df_full['ph_cor']
df_cor['do_cor'] = df_full['do_cor']

df_raw = pd.DataFrame(index=df_full.index)
df_raw['temp'] = df_full['temp']
df_raw['cond'] = df_full['cond']
df_raw['ph'] = df_full['ph']
df_raw['do'] = df_full['do']

print(df_cor.shape)
print(df_raw.shape)

# Scale data. Scale all columns and put in new data frame.
scaler = LSTM_utilities.create_scaler(df_cor)
df_scaled = pd.DataFrame(scaler.transform(df_cor), index=df_cor.index, columns=df_cor.columns)
print(df_scaled.shape)

# Create datasets with sequences
time_steps = 50
samples = 10000
X_train_mult, y_train_mult = LSTM_utilities.create_training_dataset(df_scaled, samples, time_steps)
print("X_train_mult.shape: " + str(X_train_mult.shape))
print("y_train_mult.shape: " + str(y_train_mult.shape))

# Create and train model.
# If model uses an autoencoder, the input needs to be the same shape as the output.
# For a vanilla model, it doesn't matter.
num_features = X_train_mult.shape[2]
# model = LSTM_utilities.create_model(128, time_steps, num_features, 0.2)
model = LSTM_utilities.create_vanilla_model(128, time_steps, num_features, 0.2)
model.summary()
# history = LSTM_utilities.train_model(X_train_mult, X_train_mult, model, patience=3)
history = LSTM_utilities.train_model(X_train_mult, y_train_mult, model, patience=3)

# Plot Metrics and Evaluate the Model
# plot training loss and validation loss with matplotlib and pyplot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Create dataset on full raw data. First scale according to existing scaler.
df_raw_scaled = pd.DataFrame(scaler.transform(df_raw), index=df_raw.index, columns=df_raw.columns)
print(df_raw_scaled.shape)

# Create sequenced dataset.
X_raw, y_raw = LSTM_utilities.create_sequenced_dataset(df_raw_scaled, time_steps)
print("X_raw.shape: " + str(X_raw.shape))
print("y_raw.shape: " + str(y_raw.shape))

# Evaluate the model on the raw data
# X_train_pred, train_mae_loss, model_eval, X_test_pred, test_mae_loss, predictions = LSTM_utilities.evaluate_model(X_train_mult, X_raw, X_raw, model)
train_pred, train_mae_loss, model_eval, test_pred, test_mae_loss, predictions = LSTM_utilities.evaluate_vanilla_model(X_train_mult, y_train_mult, X_raw, y_raw, model)

# Look at the distribution of the errors using distribution plots
for i in range(0, train_mae_loss.shape[1]):
    plt.figure()
    sns.distplot(train_mae_loss[i], bins=50, kde=True)
    plt.show()

# Choose thresholds to use for anomalies based on the x-axes.
threshold = [0.4, 0.5, 0.5, 0.3]
# Examine errors in the test data
for i in range(0, test_mae_loss.shape[1]):
    plt.figure()
    sns.distplot(test_mae_loss[i], bins=50, kde=True)
    plt.show()

# Unscale predictions back to original units
predictions_unscaled = pd.DataFrame(scaler.inverse_transform(predictions), columns=df_cor.columns)

# Detect anomalies
test_score_array = LSTM_utilities.detect_anomalies(df_raw_scaled, predictions, predictions_unscaled, time_steps, test_mae_loss, threshold)

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
    plt.plot(df_cor[df_cor.columns[i]], 'm', label='corrected data' )
    plt.plot(test_score_array[i]['pred_unscaled'], 'c', label='predicted values')
    plt.plot(sensor_array[sensor[i]]['raw'][sensor_array[sensor[i]]['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
    plt.plot(test_score_array[i]['pred_unscaled'][test_score_array[i]['anomaly']], 'r+', label='machine detected anomalies')
    plt.legend()
    plt.ylabel(sensor[i])
    plt.show()


print("\n LSTM script end.")