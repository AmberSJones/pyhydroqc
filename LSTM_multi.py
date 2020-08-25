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
# site = "MainStreet"
site = "Mendon"
# site = "TonyGrove"
# site = "WaterLab"
# sensor = "temp"
sensor = "cond"
# sensor = "ph"
# sensor = "do"
# sensor = "turb"
# sensor = "stage"
year = 2017

# EXECUTE FUNCTIONS #
#########################################
# Get data
df_full, df = anomaly_utilities.get_data(site, sensor, year, path="/Users/amber/PycharmProjects/LRO-anomaly-detection/LRO_data/")

# Create new data frame with corrected data for variables of interest
df_cor = pd.DataFrame(index=df_full.index)
df_cor['temp_cor'] = df_full['temp_cor']
df_cor['cond_cor'] = df_full['cond_cor']
df_cor['ph_cor'] = df_full['ph_cor']
df_cor['do_cor'] = df_full['do_cor']

print(df_cor.shape)

# Scale data. Scale all columns and put in new data frame.
scaler = LSTM_utilities.create_scaler(df_cor)
df_scaled = pd.DataFrame(scaler.transform(df_cor), index=df_cor.index, columns=df_cor.columns)
print(df_scaled.shape)

# Create datasets with sequences
time_steps = 10
samples = 5000
X_train_mult, y_train_mult = LSTM_utilities.create_training_dataset(df_scaled, samples, time_steps)
print("X_train_mult.shape: " + str(X_train_mult.shape))
print("y_train_mult.shape: " + str(y_train_mult.shape))

# Create and train model. Because model uses an autoencoder, the input is also the output i.e., y = X.
num_features = X_train_mult.shape[2]
model = LSTM_utilities.create_model(128, time_steps, num_features, 0.2)
model.summary()
history = LSTM_utilities.train_model(X_train_mult, X_train_mult, model, patience=3)

# Plot Metrics and Evaluate the Model
# plot training loss and validation loss with matplotlib and pyplot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Create dataset on full raw data. First scale according to existing scaler.
df_raw = pd.DataFrame(index=df_full.index)
df_raw['temp'] = df_full['temp']
df_raw['cond'] = df_full['cond']
df_raw['ph'] = df_full['ph']
df_raw['do'] = df_full['do']

df_raw_scaled = pd.DataFrame(scaler.transform(df_raw), index=df_raw.index, columns=df_raw.columns)
print(df_raw_scaled.shape)

# Create sequenced dataset.
X_raw, y_raw = LSTM_utilities.create_sequenced_dataset(df_raw_scaled, 10)
print("X_raw.shape: " + str(X_raw.shape))
print("y_raw.shape: " + str(y_raw.shape))

# Evaluate the model on the raw data
X_train_pred, train_mae_loss, model_eval, X_test_pred, test_mae_loss, predictions = LSTM_utilities.evaluate_model(X_train_mult, X_raw, X_raw, model)

# Look at the distribution of the errors using distribution plots
for i in range(0, train_mae_loss.shape[1]):
    plt.figure()
    sns.distplot(train_mae_loss[i], bins=50, kde=True)
    plt.show()

# Choose thresholds to use for anomalies based on the x-axes.
threshold = [0.75, 1, 1, 1.5]
# Examine errors in the test data
for i in range(0, test_mae_loss.shape[1]):
    plt.figure()
    sns.distplot(test_mae_loss[i], bins=50, kde=True)
    plt.show()
# what are these plots showing?!?

# Unscale predictions back to original units
predictions_unscaled = pd.DataFrame(scaler.inverse_transform(predictions), columns=df_cor.columns)

# Detect anomalies
test_score_array = LSTM_utilities.detect_anomalies(df_raw_scaled, predictions, predictions_unscaled, time_steps, test_mae_loss, threshold)


plt.figure()
plt.plot(df_raw["do"], 'b', label='raw')
plt.plot(df_cor["do_cor"], 'm', label='corrected')
plt.plot(test_score_array[3]["pred_unscaled"], 'c', label='predicted')
plt.legend()
plt.show()


###########################################


# Use events function to widen and number anomalous events

df_array = []
for i in range(0, len(test_score_array)):
    all_data = []
    all_data = df.iloc[time_steps:]
    all_data['labeled_event'] = anomaly_utilities.anomaly_events(all_data['labeled_anomaly'])
    all_data['detected_anomaly'] = test_score_array[i]['anomaly']
    all_data['detected_event'] = anomaly_utilities.anomaly_events(all_data['detected_anomaly'])
    df_array.append(all_data)

# Determine Metrics
######## TODO: Create an object "metric" that has all of the below as variables within the object.

labeled_in_detected, detected_in_labeled, valid_detections, invalid_detections = anomaly_utilities.compare_labeled_detected(df_array[0])
TruePositives, FalseNegatives, FalsePositives, TrueNegatives, PRC, PPV, NPV, ACC, RCL, f1, f2 \
    = anomaly_utilities.metrics(df_array[0], valid_detections, invalid_detections)

# OUTPUT RESULTS #
#########################################
print('\n\n\nScript report:\n')
print('Sensor: temp')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
print('PPV = %f' % PPV)
print('NPV = %f' % NPV)
print('Acc = %f' % ACC)
print('TP  = %i' % TruePositives)
print('TN  = %i' % TrueNegatives)
print('FP  = %i' % FalsePositives)
print('FN  = %i' % FalseNegatives)
print('F1 = %f' % f1)
print('F2 = %f' % f2)


labeled_in_detected, detected_in_labeled, valid_detections, invalid_detections = anomaly_utilities.compare_labeled_detected(df_array[1])
TruePositives, FalseNegatives, FalsePositives, TrueNegatives, PRC, PPV, NPV, ACC, RCL, f1, f2 \
    = anomaly_utilities.metrics(df_array[1], valid_detections, invalid_detections)

# OUTPUT RESULTS #
#########################################
print('\n\n\nScript report:\n')
print('Sensor: cond')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
print('PPV = %f' % PPV)
print('NPV = %f' % NPV)
print('Acc = %f' % ACC)
print('TP  = %i' % TruePositives)
print('TN  = %i' % TrueNegatives)
print('FP  = %i' % FalsePositives)
print('FN  = %i' % FalseNegatives)
print('F1 = %f' % f1)
print('F2 = %f' % f2)


labeled_in_detected, detected_in_labeled, valid_detections, invalid_detections = anomaly_utilities.compare_labeled_detected(df_array[2])
TruePositives, FalseNegatives, FalsePositives, TrueNegatives, PRC, PPV, NPV, ACC, RCL, f1, f2 \
    = anomaly_utilities.metrics(df_array[2], valid_detections, invalid_detections)

# OUTPUT RESULTS #
#########################################
print('\n\n\nScript report:\n')
print('Sensor: ph')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
print('PPV = %f' % PPV)
print('NPV = %f' % NPV)
print('Acc = %f' % ACC)
print('TP  = %i' % TruePositives)
print('TN  = %i' % TrueNegatives)
print('FP  = %i' % FalsePositives)
print('FN  = %i' % FalseNegatives)
print('F1 = %f' % f1)
print('F2 = %f' % f2)


labeled_in_detected, detected_in_labeled, valid_detections, invalid_detections = anomaly_utilities.compare_labeled_detected(df_array[3])
TruePositives, FalseNegatives, FalsePositives, TrueNegatives, PRC, PPV, NPV, ACC, RCL, f1, f2 \
    = anomaly_utilities.metrics(df_array[3], valid_detections, invalid_detections)


# OUTPUT RESULTS #
#########################################
print('\n\n\nScript report:\n')
print('Sensor: do')
print('Year: ' + str(year))
# print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
print('PPV = %f' % PPV)
print('NPV = %f' % NPV)
print('Acc = %f' % ACC)
print('TP  = %i' % TruePositives)
print('TN  = %i' % TrueNegatives)
print('FP  = %i' % FalsePositives)
print('FN  = %i' % FalseNegatives)
print('F1 = %f' % f1)
print('F2 = %f' % f2)


print("\n LSTM script end.")