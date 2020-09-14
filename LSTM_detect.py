################################
# LSTM DEVELOP AND DIAGNOSTIC #
################################
# This code takes raw data and corrected data, applies an LSTM model, and identifies anomalies.

print("LSTM exploration script begin.")

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
df_full, sensor_array = anomaly_utilities.get_data(site, sensor, year, path="/Users/amber/PycharmProjects/LRO-anomaly-detection/LRO_data/")
df = sensor_array[sensor[0]]

# Using corrected data to train detector. Remove -9999 values. Use subset of data without NaNs and data gaps.
# Other rule-based algorithms could be considered here.
# df_cor = df_cor.replace(-9999, np.NaN)
# TODO: generalize from the specific dates
df_sub = df.loc['2017-01-01 00:00':'2017-07-01 00:00']

# Scale data into new column. Scale based on the entire dataset because our model training is on a very small subset
# Use double [] to get single column as data frame rather than series
scaler = LSTM_utilities.create_scaler(df_sub[['cor']])
df_sub['cor_scaled'] = scaler.transform(df_sub[['cor']])

# Create datasets with sequences
time_steps = 50
samples = 10000
X_train, y_train = LSTM_utilities.create_training_dataset(df_sub[['cor_scaled']], samples, time_steps)
print(X_train.shape)
print(y_train.shape)

# Create and model and train to data
num_features = X_train.shape[2]
#model = LSTM_utilities.create_model(128, time_steps, num_features, 0.2)
model = LSTM_utilities.create_vanilla_model(128, time_steps, num_features, 0.2)
model.summary()
history = LSTM_utilities.train_model(X_train, y_train, model, patience=3)

# Plot Metrics and Evaluate the Model
# plot training loss and validation loss with matplotlib and pyplot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Create dataset on full raw data. First scale according to existing scaler.
df['raw_scaled'] = scaler.transform(df[['raw']])
X_raw, y_raw = LSTM_utilities.create_sequenced_dataset(df[['raw_scaled']], time_steps)
print(X_raw.shape)
print(y_raw.shape)

# Evaluate the model applied to the full dataset
# X_train_pred, train_mae_loss, model_eval, X_test_pred, test_mae_loss, predictions = LSTM_utilities.evaluate_model(X_train, X_raw, y_raw, model)
train_pred, train_mae_loss, model_eval, test_pred, test_mae_loss, predictions = LSTM_utilities.evaluate_vanilla_model(X_train, y_train, X_raw, y_raw, model)


# look at the distribution of the errors using a distribution plot
# could find a way to do a 95% percentile or some other actual value to automatically select the threshold.
# However, that is a set number of anomalies.
sns.distplot(train_mae_loss, bins=50, kde=True)
plt.show()
# choose a threshold to use for anomalies based on x-axis. try where error is greater than 0.75, it's anomalous.
threshold = [0.75]
sns.distplot(test_mae_loss, bins=50, kde=True)
plt.show()

# Transform predictions back to original units
predictions_unscaled = pd.DataFrame(scaler.inverse_transform(predictions))

# Detect anomalies
test_data = df[['raw_scaled']]
test_score_array = LSTM_utilities.detect_anomalies(test_data, predictions, predictions_unscaled, time_steps, test_mae_loss, threshold)


# Use events function to widen and number anomalous events
df_anomalies = df.iloc[time_steps:]
df_anomalies['labeled_event'] = anomaly_utilities.anomaly_events(df_anomalies['labeled_anomaly'])
df_anomalies['detected_anomaly'] = test_score_array[0]['anomaly']
df_anomalies['detected_event'] = anomaly_utilities.anomaly_events(df_anomalies['detected_anomaly'])

# Determine Metrics
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
plt.plot(test_score_array[0]['pred_unscaled'], 'c', label='predicted values')
plt.plot(df['raw'][df['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
plt.plot(test_score_array[0]['pred_unscaled'][test_score_array[0]['anomaly']], 'r+', label='machine detected anomalies')
plt.legend()
plt.ylabel(sensor)
plt.show()


print("\n LSTM script end.")
