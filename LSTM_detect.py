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


# PREPROCESSING FUNCTIONS #
################################
#
# def create_scaler(data):
#     """Creates a scaler object based on input data that removes mean and scales to unit vectors."""
#     scaler = StandardScaler()
#     scaler = scaler.fit(data)
#
#     return scaler
#
#
# def create_training_dataset(X, training_samples="", time_steps=10):
#     """Splits data into training and testing data based on random selection.
#     Reshapes data to temporalize it into (samples, timestamps, features).
#     - Samples is the number of rows/observations. Training_samples is the number of observations used for training.
#     - Time stamps defines a sequence of how far back to consider for each sample/row.
#     - Features refers to the number of columns/variables."""
#     Xs, ys = [], []  # start empty list
#     if training_samples == "":
#         training_samples = int(len(X) * 0.10)
#
#     # create sample sequences from a randomized subset of the data series for training
#     for i in range(training_samples):  # for every sample sequence to be created
#         j = randint(0, len(X) - time_steps - 2)
#         v = X.iloc[j:(j + time_steps)].values  # data from j to the end of time step
#         ys.append(X.iloc[j + time_steps])
#         Xs.append(v)
#
#     return np.array(Xs), np.array(ys)  # convert lists into numpy arrays and return
#
#
# def create_sequenced_dataset(X, time_steps=10):
#     """Reshapes data to temporalize it into (samples, timestamps, features).
#     Time stamps defines a sequence of how far back to consider for each sample/row.
#     Features refers to the number of columns/variables."""
#     Xs, ys = [], []  # start empty list
#     for i in range(len(X) - time_steps):  # loop within range of data frame minus the time steps
#         v = X.iloc[i:(i + time_steps)].values  # data from i to end of the time step
#         Xs.append(v)
#         ys.append(X.iloc[i + time_steps].values)
#
#     return np.array(Xs), np.array(ys)  # convert lists into numpy arrays and return
#
#
# def create_model(cells, time_steps, num_features, dropout, input_loss='mae', input_optimizer='adam'):
#     """Uses sequential model class from keras. Adds LSTM layer. Input samples, timesteps, features.
#     Hyperparameters include number of cells, dropout rate. Output is encoded feature vector of the input data.
#     Uses autoencoder by mirroring/reversing encoder to be a decoder."""
#     model = Sequential([
#         LSTM(cells, input_shape=(time_steps, num_features)),  # one LSTM layer
#         Dropout(dropout),  # dropout regularization
#         RepeatVector(time_steps),
#         # replicates the feature vectors from LSTM layer output vector by the number of time steps (e.g., 30 times)
#         LSTM(cells, return_sequences=True),  # mirror the encoder in the reverse fashion to create the decoder
#         Dropout(dropout),
#         TimeDistributed(Dense(num_features))  # add time distributed layer to get output in correct shape.
#         # creates a vector of length = num features output from previous layer.
#     ])
#     print(model.optimizer)
#     model.compile(loss=input_loss, optimizer=input_optimizer)
#
#     return model
#
#
# def train_model(X_train, y_train, patience, monitor='val_loss', mode='min', epochs=100, batch_size=32,
#                 validation_split=0.1):
#     """Fits the model to training data. Early stopping ensures that too many epochs of training are not used.
#     Monitors the validation loss for improvements and stops training when improvement stops."""
#     es = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, mode=mode)
#     history = model.fit(
#         X_train, y_train,
#         epochs=epochs,  # just set to something high, early stopping will monitor.
#         batch_size=batch_size,  # this can be optimized later
#         validation_split=validation_split,  # use 10% of data for validation, use 90% for training.
#         callbacks=[es],  # early stopping similar to earlier
#         shuffle=False  # because order matters
#     )
#
#     return history
#
#
# def evaluate_model(X_train, X_test, y_test):
#     """Gets model predictions on training data and test data.
#     Determines mean absolute error to evaluate model on training and test data."""
#     X_train_pred = model.predict(X_train)
#     train_mae_loss = pd.DataFrame(np.mean(np.abs(X_train_pred - X_train), axis=1), columns=['Error'])
#     model_eval = model.evaluate(X_test, y_test)
#
#     X_test_pred = model.predict(X_test)
#     predictions = pd.DataFrame(X_test_pred[:, 0])
#     test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
#
#     return X_train_pred, train_mae_loss, model_eval, X_test_pred, test_mae_loss, predictions
#
#
# def detect_anomalies(test, predictions, test_mae_loss, threshold):
#     """Examine distribution of model errors to select threshold for anomalies.
#     Add columns for loss value, threshold, anomalous T/F.
#     Creates data frame of anomalies to explore with more granularity."""
#     test_score_df = pd.DataFrame(test[time_steps:])
#     # add additional columns for loss value, threshold, whether entry is anomaly or not. could set a variable threshold.
#     test_score_df['prediction'] = np.array(predictions)
#     test_score_df['loss'] = test_mae_loss
#     test_score_df['threshold'] = threshold
#     test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
#     anomalies = test_score_df[test_score_df.anomaly == True]
#
#     return test_score_df, anomalies
#

#########################################
# IMPLEMENTATION AND FUNCTION EXECUTION #
#########################################

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
time_steps = 10
samples = 5000
X_train, y_train = LSTM_utilities.create_training_dataset(df_sub[['cor_scaled']], samples, time_steps)
print(X_train.shape)
print(y_train.shape)

# Create and model and train to data
num_features = X_train.shape[2]
model = LSTM_utilities.create_model(128, time_steps, num_features, 0.2)
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
X_raw, y_raw = LSTM_utilities.create_sequenced_dataset(df[['raw_scaled']], 10)

# Evaluate the model
X_train_pred, train_mae_loss, model_eval, X_test_pred, test_mae_loss, predictions = LSTM_utilities.evaluate_model(X_train, X_raw, y_raw)


# look at the distribution of the errors using a distribution plot
# could find a way to do a 95% percentile or some other actual value to automatically select the threshold.
# However, that is a set number of anomalies.
sns.distplot(train_mae_loss, bins=50, kde=True)
plt.show()
# choose a threshold to use for anomalies based on x-axis. try where error is greater than 0.75, it's anomalous.
threshold = 0.75
sns.distplot(test_mae_loss, bins=50, kde=True)
plt.show()

# Transform predictrions back to original units
predictions_unscaled = scaler.inverse_transform(predictions)

# Detect anomalies
test_score_array = LSTM_utilities.detect_anomalies(df[['raw_scaled']], predictions, predictions_unscaled, time_steps, test_mae_loss, threshold)





# Use events function to widen and number anomalous events
df_cut = df.iloc[time_steps:]
df_cut['labeled_event'] = anomaly_utilities.anomaly_events(df_cut['labeled_anomaly'])
df_cut['detected_anomaly'] = test_score_df['anomaly']
df_cut['detected_event'] = anomaly_utilities.anomaly_events(df_cut['detected_anomaly'])

# Determine Metrics
labeled_in_detected, detected_in_labeled, valid_detections, invalid_detections = anomaly_utilities.compare_labeled_detected(df_cut)
TruePositives, FalseNegatives, FalsePositives, TrueNegatives, PRC, PPV, NPV, ACC, RCL, f1, f2 \
    = anomaly_utilities.metrics(df_cut, valid_detections, invalid_detections)


# OUTPUT RESULTS #
#########################################
print('\n\n\nScript report:\n')
print('Sensor: ' + sensor)
print('Year: ' + str(year))
print('Parameters: LSTM, sequence length: %i, training samples: %i, Threshold = %f' %(time_steps, samples, threshold))
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


# GENERATE PLOTS #
#########################################

plt.figure()
plt.plot(df['raw'], 'b', label='original data')
plt.plot(test_score_df['pred'], 'c', label='predicted values')
plt.plot(df['raw'][df['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
plt.plot(test_score_df['pred'][test_score_df['anomaly']], 'r+', label='machine detected anomalies')
plt.legend()
plt.ylabel(sensor)
plt.show()


# PLOTTING WITH PLOTLY - allows for interactive plots with zoom, pan, etc.
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=test[time_steps:].index, y=test_score_df.loss,
#                     mode='lines',
#                     name='Test Loss'))
# fig.add_trace(go.Scatter(x=test[time_steps:].index, y=test_score_df.threshold, # add a line to indicate the threshold.
#                     mode='lines',
#                     name='Threshold'))
# fig.update_layout(showlegend=True)
# fig.show()
#
#
# # look at anomalies in the test data
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=test[time_steps:].index, y=scaler.inverse_transform(test[time_steps:][test.columns[0]]),
#                     mode='lines',
#                     name=sensor))
# fig.add_trace(go.Scatter(x=anomalies.index, y=scaler.inverse_transform(anomalies[anomalies.columns[0]]),
#                     mode='markers',
#                     name='Anomaly'))
# fig.update_layout(showlegend=True)
# fig.show()
#
# # Inspect data
# # plotting with plotly to have an interactive chart
# fig = go.Figure()
# # fig.add_trace(go.Scatter(x=df.datetime, y=df.cond, #fig.add_trace adds different types of plots to the same figure.
# fig.add_trace(go.Scatter(x=df_full.index, y=df_full.cond_cor,  # fig.add_trace adds different types of plots to the same figure.
#                     mode='lines',
#                     name='cond'))
# fig.update_layout(showlegend=True)
# fig.show()


