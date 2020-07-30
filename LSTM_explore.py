
import numpy as np
import tensorflow as tf
import pandas as pd
pd.options.mode.chained_assignment = None
import seaborn as sns
from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
import talos
import plotly.io as pio
pio.renderers.default = "browser"

sns.set(style='whitegrid', palette='muted')
rcParams['figure.figsize'] = 14, 8
np.random.seed(1)
tf.random.set_seed(1)

print('Tensorflow version:', tf.__version__)

# Import Data

### DEFINE SITE and VARIABLE ###
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


def get_data(site, sensor, year, path=""):
    """Imports a single year of data based on files named by site, sensor/variable, and year.
    Labels data as anomalous. Generates a series from the data frame."""
    # TODO: make sensors input argument a list and output df with multiple normal_lbl columns.
    if path == "":
        path = os.getcwd()
    df_full = pd.read_csv(path + site + str(year) + ".csv",
                     skipinitialspace=True,
                     engine='python',
                     header=0,
                     index_col=0,
                     parse_dates=True,
                     infer_datetime_format=True)
    # makes one-dimensional data frame of booleans based on qualifier column indicating normal (TRUE) or not (FALSE)
    normal_lbl = df_full[sensor + "_qual"].isnull()
    # generate data frames and series from dataframe - time indexed values of raw and corrected data
    df_raw = df_full[[sensor]]
    df_cor = df_full[[sensor + "_cor"]]
    srs = pd.Series(df_full[sensor])

    return df_full, df_raw, df_cor, normal_lbl, srs


df_full, df_raw, df_cor, normal_lbl, srs = get_data(site, sensor, year, path="/Users/amber/PycharmProjects/LRO-anomaly-detection/LRO_Data/")
df_cor.head()
df_cor.shape

# Inspect data
# plotting with plotly to have an interactive chart
fig = go.Figure()
# fig.add_trace(go.Scatter(x=df.datetime, y=df.cond, #fig.add_trace adds different types of plots to the same figure.
fig.add_trace(go.Scatter(x=df_full.index, y=df_full.cond_cor,  # fig.add_trace adds different types of plots to the same figure.
                    mode='lines',
                    name='cond'))
fig.update_layout(showlegend=True)
fig.show()

# Remove -9999 values from corrected data in order to train detector
df_cor = df_cor.replace(-9999, np.NaN)


def train_test(df, ratio):
    """splits data fram into training and testing. uses straight temporal split."""
    # TODO: Add ability to do randomized train/test
    train_size = int(len(df) * ratio)
    test_size = len(df) - train_size
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

    return train, test


train, test = train_test(df_cor, 0.4)


def scale_data(train, test):
    """scales data using standard scaler. removes mean and scales to unit vectors."""
    scaler = StandardScaler()
    scaler = scaler.fit(train)
    train[train.columns[0]] = scaler.transform(train)
    test[test.columns[0]] = scaler.transform(test)

    return train, test, scaler


train, test, scaler = scale_data(train, test)


def create_dataset(X, y, time_steps=1):
    """Reshapes data to temporalize it into (samples, timestamps, features).
    Time stamps defines a sequence of how far back to consider for each sample/row.
    Features refers to the number of columns/variables."""
    Xs, ys = [], [] # start empty list
    for i in range(len(X) - time_steps):  # loop within range of data frame minus the time steps
        v = X.iloc[i:(i + time_steps)].values  # data from i to end of the time step
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])

    return np.array(Xs), np.array(ys)  # convert lists into numpy arrays and return

# create X and y train and test
# set time steps to 200(30)

time_steps = 30

X_train, y_train = create_dataset(train, train, time_steps)
X_test, y_test = create_dataset(test, test, time_steps)

print(X_train.shape)
print(y_train.shape)

## Task 5: Build an LSTM Autoencoder

# train autoencoder on data with no anomalies. take new point and reconstruct with autoencoder.
# if resconstruction error is above threshold, then label as anomaly. regression
# set time steps
timesteps = X_train.shape[1]
num_features = X_train.shape[2]



# use sequential model class from keras api - can provide definition as list of layers to the sequential data class.
# input to data - sample is one instance from the dataset (row). number of timesteps by number of features.
# output is the encoded feature vector of the input data
# can change hyperparams (number cells, dropout rate)


def create_model(cells, time_steps, num_features, dropout, input_loss='mae', input_optimizer='adam'):
    """Uses sequential model class from keras. Adds LSTM layer. Input samples, timesteps, features.
    Hyperparameters include number of cells, dropout rate. Output is encoded feature vector of the input data.
    Uses autoencoder by mirroring/reversing encoder to be a decoder."""
    model = Sequential([
        LSTM(cells, input_shape=(time_steps, num_features)),  # one LSTM layer
        Dropout(dropout),  # dropout regularization
        RepeatVector(time_steps),  # replicates the feature vectors from LSTM layer output vector by the number of time steps (e.g., 30 times)
        LSTM(cells, return_sequences=True),  # mirror the encoder in the reverse fashion to create the decoder
        Dropout(dropout),
        TimeDistributed(Dense(num_features))  # add time distributed layer to get output in correct shape.
        # creates a vector of length = num features output from previous layer.
        # (128) output, creates 128 vector and duplicates the number of features. (in this case, 1)
        ])
    print(model.optimizer)
    model.compile(loss=input_loss, optimizer=input_optimizer)

    return model


model = create_model(10, timesteps, num_features, 0.2)
model.summary()

## Task 6: Train the Autoencoder

# need to fit model to training data
# early stopping to ensure that too many epochs are not used.
# monitor the validation loss for improvements. If network doesn't improve for 3 consecutive epochs, stops training.


def train_model(X_train, y_train, patience, monitor='val_loss', mode='min', epochs=100, batch_size=32, validation_split=0.1):
    """Fits the model to training data. Early stopping ensures that too many epochs of training are not used.
    Monitors the validation loss for improvements.  """
    es = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, mode=mode)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,  # just set to something high, early stopping will monitor.
        batch_size=batch_size,  # this can be optimized later
        validation_split=validation_split,  # use 10% of data for validation, use 90% for training.
        callbacks=[es],  # early stopping similar to earlier
        shuffle=False   # because order matters
    )

    return history


history = train_model(X_train, y_train, patience=3)


# EXPERIMENT WITH TALOS FOR HYPERPARAMETER TUNING

def create_model_talos(params, time_steps, num_features, input_loss='mae', input_optimizer='adam',
                 patience=3, monitor='val_loss', mode='min', epochs=100, validation_split=0.1):
    """Uses sequential model class from keras. Adds LSTM layer. Input samples, timesteps, features.
    Hyperparameters include number of cells, dropout rate. Output is encoded feature vector of the input data.
    Uses autoencoder by mirroring/reversing encoder to be a decoder."""
    model = Sequential([
        LSTM(params['cells'], input_shape=(time_steps, num_features)),  # one LSTM layer
        Dropout(params['dropout']),  # dropout regularization
        RepeatVector(time_steps),  # replicates the feature vectors from LSTM layer output vector by the number of time steps (e.g., 30 times)
        LSTM(params['cells'], return_sequences=True),  # mirror the encoder in the reverse fashion to create the decoder
        Dropout(params['dropout']),
        TimeDistributed(Dense(num_features))  # add time distributed layer to get output in correct shape.
        # creates a vector of length = num features output from previous layer.
        # (128) output, creates 128 vector and duplicates the number of features. (in this case, 1)
        ])
    print(model.optimizer)
    model.compile(loss=input_loss, optimizer=input_optimizer)

    es = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, mode=mode)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,  # just set to something high, early stopping will monitor.
        batch_size=params['batch_size'],  # this can be optimized later
        validation_split=validation_split,  # use 10% of data for validation, use 90% for training.
        callbacks=[es],  # early stopping similar to earlier
        shuffle=False   # because order matters
    )

    return model, history


p = {'cells': [4, 8, 16, 32, 64, 128],
     'dropout': (0, 0.4, 10),
     'batch_size': [5, 10, 25, 50]}

scan_object = talos.Scan(X_train, y_train, params=p, model=create_model_talos, experiment_name='test')

scan_object.data.head()











## Task 7: Plot Metrics and Evaluate the Model

# plot training loss and validation loss with matplotlib and pyplot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# validation loss consistently lower than training loss. meaning underfit on training data- likely due to high dropout values that we used. could/should modify.


# still need to detect anomalies in our data.
# get model predictions on training data and determine mean absolute error
# look at the loss for each individual point

def evaluate_model(X_train, X_test):
    """Gets model predictions on training data and determines mean absolute error. Evaluates model on test data."""
    X_train_pred = model.predict(X_train)
    train_mae_loss = pd.DataFrame(np.mean(np.abs(X_train_pred-X_train), axis=1), columns=['Error'])
    model_eval = model.evaluate(X_test, y_test)
    X_test_pred = model.predict(X_test)
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

    return X_train_pred, train_mae_loss, model_eval, X_test_pred, test_mae_loss


X_train_pred, train_mae_loss, model_eval, X_test_pred, test_mae_loss = evaluate_model(X_train, X_test)

# look at the distribution of the errors using a distribution plot
sns.distplot(train_mae_loss, bins=50, kde=True)
plt.show()
# choose a threshold to use for anomalies based on x-axis. try where error is greater than 0.75, it's anomalous.
threshold = 5
sns.distplot(test_mae_loss, bins=50, kde=True)
plt.show()


def detect_anomalies(test, test_mae_loss, threshold):
    """Examine distribution of model errors to select threshold for anomalies.
    Add columns fro loss value, threshold, anomalous T/F.
    Creates data frame of anomalies to explore with more granularity."""
    test_score_df = pd.DataFrame(test[time_steps:])
    # add additional columns for loss value, threshold, whether entry is anomaly or not. could set a variable threshold.
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = threshold
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    anomalies = test_score_df[test_score_df.anomaly == True]

    return test_score_df, anomalies


test_score_df, anomalies = detect_anomalies(test, test_mae_loss, threshold)






fig = go.Figure()
fig.add_trace(go.Scatter(x=test[time_steps:].index, y=test_score_df.loss,
                    mode='lines',
                    name='Test Loss'))
fig.add_trace(go.Scatter(x=test[time_steps:].index, y=test_score_df.threshold, # add a line to indicate the threshold.
                    mode='lines',
                    name='Threshold'))
fig.update_layout(showlegend=True)
fig.show()


# look at anomalies in the test data
fig = go.Figure()
fig.add_trace(go.Scatter(x=test[time_steps:].index, y=scaler.inverse_transform(test[time_steps:][test.columns[0]]),
                    mode='lines',
                    name=sensor))
fig.add_trace(go.Scatter(x=anomalies.index, y=scaler.inverse_transform(anomalies[anomalies.columns[0]]),
                    mode='markers',
                    name='Anomaly'))
fig.update_layout(showlegend=True)
fig.show()


