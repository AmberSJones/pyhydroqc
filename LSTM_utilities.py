################################
# LSTM UTILITIES #
################################
# This code includes utilities for running LSTM models for anomaly detection.
# A scaling function is defined.
# A function creates a training dataset based on random selection of the complete dataset,
#   which also temporalizes data to prepare for LSTM.
# Another function creates sequenced/temporalized data for LSTM.
# Separate functions create and train the model.
# A function is defined for evaluating the model.
# Another function detects anomalies.

from random import randint, sample
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed, Bidirectional

class LSTM_modelContainer:
    pass


def vanilla_LSTM_model(df, time_steps, samples, cells, dropout, patience):
    """df needs to have column det_cor and anomaly"""

    scaler = create_scaler(df[['det_cor']])
    df['det_scaled'] = scaler.transform(df[['det_cor']])

    #X_train, y_train = create_training_dataset(df[['det_scaled']], samples, time_steps)
    X_train, y_train = create_clean_training_dataset(df[['det_scaled']], df[['anomaly']], samples, time_steps)
    num_features = X_train.shape[2]

    print(X_train.shape)
    print(y_train.shape)
    print(num_features)

    model = create_vanilla_model(cells, time_steps, num_features, dropout)
    model.summary()
    history = train_model(X_train, y_train, model, patience)

    df['raw_scaled'] = scaler.transform(df[['raw']])
    X_test, y_test = create_sequenced_dataset(df[['raw_scaled']], time_steps)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    model_eval = model.evaluate(X_test, y_test)

    train_predictions = pd.DataFrame(scaler.inverse_transform(train_pred))
    predictions = pd.DataFrame(scaler.inverse_transform(test_pred))
    y_train_unscaled = pd.DataFrame(scaler.inverse_transform(y_train))
    y_test_unscaled = pd.DataFrame(scaler.inverse_transform(y_test))

    train_residuals = pd.DataFrame(np.abs(train_predictions - y_train_unscaled))
    test_residuals = pd.DataFrame(np.abs(predictions - y_test_unscaled))

    return X_train, y_train, model, history, X_test, y_test, model_eval, predictions, train_residuals, test_residuals


def multi_vanilla_LSTM_model(df_det_cor, df_anomaly, df_raw, time_steps, samples, cells, dropout, patience):
    """df needs to have column det_cor and anomaly"""

    scaler = create_scaler(df_det_cor)
    df_scaled = pd.DataFrame(scaler.transform(df_det_cor), index=df_det_cor.index, columns=df_det_cor.columns)

    X_train, y_train = create_clean_training_dataset(df_scaled, df_anomaly, samples, time_steps)
    num_features = X_train.shape[2]

    print(X_train.shape)
    print(y_train.shape)
    print(num_features)

    model = create_vanilla_model(cells, time_steps, num_features, dropout)
    model.summary()
    history = train_model(X_train, y_train, model, patience)

    df_raw_scaled = pd.DataFrame(scaler.transform(df_raw), index=df_raw.index, columns=df_raw.columns)
    X_test, y_test = create_sequenced_dataset(df_raw_scaled, time_steps)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    model_eval = model.evaluate(X_test, y_test)

    train_predictions = pd.DataFrame(scaler.inverse_transform(train_pred))
    predictions = pd.DataFrame(scaler.inverse_transform(test_pred))
    y_train_unscaled = pd.DataFrame(scaler.inverse_transform(y_train))
    y_test_unscaled = pd.DataFrame(scaler.inverse_transform(y_test))

    train_residuals = pd.DataFrame(np.abs(train_predictions - y_train_unscaled))
    test_residuals = pd.DataFrame(np.abs(predictions - y_test_unscaled))

    return X_train, y_train, model, history, X_test, y_test, model_eval, predictions, train_residuals, test_residuals



def create_scaler(data):
    """Creates a scaler object based on input data that removes mean and scales to unit vectors."""
    scaler = StandardScaler()
    scaler = scaler.fit(data)

    return scaler


def create_training_dataset(X, training_samples="", time_steps=10):
    """Splits data into training and testing data based on random selection.
    Reshapes data to temporalize it into (samples, timestamps, features).
    - Samples is the number of rows/observations. Training_samples is the number of observations used for training.
    - Time stamps defines a sequence of how far back to consider for each sample/row.
    - Features refers to the number of columns/variables."""
    Xs, ys = [], []  # start empty list
    if training_samples == "":
        training_samples = int(len(X) * 0.10)

    # create sample sequences from a randomized subset of the data series for training
    j = sample(range(0, len(X) - time_steps - 2), training_samples)
    for i in range(training_samples):  # for every sample sequence to be created
        #j = randint(0, len(X) - time_steps - 2)
        #v = X.iloc[j:(j + time_steps)].values  # data from j to the end of time step
        v = X.iloc[j[i]:(j[i] + time_steps)].values  # data from j to the end of time step
        ys.append(X.iloc[j[i] + time_steps])
        Xs.append(v)

    return np.array(Xs), np.array(ys)  # convert lists into numpy arrays and return


def create_clean_training_dataset(X, anomalies, training_samples="", time_steps=10):
    """Splits data into training and testing data based on random selection.
    Reshapes data to temporalize it into (samples, timestamps, features).
    - Samples is the number of rows/observations. Training_samples is the number of observations used for training.
    - Time stamps defines a sequence of how far back to consider for each sample/row.
    - Features refers to the number of columns/variables."""

    Xs, ys = [], []  # start empty list
    if training_samples == "":
        training_samples = int(len(X) * 0.10)

    # create sample sequences from a randomized subset of the data series for training
    j = sample(range(0, len(X) - time_steps - 2), len(X) - time_steps - 2)
    i = 0
    while (training_samples > len(ys)) and (i < len(j)):
        if not np.any(anomalies.iloc[j[i]:(j[i] + time_steps + 1)]):
            v = X.iloc[j[i]:(j[i] + time_steps)].values
            ys.append(X.iloc[j[i] + time_steps])
            Xs.append(v)
        i += 1

    return np.array(Xs), np.array(ys)  # convert lists into numpy arrays and return


def create_sequenced_dataset(X, time_steps=10):
    """Reshapes data to temporalize it into (samples, timestamps, features).
    Time stamps defines a sequence of how far back to consider for each sample/row.
    Features refers to the number of columns/variables."""
    Xs, ys = [], []  # start empty list
    for i in range(len(X) - time_steps):  # loop within range of data frame minus the time steps
        v = X.iloc[i:(i + time_steps)].values  # data from i to end of the time step
        Xs.append(v)
        ys.append(X.iloc[i + time_steps].values)

    return np.array(Xs), np.array(ys)  # convert lists into numpy arrays and return


def create_bidir_training_dataset(X, training_samples="", time_steps=10):
    """Splits data into training and testing data based on random selection.
    Reshapes data to temporalize it into (samples, timestamps, features).
    - Samples is the number of rows/observations. Training_samples is the number of observations used for training.
    - Time stamps defines a sequence of how far back to consider for each sample/row.
    - Features refers to the number of columns/variables."""
    Xs, ys = [], []  # start empty list
    if training_samples == "":
        training_samples = int(len(X) * 0.10)

    # create sample sequences from a randomized subset of the data series for training
    for i in range(training_samples):  # for every sample sequence to be created
        j = randint(time_steps, len(X) - time_steps)
        v = X.iloc[(j - time_steps):(j + time_steps)].values  # data from j backward and forward the specified number of time steps
        Xs.append(v)
        ys.append(X.iloc[j].values)

    return np.array(Xs).astype(np.float32), np.array(ys)  # convert lists into numpy arrays and return


def create_bidir_sequenced_dataset(X, time_steps=10):
    """Reshapes data to temporalize it into (samples, timestamps, features).
    Time stamps defines a sequence of how far back to consider for each sample/row.
    Features refers to the number of columns/variables."""
    Xs, ys = [], []  # start empty list
    for i in range(time_steps, len(X) - time_steps):  # loop within range of data frame minus the time steps
        v = X.iloc[(i - time_steps):(i + time_steps)].values  # data from i backward and forward the specified number of time steps
        Xs.append(v)
        ys.append(X.iloc[i].values)

    return np.array(Xs).astype(np.float32), np.array(ys)  # convert lists into numpy arrays and return


def create_model(cells, time_steps, num_features, dropout, input_loss='mae', input_optimizer='adam'):
    """Uses sequential model class from keras. Adds LSTM layer. Input samples, timesteps, features.
    Hyperparameters include number of cells, dropout rate. Output is encoded feature vector of the input data.
    Uses autoencoder by mirroring/reversing encoder to be a decoder."""
    model = Sequential()
    model.add(LSTM(cells, input_shape=(time_steps, num_features), return_sequences=False, dropout=dropout)),  # one LSTM layer with dropout regularization
    model.add(RepeatVector(time_steps))  # replicates the feature vectors from LSTM layer output vector by the number of time steps (e.g., 30 times)
    model.add(LSTM(cells, return_sequences=True, dropout=dropout))   # mirror the encoder in the reverse fashion to create the decoder
    model.add(TimeDistributed(Dense(num_features)))  # add time distributed layer to get output in correct shape. creates a vector of length = num features output from previous layer.
    model.compile(loss=input_loss, optimizer=input_optimizer)

    return model


def create_vanilla_model(cells, time_steps, num_features, dropout, input_loss='mae', input_optimizer='adam'):
    """Uses sequential model class from keras. Adds LSTM layer. Input samples, timesteps, features.
    Hyperparameters include number of cells, dropout rate. Output is encoded feature vector of the input data.
    Uses vanilla LSTM - not autoencoder."""
    model = Sequential()
    model.add(LSTM(cells, input_shape=(time_steps, num_features), dropout=dropout)),  # one LSTM layer with dropout regularization
    model.add(Dense(num_features))
    model.compile(loss=input_loss, optimizer=input_optimizer)

    return model


def create_bidir_model(cells, time_steps, num_features, dropout, input_loss='mae', input_optimizer='adam'):
    """Uses sequential model class from keras. Adds Bidirectional LSTM layer. Input is samples, timesteps, features.
    Hyperparameters include number of cells, dropout rate. Output is encoded feature vector of the input data.
    Uses bidirectional LSTM."""
    model = Sequential()
    model.add(Bidirectional(LSTM(cells, dropout=dropout), input_shape=(time_steps*2, num_features)))
    model.add(Dense(num_features))
    model.compile(loss=input_loss, optimizer=input_optimizer)

    return model


def train_model(X_train, y_train, model, patience, monitor='val_loss', mode='min', epochs=100, batch_size=32,
                validation_split=0.1):
    """Fits the model to training data. Early stopping ensures that too many epochs of training are not used.
    Monitors the validation loss for improvements and stops training when improvement stops."""
    es = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, mode=mode)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,  # just set to something high, early stopping will monitor.
        batch_size=batch_size,  # this can be optimized later
        validation_split=validation_split,  # use 10% of data for validation, use 90% for training.
        callbacks=[es],  # early stopping similar to earlier
        shuffle=False  # because order matters
    )

    return history


def detect_anomalies(test, predictions, unscaled_predictions, time_steps, test_residuals, threshold):
    """Create array of data frames for each variable.
    Add columns for raw data, model prediction, threshold, anomalous T/F, and unscaled prediction.
    test is a data frame of raw scaled data. predictions is the model predictions from the evaluate_model function.
    unscaled_predictions are the model prediction inverse scaled to original units.
    time_steps is the number of time steps used to predict.
    test_mae_loss is the model error from the evaluate_model function.
    threshold is a list of thresholds for detecting anomalies from the model errors. length should = number of variables."""
    # create array to store results for all variables
    test_score_array = []
    for i in range(0, test.shape[1]):
        test_score_df = []
        test_score_df = pd.DataFrame(test[test.columns[i]])
        test_score_df = test_score_df[time_steps:]
        # add additional columns for loss value, threshold, whether entry is anomaly or not. could set a variable threshold.
        test_score_df['prediction'] = np.array(predictions[predictions.columns[i]])
        test_score_df['loss'] = np.array(test_residuals[test_residuals.columns[i]])
        test_score_df['threshold'] = threshold[i]
        test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
        test_score_df['pred_unscaled'] = np.array(unscaled_predictions[unscaled_predictions.columns[i]])
        # anomalies = test_score_df[test_score_df.anomaly == True]
        test_score_array.append(test_score_df)

    return test_score_array

def detect_anomalies_bidir(test, predictions, unscaled_predictions, time_steps, test_residuals, threshold):
    """Create array of data frames for each variable.
    Add columns for raw data, model prediction, threshold, anomalous T/F, and unscaled prediction.
    test is a data frame of raw scaled data. predictions is the model predictions from the evaluate_model function.
    unscaled_predictions are the model prediction inverse scaled to original units.
    time_steps is the number of time steps used to predict.
    test_mae_loss is the model error from the evaluate_model function.
    threshold is a list of thresholds for detecting anomalies from the model errors. length should = number of variables."""
    # create array to store results for all variables
    test_score_array = []
    for i in range(0, test.shape[1]):
        test_score_df = []
        test_score_df = pd.DataFrame(test[test.columns[i]])
        test_score_df = test_score_df[time_steps:-time_steps]
        # add additional columns for loss value, threshold, whether entry is anomaly or not. could set a variable threshold.
        test_score_df['prediction'] = np.array(predictions[predictions.columns[i]])
        test_score_df['loss'] = np.array(test_residuals[test_residuals.columns[i]])
        test_score_df['threshold'] = threshold[i]
        test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
        test_score_df['pred_unscaled'] = np.array(unscaled_predictions[unscaled_predictions.columns[i]])
        # anomalies = test_score_df[test_score_df.anomaly == True]
        test_score_array.append(test_score_df)

    return test_score_array

