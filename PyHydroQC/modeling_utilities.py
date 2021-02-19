################################
# MODELING UTILITIES #
################################
# This code includes utilities for developing models for anomaly detection.
# Currently, functions are defined for LSTM and ARIMA models.
# Wrapper functions for each type of LSTM model call other functions to build, train, and evaluate the models.
# Other functions are for scaling data and sequencing/temporalizing for LSTM.
# A function creates a training dataset based on random selection of the complete dataset,
#   which also temporalizes data to prepare for LSTM.

from random import sample
import numpy as np
import tensorflow as tf
import pandas as pd
import statsmodels.api as api
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
import warnings


def build_arima_model(data, p, d, q, summary):
    """
    build_arima_model constructs and trains an ARIMA model.
    data is a series or data frame of time series inputs.
    p, d, q are the ARIMA hyperparameters that can be determined by manual assessment or by automated means.
    summary indicates if the model summary should be printed.
    Outputs:
    model_fit is the SARIMAX model object.
    residuals are the model errors.
    predictions are the in sample, one step ahead model forecasted values.
    """

    warnings.filterwarnings('ignore', message='A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting')
    model = api.tsa.SARIMAX(data, order=(p, d, q))
    warnings.filterwarnings('ignore', message='Non-stationary starting autoregressive parameters')
    warnings.filterwarnings('ignore', message='Non-invertible starting MA parameters found.')
    warnings.filterwarnings('ignore', message='ConvergenceWarning: Maximum Likelihood optimization failed to converge.')
    model_fit = model.fit(disp=0, warn_convergence=False)
    warnings.filterwarnings('default')
    residuals = pd.DataFrame(model_fit.resid)
    predict = model_fit.get_prediction()
    predictions = pd.DataFrame(predict.predicted_mean)
    residuals[0][0] = 0
    predictions.predicted_mean[0] = data[0]

    # output summary
    if summary:
        print('\n\n')
        print(model_fit.summary())
        print('\n\nresiduals description:')
        print(residuals.describe())

    return model_fit, residuals, predictions


class LSTMModelContainer:
    pass
    """
    Objects of the class LSTM_modelContainer are wrappers that call functions to build, train, and evaluate LSTM models.
    All have the following input/output.
    For univariate: df is a data frame with columns:
        'raw' observed, uncorrected data
        'observed' observed data, corrected with preprocessing
        'anomaly' a boolean where True (1) = anomalous data point corresponding to the results of preprocessing.
    For multivariate: 
        df_raw is a data frame of uncorrected raw observed data.
        df_observed is a data frame containing preprocessed observed data with one column for each variable.
        df_anomaly is a data frame of booleans where True (1) = anomalous data point corresponding 
            to the results of preprocessing with one column for each variable.
    time_steps is the number of past data points for LSTM to consider.
    cells is the number of cells for the LSTM model.
    dropout is the rate of cells to ignore for model training.
    patience indicates how long to wait for model training.  
    
    Outputs:
    X_train is the reshaped array of input data used to train the model.
    y_train is the array of output data used to train the model.
    model is the keras model object.
    history is the results of model training.
    X_test is the reshaped array of input data used to test the model. For this work, we use the full dataset.
    y_test is the array of outpus data used to test the model. For this work, we use the full dataset.
    model_eval is an evaluation of the model with test data.
    predictions is the model predictions for the full dataset.
    train_residuals is the residuals of the data used for training.
    test_residuals is the residuals of the data used for testing.
    """


def LSTM_univar(df, LSTM_params, summary, name, model_output=True, model_save=True):
    """
    LSTM_univar builds, trains, and evaluates a vanilla LSTM model for univariate data.
    """
    scaler = create_scaler(df[['observed']])
    df['obs_scaled'] = scaler.transform(df[['observed']])

    X_train, y_train = create_training_dataset(df[['obs_scaled']], df[['anomaly']], LSTM_params['samples'], LSTM_params['time_steps'])
    num_features = X_train.shape[2]

    if summary:
        print('X_train shape: ' + str(X_train.shape))
        print('y_train shape: ' + str(y_train.shape))
        print('Number of features: ' + str(num_features))

    model = create_vanilla_model(LSTM_params['cells'], LSTM_params['time_steps'], num_features, LSTM_params['dropout'])
    if summary:
        model.summary()
        verbose = 1
    else:
        verbose = 0
    history = train_model(X_train, y_train, model, LSTM_params['patience'], verbose=verbose)

    df['raw_scaled'] = scaler.transform(df[['raw']])
    X_test, y_test = create_sequenced_dataset(df[['raw_scaled']], LSTM_params['time_steps'])

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    model_eval = model.evaluate(X_test, y_test)

    train_predictions = pd.DataFrame(scaler.inverse_transform(train_pred))
    predictions = pd.DataFrame(scaler.inverse_transform(test_pred))
    y_train_unscaled = pd.DataFrame(scaler.inverse_transform(y_train))
    y_test_unscaled = pd.DataFrame(scaler.inverse_transform(y_test))

    train_residuals = pd.DataFrame(np.abs(train_predictions - y_train_unscaled))
    test_residuals = pd.DataFrame(np.abs(predictions - y_test_unscaled))

    if model_save:
        model.save('originalsavedoutput/models/LSTM_univar_' + str(name))

    LSTM_univar = LSTMModelContainer()
    if model_output:
        LSTM_univar.model = model
        LSTM_univar.history = history
    LSTM_univar.X_train = X_train
    LSTM_univar.y_train = y_train
    LSTM_univar.X_test = X_test
    LSTM_univar.y_test = y_test
    LSTM_univar.model_eval = model_eval
    LSTM_univar.predictions = predictions
    LSTM_univar.train_residuals = train_residuals
    LSTM_univar.test_residuals = test_residuals

    return LSTM_univar


def LSTM_multivar(df_observed, df_anomaly, df_raw, LSTM_params, summary, name, model_output=True, model_save=True):
    """
    LSTM_multivar builds, trains, and evaluates a vanilla LSTM model for multivariate data.
    """
    scaler = create_scaler(df_observed)
    df_scaled = pd.DataFrame(scaler.transform(df_observed), index=df_observed.index, columns=df_observed.columns)

    X_train, y_train = create_training_dataset(df_scaled, df_anomaly, LSTM_params['samples'], LSTM_params['time_steps'])
    num_features = X_train.shape[2]

    if summary:
        print('X_train shape: ' + str(X_train.shape))
        print('y_train shape: ' + str(y_train.shape))
        print('Number of features: ' + str(num_features))

    model = create_vanilla_model(LSTM_params['cells'], LSTM_params['time_steps'], num_features, LSTM_params['dropout'])
    if summary:
        model.summary()
        verbose = 1
    else:
        verbose = 0
    history = train_model(X_train, y_train, model, LSTM_params['patience'], verbose=verbose)

    df_raw_scaled = pd.DataFrame(scaler.transform(df_raw), index=df_raw.index, columns=df_raw.columns)
    X_test, y_test = create_sequenced_dataset(df_raw_scaled, LSTM_params['time_steps'])

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    model_eval = model.evaluate(X_test, y_test)

    train_predictions = pd.DataFrame(scaler.inverse_transform(train_pred))
    predictions = pd.DataFrame(scaler.inverse_transform(test_pred))
    y_train_unscaled = pd.DataFrame(scaler.inverse_transform(y_train))
    y_test_unscaled = pd.DataFrame(scaler.inverse_transform(y_test))

    train_residuals = pd.DataFrame(np.abs(train_predictions - y_train_unscaled))
    test_residuals = pd.DataFrame(np.abs(predictions - y_test_unscaled))

    if model_save:
        model.save('originalsavedoutput/models/LSTM_multivar_' + str(name))

    LSTM_multivar = LSTMModelContainer()
    if model_output:
        LSTM_multivar.model = model
        LSTM_multivar.history = history
    LSTM_multivar.X_train = X_train
    LSTM_multivar.y_train = y_train
    LSTM_multivar.X_test = X_test
    LSTM_multivar.y_test = y_test
    LSTM_multivar.model_eval = model_eval
    LSTM_multivar.predictions = predictions
    LSTM_multivar.train_residuals = train_residuals
    LSTM_multivar.test_residuals = test_residuals

    return LSTM_multivar


def LSTM_univar_bidir(df, LSTM_params, summary, name, model_output=True, model_save=True):
    """
    LSTM_univar_bidir builds, trains, and evaluates a bidirectional LSTM model for univariate data.
    """
    scaler = create_scaler(df[['observed']])
    df['obs_scaled'] = scaler.transform(df[['observed']])

    X_train, y_train = create_bidir_training_dataset(df[['obs_scaled']], df[['anomaly']], LSTM_params['samples'], LSTM_params['time_steps'])

    num_features = X_train.shape[2]

    if summary:
        print('X_train shape: ' + str(X_train.shape))
        print('y_train shape: ' + str(y_train.shape))
        print('Number of features: ' + str(num_features))

    model = create_bidir_model(LSTM_params['cells'], LSTM_params['time_steps'], num_features, LSTM_params['dropout'])
    if summary:
        model.summary()
        verbose = 1
    else:
        verbose = 0
    history = train_model(X_train, y_train, model, LSTM_params['patience'], verbose=verbose)

    df['raw_scaled'] = scaler.transform(df[['raw']])
    X_test, y_test = create_bidir_sequenced_dataset(df[['raw_scaled']], LSTM_params['time_steps'])

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    model_eval = model.evaluate(X_test, y_test)

    train_predictions = pd.DataFrame(scaler.inverse_transform(train_pred))
    predictions = pd.DataFrame(scaler.inverse_transform(test_pred))
    y_train_unscaled = pd.DataFrame(scaler.inverse_transform(y_train))
    y_test_unscaled = pd.DataFrame(scaler.inverse_transform(y_test))

    train_residuals = pd.DataFrame(np.abs(train_predictions - y_train_unscaled))
    test_residuals = pd.DataFrame(np.abs(predictions - y_test_unscaled))

    if model_save:
        model.save('originalsavedoutput/models/LSTM_univar_bidir_' + str(name))

    LSTM_univar_bidir = LSTMModelContainer()
    if model_output:
        LSTM_univar_bidir.model = model
        LSTM_univar_bidir.history = history
    LSTM_univar_bidir.X_train = X_train
    LSTM_univar_bidir.y_train = y_train
    LSTM_univar_bidir.X_test = X_test
    LSTM_univar_bidir.y_test = y_test
    LSTM_univar_bidir.model_eval = model_eval
    LSTM_univar_bidir.predictions = predictions
    LSTM_univar_bidir.train_residuals = train_residuals
    LSTM_univar_bidir.test_residuals = test_residuals

    return LSTM_univar_bidir


def LSTM_multivar_bidir(df_observed, df_anomaly, df_raw, LSTM_params, summary, name, model_output=True, model_save=True):
    """
    LSTM_multivar_bidir builds, trains, and evaluates a bidirectional LSTM model for multivariate data.
    """
    scaler = create_scaler(df_observed)
    df_scaled = pd.DataFrame(scaler.transform(df_observed), index=df_observed.index, columns=df_observed.columns)

    X_train, y_train = create_bidir_training_dataset(df_scaled, df_anomaly, LSTM_params['samples'], LSTM_params['time_steps'])
    num_features = X_train.shape[2]

    if summary:
        print('X_train shape: ' + str(X_train.shape))
        print('y_train shape: ' + str(y_train.shape))
        print('Number of features: ' + str(num_features))

    model = create_bidir_model(LSTM_params['cells'], LSTM_params['time_steps'], num_features, LSTM_params['dropout'])
    if summary:
        model.summary()
        verbose = 1
    else:
        verbose = 0
    history = train_model(X_train, y_train, model, LSTM_params['patience'], verbose=verbose)

    df_raw_scaled = pd.DataFrame(scaler.transform(df_raw), index=df_raw.index, columns=df_raw.columns)
    X_test, y_test = create_bidir_sequenced_dataset(df_raw_scaled, LSTM_params['time_steps'])

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    model_eval = model.evaluate(X_test, y_test)

    train_predictions = pd.DataFrame(scaler.inverse_transform(train_pred))
    predictions = pd.DataFrame(scaler.inverse_transform(test_pred))
    y_train_unscaled = pd.DataFrame(scaler.inverse_transform(y_train))
    y_test_unscaled = pd.DataFrame(scaler.inverse_transform(y_test))

    train_residuals = pd.DataFrame(np.abs(train_predictions - y_train_unscaled))
    test_residuals = pd.DataFrame(np.abs(predictions - y_test_unscaled))

    if model_save:
        model.save('originalsavedoutput/models/LSTM_multiivar_bidir_' + str(name))

    LSTM_multivar_bidir = LSTMModelContainer()
    if model_output:
        LSTM_univar_bidir.model = model
        LSTM_univar_bidir.history = history
    LSTM_multivar_bidir.X_train = X_train
    LSTM_multivar_bidir.y_train = y_train
    LSTM_multivar_bidir.X_test = X_test
    LSTM_multivar_bidir.y_test = y_test
    LSTM_multivar_bidir.model_eval = model_eval
    LSTM_multivar_bidir.predictions = predictions
    LSTM_multivar_bidir.train_residuals = train_residuals
    LSTM_multivar_bidir.test_residuals = test_residuals

    return LSTM_multivar_bidir


def create_scaler(data):
    """
    create_scaler creates a scaler object based on input data that removes mean and scales to unit vectors.
    """
    scaler = StandardScaler()
    scaler = scaler.fit(data)

    return scaler


def create_training_dataset(X, anomalies, training_samples="", time_steps=10):
    """
    create_training_dataset creates a training dataset based on random selection.
    Reshapes data to temporalize it into (samples, timestamps, features). Ensures that no data that has been corrected
    as part of preprocessing will be used for training the model.
    X is the data to be reshaped.
    anomalies is a booleans where True (1) = anomalous data point corresponding to the results of preprocessing.
    training_samples is the number of observations used for training.
    time_stamps defines a sequence of how far back to consider for each sample/row.
    Outputs:
    Xs is an array of data reshaped for input into an LSTM model.
    ys is an array of data outputs corresponding to each Xs input.
    """
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
    """
    create_sequenced_dataset reshapes data to temporalize it into (samples, timestamps, features).
    X is the data to be reshaped.
    time_stamps defines a sequence of how far back to consider for each sample/row.
    Outputs:
    Xs is an array of data reshaped for input into an LSTM model.
    ys is an array of data outputs corresponding to each Xs input.
    """
    Xs, ys = [], []  # start empty list
    for i in range(len(X) - time_steps):  # loop within range of data frame minus the time steps
        v = X.iloc[i:(i + time_steps)].values  # data from i to end of the time step
        Xs.append(v)
        ys.append(X.iloc[i + time_steps].values)

    return np.array(Xs), np.array(ys)  # convert lists into numpy arrays and return


def create_bidir_training_dataset(X, anomalies, training_samples="", time_steps=10):
    """
    create_bidir_training_dataset creates a training dataset based on random selection.
    Reshapes data to temporalize it into (samples, timestamps, features). Ensures that no data that has been corrected
    as part of preprocessing will be used for training the model.
    X is the data to be reshaped.
    anomalies is a booleans where True (1) = anomalous data point corresponding to the results of preprocessing.
    training_samples is the number of observations used for training.
    time_stamps defines a sequence of how far backward and forward to consider for each sample/row.
    Outputs:
    Xs is an array of data reshaped for input into an LSTM model.
    ys is an array of data outputs corresponding to each Xs input.
    """
    Xs, ys = [], []  # start empty list
    if training_samples == "":
        training_samples = int(len(X) * 0.10)

    # create sample sequences from a randomized subset of the data series for training
    j = sample(range(time_steps, len(X) - time_steps), len(X) - 2 * time_steps)
    i = 0
    while (training_samples > len(ys)) and (i < len(j)):
        if not np.any(anomalies.iloc[(j[i] - time_steps):(j[i] + time_steps + 1)]):
            v = pd.concat([X.iloc[(j[i] - time_steps):j[i]], X.iloc[(j[i] + 1):(j[i] + time_steps + 1)]]).values
            ys.append(X.iloc[j[i]])
            Xs.append(v)
        i += 1

    return np.array(Xs).astype(np.float32), np.array(ys)  # convert lists into numpy arrays and return


def create_bidir_sequenced_dataset(X, time_steps=10):
    """
    create_bidir_sequenced_dataset reshapes data to temporalize it into (samples, timestamps, features).
    X is the data to be reshaped.
    time_stamps defines a sequence of how far backward and forward to consider for each sample/row.
    Outputs:
    Xs is an array of data reshaped for input into an LSTM model.
    ys is an array of data outputs corresponding to each Xs input.
    """
    Xs, ys = [], []  # start empty list
    for i in range(time_steps, len(X) - time_steps):  # loop within range of data frame minus the time steps
        v = pd.concat([X.iloc[(i - time_steps):i], X.iloc[(i + 1):(i + time_steps + 1)]]).values  # data from i backward and forward the specified number of time steps
        Xs.append(v)
        ys.append(X.iloc[i].values)

    return np.array(Xs).astype(np.float32), np.array(ys)  # convert lists into numpy arrays and return


def create_vanilla_model(cells, time_steps, num_features, dropout, input_loss='mae', input_optimizer='adam'):
    """
    Uses sequential model class from keras. Adds LSTM vanilla layer.
    time_steps is number of steps to consider for each point.
    num_features is the number of variables being considered
    cells and dropout rate are hyper parameters.
    Output is a model structure.
    """
    model = Sequential()
    model.add(LSTM(cells, input_shape=(time_steps, num_features), dropout=dropout)),  # one LSTM layer with dropout regularization
    model.add(Dense(num_features))
    model.compile(loss=input_loss, optimizer=input_optimizer)

    return model


def create_bidir_model(cells, time_steps, num_features, dropout, input_loss='mae', input_optimizer='adam'):
    """
    Uses sequential model class from keras. Adds bidirectional layer. Adds LSTM vanilla layer.
    time_steps is number of steps to consider for each point in each direction.
    num_features is the number of variables being considered.
    cells and dropout rate are hyper parameters.
    Output is a model structure.
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(cells, dropout=dropout), input_shape=(time_steps*2, num_features)))
    model.add(Dense(num_features))
    model.compile(loss=input_loss, optimizer=input_optimizer)

    return model


def train_model(X_train, y_train, model, patience, monitor='val_loss', mode='min', epochs=100, verbose=1, batch_size=32,
                validation_split=0.1):
    """
    train_model fits the model to training data. Early stopping ensures that too many epochs of training are not used.
    Monitors the validation loss for improvements and stops training when improvement stops.
    X_train is training input data.
    y_train is training output data.
    model is a created LSTM model.
    patience indicates how long to wait.
    epochs, batch_size are hyperparameters.
    validation_split indicates how much data to use for internal training.
    """
    es = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, mode=mode)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,  # just set to something high, early stopping will monitor.
        verbose=verbose, # how to give output. 0 is silent. 1 is progress bar. 2 is one line per epoch.
        batch_size=batch_size,  # this can be optimized later
        validation_split=validation_split,  # use 10% of data for validation, use 90% for training.
        callbacks=[es],  # early stopping similar to earlier
        shuffle=False,  # because order matters
        )

    return history
