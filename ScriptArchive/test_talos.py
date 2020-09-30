# EXPERIMENT WITH TALOS FOR HYPERPARAMETER TUNING

def create_model_talos(params, time_steps, num_features, input_loss='mae', input_optimizer='adam',
                 patience=3, monitor='val_loss', mode='min', epochs=100, validation_split=0.1):
    """Uses sequential model class from keras. Adds LSTM layer. Input samples, timesteps, features.
    Hyperparameters include number of cells, dropout rate. Output is encoded feature vector of the input data.
    Uses autoencoder by mirroring/reversing encoder to be a decoder."""
    model = Sequential()
    model.add(LSTM(params['cells'], input_shape=(time_steps, num_features)))  # one LSTM layer
    model.add(Dropout(params['dropout']))
    model.add(RepeatVector(time_steps))
    model.add(LSTM(params['cells'], return_sequences=True))  # mirror the encoder in the reverse fashion to create the decoder
    model.add(Dropout(params['dropout']))
    model.add(TimeDistributed(Dense(num_features)))

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

    return history, model


p = {'cells': [4, 8, 16, 32, 64, 128],
     'dropout': (0, 0.4, 10),
     'batch_size': [5, 10, 25, 50]}

scan_object = talos.Scan(X_train, y_train, params=p, model=create_model_talos, experiment_name='test')

scan_object.data.head()
