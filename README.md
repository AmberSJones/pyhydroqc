# Anomaly Detection and Correction for Aquatic Sensor Data
This repository contains a set of scripts to identify and correct anomalous values in time series data collected by in situ aquatic sensors. The code was developed for application to data colleccted in the Logan River Observatory, sourced at http://lrodata.usu.edu/tsa/ or https://www.hydroshare.org/search/?q=logan%20river%20observatory.

Methods currently implemented include ARIMA (AutoRegressive Integrated Moving Average), LSTM (Long Short Term Memory), and Prophet. All are time series regresion methods that detect anomalies by comparing model estimates to sensor observations and labeling points as anomlous when they exceed a threshold.

There are multiple possible approaches for applying LSTM for anomaly detection/correction. 
- Vanilla LSTM: uses past values of a single variable to estimate the next value of that variable.
- Multivariate Vanilla LSTM: uses past values of multiple variables to estimate the next value for all variables.
- Bidirectional LSTM: uses past and future values of a single variable to estimate a value for that variable at the time step of interest.
- Multivariate Bidirectional LSTM: uses past and future values of multiple variables to estimate a value for all variables at the time step of interest.

Correction approaches depend on the method. For ARIMA, each group of consecutive anomalous points is considered as a unit. Separate ARIMA models are developed for valid points preceding and following the anomolous group. Model estimates are blended to achieve a correction. For LSTM, correction may be based on a univariate or multivariate approach. Each point is considered as a unit. The developed model is used to estimate a correction the first anomalous point in a group, which will then be used as input to estimate the following anomalous point, and so on.

Scripts are organized by method for anomaly detection and data correction. Several scripts are utilities files containing functions called by the other scripts. A typical workflow involves:
1. Retrieving data
2. Applying rules-based detection to screen data and apply initial corrections
3. Developing a model (i.e., ARIMA or LSTM). 
4. Applying model to make time series predicitons.
5. Determining a threshold and detecting anomalies by comparing sensor observations to modeled results.
6. Widening the window over which an anomaly is identified.
7. Comparing anomaly detections to data labeled by technicians (if available) and determining metrics.
8. Applying developed models to make corrections for anomalous events.

## Script Descriptions

### rules_detect
Contains functions for rules-based anomaly detection and preprocessing. Functions include a range check with user defined limits, a check for data persistence, and a function to identify the maximum length of identified anomalous groups. A function is also included to correct data with linear interpolation, a typical approach for short anomalous events.

### anomaly_utilities
Contains functions for performing anomaly detection:
- get_data: Retrieves and formats data
- anomaly_events: Widens the window of anomaly detection
- compare_labeled_detected: Compares events detected by an algorithm to events labeled by a technician
- metrics: Determines performance metrics of the detections relative to labeled data.
- group_bools: Indexes groups of anomalous and normal data to facilitate correction.
- xfade: Uses a cross-fade to blend forecasted and backcasted data to correct periods of anomalous data.

### ARIMA_detect
Contains functionality to build an ARIMA model, apply it to make predictions, set a threshold, and detect anomalies. Includes an example of application to Logan River data with ARIMA hyperparameters (p, d, q) determined by an external automatic procedure. Example uses widening events, comparing, and metrics functions from the anomaly_utilities.

### LSTM_utilities
Contains utilities for detecting anomalies using LSTM models:
- vanilla_LSTM_model, multi_vanilla_LSTM_wrapper, bidir_LSTM_model, multi_bidir_LSTM_model: wrappers that call other functions to scale and reshape data, create and train a model, and output model predictions and residuals.
- create_scaler: Creates a scaler object based on input data.
- create_clean_training_dataset and create_bidir_training_dataset: Creates a training dataset based on a random selection of points. Reshapes data to include the desired time_steps for input to the LSTM model - either only past data or past and future data (bidirectional). Ensures that data already identified as anomalous are not used.
- create_sequenced_dataset and create_bidir_sequenced_dataset: Reshapes all inputs into sequences that include time_steps for input to the LSTM model - either only past data or past and future data (bidirectional). Used for testing or for applying the model to a full dataset.
- create_vanilla_model, create_bidir_model: Creates single layer LSTM model (either vanilla or bidirectional).
- train_model: Fits the model to training data. Uses a validation subset to monitor for improvemens to ensure that training is not too long.
- detect_anomalies, detect_anomalies_bidir: Uses model results and user input threshold to detect anomalies in observations.

### LSTM_detect, LSTM_bidirectional, LSTM_multi, LSTM_multi_bidir
Each model type includes an example of application to the Logan River data. Fetches data, creates and trains the model, applies the model and identifies anomalies, widens anomalous events and determines metrics, all based on functions from the anomaly_utilities and LSTM_utilities files.

### prophet_detct

### ARIMA_correct
Contains functionality to perform corrections using ARIMA models:
- ARIMA_group:
- ARIMA_forecast: Creates predictions of data where anomalies occur
- generate_corrections: The primary function called to determine corrections. Passes through data with anomalies and determines corrections using picewise ARIMA models. Corrections are determined by averaging both a forecast and a backcast.

### LSTM_correct and LSTM_correct_bidir
Each model type contains a function for correcting both univariate and multivariate data:
- LSTM_correct
- LSTM_multi_correct


