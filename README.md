# Anomaly Detection and Correction for Aquatic Sensor Data
This repository contains a software to identify and correct anomalous values in time series data collected by in situ aquatic sensors. The code was developed for application to data colleccted in the Logan River Observatory, sourced at http://lrodata.usu.edu/tsa/ or https://www.hydroshare.org/search/?q=logan%20river%20observatory.

Methods currently implemented include ARIMA (AutoRegressive Integrated Moving Average), LSTM (Long Short Term Memory), and Prophet. All are time series regresion methods that detect anomalies by comparing model estimates to sensor observations and labeling points as anomlous when they exceed a threshold.

There are multiple possible approaches for applying LSTM for anomaly detection/correction. 
- Vanilla LSTM: uses past values of a single variable to estimate the next value of that variable.
- Multivariate Vanilla LSTM: uses past values of multiple variables to estimate the next value for all variables.
- Bidirectional LSTM: uses past and future values of a single variable to estimate a value for that variable at the time step of interest.
- Multivariate Bidirectional LSTM: uses past and future values of multiple variables to estimate a value for all variables at the time step of interest.

Correction approaches depend on the method. For ARIMA, each group of consecutive anomalous points is considered as a unit. Separate ARIMA models are developed for valid points preceding and following the anomolous group. Model estimates are blended to achieve a correction. For LSTM, correction may be based on a univariate or multivariate approach. Each point is considered as a unit. The developed model is used to estimate a correction to the first anomalous point in a group, which will then be used as input to estimate the following anomalous point, and so on.

Files are organized by method for anomaly detection and data correction. Utilities files contain functions called by the other scripts. A typical workflow involves:
1. Retrieving data
2. Applying rules-based detection to screen data and apply initial corrections
3. Developing a model (i.e., ARIMA or LSTM)
4. Applying model to make time series predicitons
5. Determining a threshold and detecting anomalies by comparing sensor observations to modeled results
6. Widening the window over which an anomaly is identified
7. Comparing anomaly detections to data labeled by technicians (if available) and determining metrics
8. Applying developed models to make corrections for anomalous events

## File Descriptions

### anomaly_utilities.py
Contains functions for performing anomaly detection and correction:
- get_data: Retrieves and formats data
- anomaly_events: Widens anomalies and indexes events or groups of anomalous data
- assign_cm: A helper function for resizing anomaly events to the original size before widening
- compare_events: Compares anomaly events detected by an algorithm to events labeled by a technician
- metrics: Determines performance metrics of the detections relative to labeled data.
- event_metrics: Determines performance metrics based on number of events rather than the number of data points.
- print_metrics: Prints the metrics to the console.
- group_bools: Indexes contiguous groups of anomalous and normal data to facilitate correction.
- xfade: Uses a cross-fade to blend forecasted and backcasted data over anomaly events for generating data correction.
- set_dynamic_threshold: Creates a threshold envelope based on residual deviations.
- set_cons_threshold: Creates a threshold of constant value.
- detect_anomalies: Uses model residuals and threshold values to classify anomalous data
- plt_threshold: Plots thresholds and residuals
- plt_results: Plots the results of raw data, model predictions, detected and labeled anomalies

### modeling_utilities.py
Contains functions for building and training models:
- build_arima_model, LSTM_univar, LSTM_multivar, LSTM_univar_bidir, LSTM_multivar_bidir: are wrappers that call other functions to scale (for LSTM models only) and reshape data, create and train a model, and output model predictions and residuals.
- create_scaler: Creates a scaler object for scaling and unscaling data.
- create_training_dataset and create_bidir_training_dataset: Creates a training dataset based on a random selection of points. Reshapes data to include the desired time_steps for input to the LSTM model - either only past data or past and future data (bidirectional). Ensures that data already identified as anomalous are not used.
- create_sequenced_dataset and create_bidir_sequenced_dataset: Reshapes all inputs into sequences that include time_steps for input to the LSTM model - either only past data or past and future data (bidirectional). Used for testing or for applying the model to a full dataset.
- create_vanilla_model, create_bidir_model: are helper functions used to create single layer LSTM models.
- train_model: Fits the model to training data. Uses a validation subset to monitor for improvemens to ensure that training is not too long.

### rules_detect.py
Contains functions for rules-based anomaly detection and preprocessing. Functions include:
- range_check: Scans for data outside of user defined limits and marks them as anomalous
- persistence: Scans for repeated values in the data and marks them as anomalous
- group_size: Identifies the maximum length of identified anomalous groups
- interpolate: Corrects data with linear interpolation, a typical approach for short anomalous events
Depends on anomaly_utilities.py

### model_workflow.py
Contain functionality to build and train ARIMA and LSTM models, apply it to make predictions, set a threshold, detect anomalies, widens anomalous events, and determines metrics. Function names are: ARIMA_detect, LSTM_detect_univar, and LSTM_detect_multivar. 
Depends on anomaly_utilities.py, modeling_utilities.py, and rules_detect.py

### ARIMA_correct.py
Contains functionality to perform corrections and plot results using ARIMA models:
- ARIMA_group:
- ARIMA_forecast: Creates predictions of data where anomalies occur
- generate_corrections: The primary function called to determine corrections. Passes through data with anomalies and determines corrections using picewise ARIMA models. Corrections are determined by averaging both a forecast and a backcast.
Depends on anomaly_utilities.py

### LSTM_correct.py
Separate functions correct univariate and multivariate data and plot results. The functions step through each data point and determine a correction based on the previous time_steps.
- LSTM_correct
- LSTM_multi_correct

## Dependencies
This software depends on the following Python packages:
- numpy
- pandas
- matplotlib
- os
- scipy
- warnings
- random
- tensorflow
- statsmodels
- sklearn
- pmdarima
