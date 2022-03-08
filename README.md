# Anomaly Detection and Correction for Aquatic Sensor Data
This repository contains software to identify and correct anomalous values in time series data collected by in situ aquatic sensors. The code was developed for application to data collected in the Logan River Observatory, sourced at <http://lrodata.usu.edu/tsa/> or on [HydroShare](https://www.hydroshare.org/search/?q=logan%20river%20observatory). All functions contained in the package are [documented here](https://ambersjones.github.io/pyhydroqc/). The package may be installed from the [Python Package Index](https://pypi.org/project/pyhydroqc/).

The package development, testing, and performance are reported in Jones, A.S., Jones, T.L., Horsburgh, J.S. (2022). Toward automated post processing of aquatic sensor data, Environmental Modelling and Software, <https://doi.org/10.1016/j.envsoft.2022.105364>

Methods currently implemented include ARIMA (AutoRegressive Integrated Moving Average) and LSTM (Long Short Term Memory). These are time series regression methods that detect anomalies by comparing model estimates to sensor observations and labeling points as anomalous when they exceed a threshold.

There are multiple possible approaches for applying LSTM for anomaly detection/correction. 
- Vanilla LSTM: uses past values of a single variable to estimate the next value of that variable.
- Multivariate Vanilla LSTM: uses past values of multiple variables to estimate the next value for all variables.
- Bidirectional LSTM: uses past and future values of a single variable to estimate a value for that variable at the time step of interest.
- Multivariate Bidirectional LSTM: uses past and future values of multiple variables to estimate a value for all variables at the time step of interest.

Correction approaches depend on the method. For ARIMA, each group of consecutive anomalous points is considered as a unit to be corrected. Separate ARIMA models are developed for valid points preceding and following the anomalous group. Model estimates are blended to achieve a correction. For LSTM, correction may be based on a univariate or multivariate approach. Correction is made on a point-by-point basis where each point is considered a separate unit to be corrected. The developed model is used to estimate a correction to the first anomalous point in a group, which is then used as input to estimate the following anomalous point, and so on.

Files are organized by method for anomaly detection and data correction. Utilities files contain functions, wrappers, and parameter definitions called by the other scripts. A typical workflow involves:
1. Retrieving data
2. Applying rules-based detection to screen data and apply initial corrections
3. Developing a model (i.e., ARIMA or LSTM)
4. Applying model to make time series predictions
5. Determining a threshold and detecting anomalies by comparing sensor observations to modeled results
6. Widening the window over which an anomaly is identified
7. Comparing anomaly detections to data labeled by technicians (if available) and determining metrics
8. Applying developed models to make corrections for anomalous events

## File Descriptions

### detect.script.py
This script contains the code to apply anomaly detection methods to data from four sensors (water temperature, specific conductance, pH, dissolved oxygen) at six sites in the Logan River Observatory. The script calls functions to retrieve data, perform rules based anomaly detection and correction, develop and get estimates from five models (ARIMA, LSTM univaraite, LSTM univariate bidirectional, LSTM multivaraiate, and LSTM multivariate bidirectional), determine dynamic thresholds and detect anomalies, widen the window of detection and compare to raw data, and determine metrics. This application script refers to parameters stored in the parameters file.

### parameters.py
This file contains assignments of parameters for all steps of the anomaly detection workflow. Parameters are defined specific to each site and sensor that are referenced in the detect script. LSTM parameters are consistent across sites and variables. ARIMA hyper parameters are specific to each site/sensor combination, other parameters are used for rules based anomaly detection, determining dynamic thresholds, and for widening anomalous events.  

### anomaly_utilities.py
Contains functions for performing anomaly detection and correction:
- get_data: Retrieves and formats data. Data from the LRO was extracted from the database to csv files, and retrieval is based on site, sensor, and year according to the file organization. To pass through subsequent steps, the required format is a data frame with columns corresponding to datetime (as the index), raw data, corrected data, and data labels (anomalies identified by technicians).
- anomaly_events: Widens anomalies and indexes events or groups of anomalous data.
- assign_cm: A helper function for resizing anomaly events to the original size for determining metrics.
- compare_events: Compares anomaly events detected by an algorithm to events labeled by a technician.
- metrics: Determines performance metrics of the detections relative to labeled data.
- event_metrics: Determines performance metrics based on number of events rather than the number of data points.
- print_metrics: Prints the metrics to the console.
- group_bools: Indexes contiguous groups of anomalous and valid data to facilitate correction.
- xfade: Uses a cross-fade to blend forecasted and backcasted data over anomaly events for generating data correction.
- set_dynamic_threshold: Creates a threshold that varies dynamically based on the model residuals.
- set_cons_threshold: Creates a threshold of constant value.
- detect_anomalies: Uses model residuals and threshold values to classify anomalous data.
- aggregate_results: Combines the detections from multiple models to give a single output of anomaly detections.
- plt_threshold: Plots thresholds and model residuals.
- plt_results: Plots raw data, model predictions, detected and labeled anomalies.

### modeling_utilities.py
Contains functions for building and training models:
- pdq: Automatically determines the (p, d, q) hyperparameters of a time series for ARIMA modeling.
- build_arima_model, lstm_univar, lstm_multivar, lstm_univar_bidir, lstm_multivar_bidir: wrappers that call other functions in the file to scale and reshape data (for LSTM models only), create and train a model, and output model predictions and residuals.
- create_scaler: Creates a scaler object for scaling and unscaling data.
- create_training_dataset and create_bidir_training_dataset: Creates a training dataset based on a random selection of points from the dataset. Reshapes data to include the desired time_steps for input to the LSTM model - the number of past data points to examine or past and future points (bidirectional). Ensures that data already identified as anomalous (i.e., by rules based detection) are not used.
- create_sequenced_dataset and create_bidir_sequenced_dataset: Reshapes all inputs into sequences that include time_steps for input to the LSTM model - using either only past data points or past and future data points (bidirectional). Used for testing or for applying the model to a full dataset.
- create_vanilla_model, create_bidir_model: Helper functions used to create single layer LSTM models.
- train_model: Fits the model to training data. Uses a validation subset to monitor for improvements to ensure that training is not too long.

### rules_detect.py
Contains functions for rules based anomaly detection and preprocessing. Depends on anomaly_utilities.py Functions include:
- range_check: Scans for data outside of user defined limits and marks the points as anomalous.
- persistence: Scans for repeated values in the data and marks the points as anomalous if the duration exceeds a user defined length.
- group_size: Determines the maximum length of anomalous groups identified by the previous steps.
- interpolate: Corrects data with linear interpolation, a typical approach for short anomalous events.
- add_labels: Enables the addition of anomaly labels (referring to anomalies previously identified by an expert) in the case that labels may have been missed for corrected data that are NaN or a no data value (e.g, -9999).

### calibration.py
Contains functions for identifying and correcting calibration events. Functions include:
- calib_edge_detect: identifies possible calibration event candidates by using edge filtering.
- calib_persist_detect: identifies possible calibration event candidates based on persistence of a user defined length.
- calib_overlap: identifies possible calibration event candidates by finding concurrent events of multiple sensors from the calib_persist_detect function.
- find_gap: determines a gap value for a calibration event based on the largest data difference within a time window around a datetime.
- lin_drift_cor: performs linear drift correction to address sensor drift given calibration dates and a gap value.

### model_workflow.py
Contains functionality to build and train ARIMA and LSTM models, apply the models to make predictions, set thresholds, detect anomalies, widen anomalous events, and determine metrics. Depends on anomaly_utilities.py, modeling_utilities.py, and rules_detect.py. 
Wrapper function names are: arima_detect, lstm_detect_univar, and lstm_detect_multivar. LSTM model workflows include options for vanilla or bidirectional. Within each wrapper function, the full detection workflow is followed. Options allow for output of plots, summaries, and metrics.

### ARIMA_correct.py
Contains functionality to perform corrections and plot results using ARIMA models. Depends on anomaly_utilities.py.
- arima_group: Ensures that the valid data surrounding anomalous points and groups of points are sufficient forecasting/backcasting.
- arima_forecast: Creates predictions of data where anomalies occur.
- generate_corrections: The primary function for determining corrections. Passes through data with anomalies and determines corrections using piecewise ARIMA models. Corrections are determined by averaging together (cross fade) both a forecast and a backcast.

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
- copy

## Sponsors and Credits
[![NSF-1931297](https://img.shields.io/badge/NSF-1931297-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1931297)

The material in this repository is based on work supported by National Science Foundation Grant [1931297](http://www.nsf.gov/awardsearch/showAward?AWD_ID=1931297). Any opinions, findings, and conclusions or recommendations expressed are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
