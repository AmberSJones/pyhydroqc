# Anomaly Detection and Correction for Aquatic Sensor Data
This repository contains software to identify and correct anomalous values in time series data collected by in situ aquatic sensors. The code was developed for application to data collected in the Logan River Observatory, sourced at <http://lrodata.usu.edu/tsa/> or on [HydroShare](https://www.hydroshare.org/search/?q=logan%20river%20observatory). All functions contained in the package are [documented here](https://ambersjones.github.io/PyHydroQC/). The package may be installed from the [Test Python Package Index](https://test.pypi.org/project/pyhydroqc-AmberSJones/).

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