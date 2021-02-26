#####################################
# FRANKLIN BASIN PARAMETERS  #
#####################################
# This file assigns parameters for all steps of the anomaly detection workflow.
# The following describes the parameters generally:
#   LSTM parameters are not site specific.
#   max_range, min_range, persist are used for rules based anomaly detection.
#   window_sz, alpha, threhsold_min are used in determining the dynamic threshold.
#   widen is the widening factor.
#   pdq are ARIMA hyperparameters.
# To reference these parameters in a script, call: from parameters import site_params.

# CALIBRATION PARAMETERS #
#########################################
calib_params = {
    'persist_low': 3,
    'persist_high': 7,
    'hour_low': 7,
    'hour_high': 17,
}

# LSTM PARAMETERS #
#########################################
LSTM_params = {
    'time_steps': 5,
    'samples': 3500,
    'cells': 128,
    'dropout': 0.2,
    'patience': 6,
}

# SITE AND SENSOR SPECIFIC PARAMETERS
#########################################
site_params = {
    'FranklinBasin': {
        'temp': {
            'max_range': 13,
            'min_range': -2,
            'persist': 30,
            'window_sz': 30,
            'alpha': 0.0001,
            'threshold_min': 0.25,
            'widen': 1,
            'pdq': [1, 1, 1],
        },
        'cond': {
            'max_range': 380,
            'min_range': 120,
            'persist': 30,
            'window_sz': 30,
            'alpha': 0.0001,
            'threshold_min': 4.0,
            'widen': 1,
            'pdq': [1, 1, 1],
        },
        'ph': {
            'max_range': 9.2,
            'min_range': 7.5,
            'persist': 45,
            'window_sz': 30,
            'alpha': 0.00001,
            'threshold_min': 0.02,
            'widen': 1,
            'pdq': [1, 1, 1],
        },
        'do': {
            'max_range': 13,
            'min_range': 8,
            'persist': 45,
            'window_sz': 30,
            'alpha': 0.0001,
            'threshold_min': 0.15,
            'widen': 1,
            'pdq': [1, 1, 1],
        },
    },

}
