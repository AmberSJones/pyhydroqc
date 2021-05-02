#####################################
# PARAMETERS #
#####################################
# This file assigns parameters for all steps of the anomaly detection workflow.
# A workflow parameter object is defined. to which parameters are added. Parameters are defined specific to each site
#   and sensor contained in a list that can be referenced in the workflow. Site and sensor order/indices matter.
#   Franklin Basin, Tony Grove, Water Lab, Main Street, Mendon, Blacksmith Fork. Temperature, Specific Conductance,
#   pH, dissolved oxygen.
# The following describes the parameters generally:
#   LSTM parameters are not site specific.
#   max_range, min_range, persist are used for rules based anomaly detection.
#   window_sz, alpha, threhsold_min are used in determining the dynamic threshold.
#   widen is the widening factor.
#   pdq are ARIMA hyper parameters.
# To reference these parameters in a script, call: from parameters import site_params.


from dataclasses import dataclass

# CALIBRATION PARAMETERS #
#########################################

@dataclass
class CalibrationParameters:
    hour_low: int
    hour_high: int
    persist_low: int = None
    persist_high: int = None


calib_params = CalibrationParameters(hour_low=7,
                                     hour_high=17,
                                     persist_low=3,
                                     persist_high=7)

# LSTM PARAMETERS #
#########################################

@dataclass
class LSTMParameters:
    time_steps: int
    samples: int
    cells: int
    dropout: float
    patience: int


LSTM_params = LSTMParameters(time_steps=5,
                             samples=20000,
                             cells=128,
                             dropout=0.2,
                             patience=6)

# SITE AND SENSOR SPECIFIC PARAMETERS
#########################################

@dataclass
class Parameters:
    max_range: float = None
    min_range: float = None
    persist: int = None
    calib_threshold: float = None
    window_sz: int = None
    alpha: float = None
    threshold_min: float = None
    widen: int = None
    pdq: [int, int, int] = None


site_params = {
    'FranklinBasin': {
        'temp' : Parameters(max_range=13,
                           min_range=-2,
                           persist=30,
                           window_sz=30,
                           alpha=0.0001,
                           threshold_min=0.25,
                           widen=1,
                           pdq=[1, 1, 3]),
        'cond' : Parameters(max_range=380,
                            min_range=120,
                            persist=30,
                            window_sz=30,
                            alpha=0.0001,
                            threshold_min=4.0,
                            widen=1,
                            pdq=[10, 1, 3]),
        'ph' : Parameters(max_range=9.2,
                          min_range=7.5,
                          persist=45,
                          window_sz=30,
                          alpha=0.00001,
                          threshold_min=0.02,
                          widen=1,
                          pdq=[10, 1, 1]),
        'do' : Parameters(max_range=13,
                          min_range=8,
                          persist=45,
                          window_sz=30,
                          alpha=0.0001,
                          threshold_min=0.15,
                          widen=1,
                          pdq=[0, 1, 5])},
    'TonyGrove': {
        'temp': Parameters(max_range=20,
                           min_range=-2,
                           persist=30,
                           window_sz=30,
                           alpha=0.00001,
                           threshold_min=0.4,
                           widen=1,
                           pdq=[10, 1, 0]),
        'cond': Parameters(max_range=500,
                           min_range=175,
                           persist=30,
                           window_sz=40,
                           alpha=0.00001,
                           threshold_min=5.0,
                           widen=1,
                           pdq=[6, 1, 2]),
        'ph': Parameters(max_range=9.0,
                         min_range=8.0,
                         persist=45,
                         window_sz=40,
                         alpha=0.00001,
                         threshold_min=0.02,
                         widen=1,
                         pdq=[8, 1, 4]),
        'do': Parameters(max_range=14,
                         min_range=7,
                         persist=45,
                         window_sz=30,
                         alpha=0.0001,
                         threshold_min=0.15,
                         widen=1,
                         pdq=[10, 1, 0])},
    'Waterlab': {
        'temp': Parameters(max_range=18,
                           min_range=-2,
                           persist=30,
                           window_sz=30,
                           alpha=0.0001,
                           threshold_min=0.4,
                           widen=1,
                           pdq=[0, 1, 5]),
        'cond': Parameters(max_range=450,
                           min_range=200,
                           persist=30,
                           window_sz=40,
                           alpha=0.0001,
                           threshold_min=5.0,
                           widen=1,
                           pdq=[7, 1, 0]),
        'ph': Parameters(max_range=9.2,
                         min_range=8.0,
                         persist=45,
                         window_sz=40,
                         alpha=0.00001,
                         threshold_min=0.02,
                         widen=1,
                         pdq=[10, 1, 0]),
        'do': Parameters(max_range=14,
                         min_range=7,
                         persist=45,
                         window_sz=30,
                         alpha=0.00001,
                         threshold_min=0.15,
                         widen=1,
                         pdq=[1, 1, 1])},
    'MainStreet': {
        'temp': Parameters(max_range=20,
                           min_range=-2,
                           persist=30,
                           window_sz=30,
                           alpha=0.00001,
                           threshold_min=0.4,
                           widen=1,
                           pdq=[0, 0, 0]),
        'cond': Parameters(max_range=2700,
                           min_range=150,
                           persist=30,
                           window_sz=40,
                           alpha=0.000001,
                           threshold_min=5.0,
                           widen=1,
                           pdq=[1, 1, 5],
                           calib_threshold=60),
        'ph': Parameters(max_range=9.5,
                         min_range=7.5,
                         persist=45,
                         window_sz=20,
                         alpha=0.0001,
                         threshold_min=0.03,
                         widen=1,
                         pdq=[3, 1, 1],
                         calib_threshold=0.1),
        'do': Parameters(max_range=15,
                         min_range=5,
                         persist=45,
                         window_sz=30,
                         alpha=0.00001,
                         threshold_min=0.25,
                         widen=1,
                         pdq=[1, 1, 1],
                         calib_threshold=0.4)},
    'Mendon': {
        'temp': Parameters(max_range=28,
                           min_range=-2,
                           persist=30,
                           window_sz=30,
                           alpha=0.0001,
                           threshold_min=0.4,
                           widen=1,
                           pdq=[3, 1, 1]),
        'cond': Parameters(max_range=800,
                           min_range=200,
                           persist=30,
                           window_sz=40,
                           alpha=0.00001,
                           threshold_min=5.0,
                           widen=1,
                           pdq=[9, 1, 4]),
        'ph': Parameters(max_range=9.0,
                         min_range=7.4,
                         persist=45,
                         window_sz=20,
                         alpha=0.0001,
                         threshold_min=0.03,
                         widen=1,
                         pdq=[0, 1, 2]),
        'do': Parameters(max_range=15,
                         min_range=3,
                         persist=45,
                         window_sz=30,
                         alpha=0.001,
                         threshold_min=0.15,
                         widen=1,
                         pdq=[10, 1, 3])},
    'BlacksmithFork': {
        'temp': Parameters(max_range=28,
                           min_range=-2,
                           persist=30,
                           window_sz=30,
                           alpha=0.0001,
                           threshold_min=0.4,
                           widen=1,
                           pdq=[1, 1, 0]),
        'cond': Parameters(max_range=900,
                           min_range=200,
                           persist=30,
                           window_sz=20,
                           alpha=0.01,
                           threshold_min=4.0,
                           widen=1,
                           pdq=[0, 0, 5]),
        'ph': Parameters(max_range=9.2,
                         min_range=7.2,
                         persist=45,
                         window_sz=30,
                         alpha=0.00001,
                         threshold_min=0.03,
                         widen=1,
                         pdq=[0, 1, 4]),
        'do': Parameters(max_range=14,
                         min_range=2,
                         persist=45,
                         window_sz=30,
                         alpha=0.0001,
                         threshold_min=0.15,
                         widen=1,
                         pdq=[0, 0, 5])},
}