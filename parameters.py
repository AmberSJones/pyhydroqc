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


import copy


class WFParams:
    pass
    """
    """


wfp_data = []
wfparam = WFParams()
site_params = []

# LSTM PARAMETERS #
#########################################
wfparam.time_steps = 5
wfparam.samples = 20000
wfparam.cells = 128
wfparam.dropout = 0.2
wfparam.patience = 6


############################
# SITE SPECIFIC Parameters #
############################

# FRANKLIN BASIN PARAMETERS #
#########################################
sensor_params = []
#temp params
wfparam.max_range = 13
wfparam.min_range = -2
wfparam.persist = 10
wfparam.window_sz = 30
wfparam.alpha = 0.0001
wfparam.threshold_min = 0.25
wfparam.widen = 1
wfparam.pdq = [1, 1, 3]
sensor_params.append(copy.deepcopy(wfparam))

#cond params
wfparam.max_range = 380
wfparam.min_range = 120
wfparam.persist = 10
wfparam.window_sz = 30
wfparam.alpha = 0.0001
wfparam.threshold_min = 4.0
wfparam.widen = 1
wfparam.pdq = [10, 1, 3]
sensor_params.append(copy.deepcopy(wfparam))

#ph params
wfparam.max_range = 9.2
wfparam.min_range = 7.5
wfparam.persist = 30
wfparam.window_sz = 30
wfparam.alpha = 0.00001
wfparam.threshold_min = 0.02
wfparam.widen = 1
wfparam.pdq = [10, 1, 1]
sensor_params.append(copy.deepcopy(wfparam))

#do params
wfparam.max_range = 13
wfparam.min_range = 8
wfparam.persist = 10
wfparam.window_sz = 30
wfparam.alpha = 0.0001
wfparam.threshold_min = 0.15
wfparam.widen = 1
wfparam.pdq = [0, 1, 5]
sensor_params.append(copy.deepcopy(wfparam))

site_params.append(copy.deepcopy(copy.deepcopy(sensor_params)))

# TONY GROVE PARAMETERS #
#########################################
sensor_params = []
#temp params
wfparam.max_range = 20
wfparam.min_range = -2
wfparam.persist = 10
wfparam.window_sz = 30
wfparam.alpha = 0.00001
wfparam.threshold_min = 0.4
wfparam.widen = 1
wfparam.pdq = [10, 1, 0]
sensor_params.append(copy.deepcopy(wfparam))

#cond params
wfparam.max_range = 500
wfparam.min_range = 175
wfparam.persist = 10
wfparam.window_sz = 40
wfparam.alpha = 0.00001
wfparam.threshold_min = 5.0
wfparam.widen = 1
wfparam.pdq = [6, 1, 2]
sensor_params.append(copy.deepcopy(wfparam))

#ph params
wfparam.max_range = 9.0
wfparam.min_range = 8.0
wfparam.persist = 30
wfparam.window_sz = 40
wfparam.alpha = 0.00001
wfparam.threshold_min = 0.02
wfparam.widen = 1
wfparam.pdq = [8, 1, 4]
sensor_params.append(copy.deepcopy(wfparam))

#do params
wfparam.max_range = 14
wfparam.min_range = 7
wfparam.persist = 10
wfparam.window_sz = 30
wfparam.alpha = 0.0001
wfparam.threshold_min = 0.15
wfparam.widen = 1
wfparam.pdq = [10, 1, 0]
sensor_params.append(copy.deepcopy(wfparam))

site_params.append(copy.deepcopy(copy.deepcopy(sensor_params)))

# WATERLAB PARAMETERS #
#########################################
sensor_params = []
#temp params
wfparam.max_range = 18
wfparam.min_range = -2
wfparam.persist = 10
wfparam.window_sz = 30
wfparam.alpha = 0.0001
wfparam.threshold_min = 0.4
wfparam.widen = 1
wfparam.pdq = [0, 1, 5]
sensor_params.append(copy.deepcopy(wfparam))

#cond params
wfparam.max_range = 450
wfparam.min_range = 200
wfparam.persist = 10
wfparam.window_sz = 40
wfparam.alpha = 0.0001
wfparam.threshold_min = 5.0
wfparam.widen = 1
wfparam.pdq = [7, 1, 0]
sensor_params.append(copy.deepcopy(wfparam))

#ph params
wfparam.max_range = 9.2
wfparam.min_range = 8.0
wfparam.persist = 30
wfparam.window_sz = 40
wfparam.alpha = 0.00001
wfparam.threshold_min = 0.02
wfparam.widen = 1
wfparam.pdq = [10, 1, 0]
sensor_params.append(copy.deepcopy(wfparam))

#do params
wfparam.max_range = 14
wfparam.min_range = 7
wfparam.persist = 10
wfparam.window_sz = 30
wfparam.alpha = 0.00001
wfparam.threshold_min = 0.15
wfparam.widen = 1
wfparam.pdq = [1, 1, 1]
sensor_params.append(copy.deepcopy(wfparam))

site_params.append(copy.deepcopy(copy.deepcopy(sensor_params)))

# MAIN STREET PARAMETERS #
#########################################
sensor_params = []
#temp params
wfparam.max_range = 20
wfparam.min_range = -2
wfparam.persist = 10
wfparam.window_sz = 30
wfparam.alpha = 0.00001
wfparam.threshold_min = 0.4
wfparam.widen = 1
wfparam.pdq = [0, 0, 0]
sensor_params.append(copy.deepcopy(wfparam))

#cond params
wfparam.max_range = 2700
wfparam.min_range = 150
wfparam.persist = 10
wfparam.window_sz = 40
wfparam.alpha = 0.000001
wfparam.threshold_min = 5.0
wfparam.widen = 1
wfparam.pdq = [1, 1, 5]
sensor_params.append(copy.deepcopy(wfparam))

#ph params
wfparam.max_range = 9.5
wfparam.min_range = 7.5
wfparam.persist = 30
wfparam.window_sz = 20
wfparam.alpha = 0.0001
wfparam.threshold_min = 0.03
wfparam.widen = 1
wfparam.pdq = [3, 1, 1]
sensor_params.append(copy.deepcopy(wfparam))

#do params
wfparam.max_range = 15
wfparam.min_range = 5
wfparam.persist = 10
wfparam.window_sz = 30
wfparam.alpha = 0.00001
wfparam.threshold_min = 0.25
wfparam.widen = 1
wfparam.pdq = [1, 1, 1]
sensor_params.append(copy.deepcopy(wfparam))

site_params.append(copy.deepcopy(copy.deepcopy(sensor_params)))

# MENDON PARAMETERS #
#########################################
sensor_params = []
#temp params
wfparam.max_range = 28
wfparam.min_range = -2
wfparam.persist = 10
wfparam.window_sz = 30
wfparam.alpha = 0.0001
wfparam.threshold_min = 0.4
wfparam.widen = 1
wfparam.pdq = [3, 1, 1]
sensor_params.append(copy.deepcopy(wfparam))

#cond params
wfparam.max_range = 800
wfparam.min_range = 200
wfparam.persist = 10
wfparam.window_sz = 40
wfparam.alpha = 0.00001
wfparam.threshold_min = 5.0
wfparam.widen = 1
wfparam.pdq = [9, 1, 4]
sensor_params.append(copy.deepcopy(wfparam))

#ph params
wfparam.max_range = 9.0
wfparam.min_range = 7.4
wfparam.persist = 30
wfparam.window_sz = 20
wfparam.alpha = 0.0001
wfparam.threshold_min = 0.03
wfparam.widen = 1
wfparam.pdq = [0, 1, 2]
sensor_params.append(copy.deepcopy(wfparam))

#do params
wfparam.max_range = 15
wfparam.min_range = 3
wfparam.persist = 10
wfparam.window_sz = 30
wfparam.alpha = 0.001
wfparam.threshold_min = 0.15
wfparam.widen = 1
wfparam.pdq = [10, 1, 3]
sensor_params.append(copy.deepcopy(wfparam))

site_params.append(copy.deepcopy(copy.deepcopy(sensor_params)))

# BLACKSMITH FORK PARAMETERS #
#########################################
sensor_params = []
#temp params
wfparam.max_range = 28
wfparam.min_range = -2
wfparam.persist = 10
wfparam.window_sz = 30
wfparam.alpha = 0.0001
wfparam.threshold_min = 0.4
wfparam.widen = 1
wfparam.pdq = [1, 1, 0]
sensor_params.append(copy.deepcopy(wfparam))

#cond params
wfparam.max_range = 900
wfparam.min_range = 200
wfparam.persist = 10
wfparam.window_sz = 20
wfparam.alpha = 0.01
wfparam.threshold_min = 4.0
wfparam.widen = 1
wfparam.pdq = [0, 0, 5]
sensor_params.append(copy.deepcopy(wfparam))

#ph params
wfparam.max_range = 9.2
wfparam.min_range = 7.2
wfparam.persist = 30
wfparam.window_sz = 30
wfparam.alpha = 0.00001
wfparam.threshold_min = 0.03
wfparam.widen = 1
wfparam.pdq = [0, 1, 4]
sensor_params.append(copy.deepcopy(wfparam))

#do params
wfparam.max_range = 14
wfparam.min_range = 2
wfparam.persist = 10
wfparam.window_sz = 30
wfparam.alpha = 0.0001
wfparam.threshold_min = 0.15
wfparam.widen = 1
wfparam.pdq = [0, 0, 5]
sensor_params.append(copy.deepcopy(wfparam))

site_params.append(copy.deepcopy(copy.deepcopy(sensor_params)))

