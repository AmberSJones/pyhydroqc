#####################################
# ANOMALY DETECTION #
#####################################
# This script performs anomaly detection for multiple variables. Parameters are defined.
# The complete workflow for model development and anomaly detection is carried out.
# Model types include ARIMA and LSTM (univariate/multivariate and vanilla/bidirectional).

import anomaly_utilities
import model_workflow
import rules_detect
import copy
import pandas as pd


class WFParams:
    pass
    """
    """


wfp_data = []
wfparam = WFParams()
site_params = []

# LSTM PARAMETERS #
#########################################
wfparam.time_steps = 10
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
wfparam.persist = 6
wfparam.window_sz = 30
wfparam.alpha = 0.0001
wfparam.threshold_min = 0.25
wfparam.widen = 1
wfparam.pdq = [1, 1, 3]
sensor_params.append(copy.deepcopy(wfparam))

#cond params
wfparam.max_range = 380
wfparam.min_range = 120
wfparam.persist = 6
wfparam.window_sz = 30
wfparam.alpha = 0.0001
wfparam.threshold_min = 4.0
wfparam.widen = 1
wfparam.pdq = [10, 1, 3]
sensor_params.append(copy.deepcopy(wfparam))

#ph params
wfparam.max_range = 9.2
wfparam.min_range = 7.5
wfparam.persist = 18
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
wfparam.persist = 6
wfparam.window_sz = 30
wfparam.alpha = 0.00001
wfparam.threshold_min = 0.4
wfparam.widen = 1
wfparam.pdq = [10, 1, 0]
sensor_params.append(copy.deepcopy(wfparam))

#cond params
wfparam.max_range = 500
wfparam.min_range = 175
wfparam.persist = 6
wfparam.window_sz = 40
wfparam.alpha = 0.00001
wfparam.threshold_min = 5.0
wfparam.widen = 1
wfparam.pdq = [6, 1, 2]
sensor_params.append(copy.deepcopy(wfparam))

#ph params
wfparam.max_range = 9.0
wfparam.min_range = 8.0
wfparam.persist = 18
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
wfparam.persist = 6
wfparam.window_sz = 30
wfparam.alpha = 0.0001
wfparam.threshold_min = 0.4
wfparam.widen = 1
wfparam.pdq = [0, 1, 5]
sensor_params.append(copy.deepcopy(wfparam))

#cond params
wfparam.max_range = 450
wfparam.min_range = 200
wfparam.persist = 6
wfparam.window_sz = 40
wfparam.alpha = 0.0001
wfparam.threshold_min = 5.0
wfparam.widen = 1
wfparam.pdq = [7, 1, 0]
sensor_params.append(copy.deepcopy(wfparam))

#ph params
wfparam.max_range = 9.2
wfparam.min_range = 8.0
wfparam.persist = 18
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
wfparam.persist = 6
wfparam.window_sz = 30
wfparam.alpha = 0.00001
wfparam.threshold_min = 0.4
wfparam.widen = 1
wfparam.pdq = [0, 0, 0]
sensor_params.append(copy.deepcopy(wfparam))

#cond params
wfparam.max_range = 2700
wfparam.min_range = 150
wfparam.persist = 6
wfparam.window_sz = 40
wfparam.alpha = 0.000001
wfparam.threshold_min = 5.0
wfparam.widen = 1
wfparam.pdq = [1, 1, 5]
sensor_params.append(copy.deepcopy(wfparam))

#ph params
wfparam.max_range = 9.5
wfparam.min_range = 7.5
wfparam.persist = 18
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
wfparam.persist = 6
wfparam.window_sz = 30
wfparam.alpha = 0.0001
wfparam.threshold_min = 0.4
wfparam.widen = 1
wfparam.pdq = [3, 1, 1]
sensor_params.append(copy.deepcopy(wfparam))

#cond params
wfparam.max_range = 800
wfparam.min_range = 200
wfparam.persist = 6
wfparam.window_sz = 40
wfparam.alpha = 0.00001
wfparam.threshold_min = 5.0
wfparam.widen = 1
wfparam.pdq = [9, 1, 4]
sensor_params.append(copy.deepcopy(wfparam))

#ph params
wfparam.max_range = 9.0
wfparam.min_range = 7.4
wfparam.persist = 18
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
wfparam.persist = 6
wfparam.window_sz = 30
wfparam.alpha = 0.0001
wfparam.threshold_min = 0.4
wfparam.widen = 1
wfparam.pdq = [1, 1, 0]
sensor_params.append(copy.deepcopy(wfparam))

#cond params
wfparam.max_range = 900
wfparam.min_range = 200
wfparam.persist = 6
wfparam.window_sz = 20
wfparam.alpha = 0.01
wfparam.threshold_min = 4.0
wfparam.widen = 1
wfparam.pdq = [0, 0, 5]
sensor_params.append(copy.deepcopy(wfparam))

#ph params
wfparam.max_range = 9.2
wfparam.min_range = 7.2
wfparam.persist = 18
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


# RETRIEVE DATA #
#########################################

sites = ['FranklinBasin', 'TonyGrove', 'WaterLab', 'MainStreet', 'Mendon', 'BlackSmithFork']
year = [2014, 2015, 2016, 2017, 2018, 2019]
sensor = ['temp', 'cond', 'ph', 'do']


for j in range(0, len(sites)):
    site = sites[j]
    if site == 'BlackSmithFork': year.pop(0)
    print("\n\n###########################################\n#Processing data for site: "
          + sites[j] + ".\n###########################################")
    df_full, sensor_array = anomaly_utilities.get_data(sites[j], sensor, year, path="./LRO_data/")


    # RULES BASED ANOMALY DETECTION #
    #########################################
    range_count = []
    persist_count = []
    # size = []
    for i in range(0, len(sensor_array)):
        sensor_array[sensor[i]], r_c = rules_detect.range_check(sensor_array[sensor[i]], site_params[j][i].max_range, site_params[j][i].min_range)
        range_count.append(r_c)
        sensor_array[sensor[i]], p_c = rules_detect.persistence(sensor_array[sensor[i]], site_params[j][i].persist)
        persist_count.append(p_c)
        # s = rules_detect.group_size(sensor_array[sensor[i]])
        # size.append(s)
        sensor_array[sensor[i]] = rules_detect.add_labels(sensor_array[sensor[i]], -9999)
        sensor_array[sensor[i]] = rules_detect.interpolate(sensor_array[sensor[i]])
        # print(str(sensor[i]) + ' longest detected group = ' + str(size[i]))
    print('Rules based detection complete.\n')


    ##############################################
    # MODEL AND ANOMALY DETECTION IMPLEMENTATION #
    ##############################################

    # ARIMA BASED DETECTION #
    #########################################
    ARIMA_detect = []
    for i in range(0, len(sensor)):
        df = sensor_array[sensor[i]]
        ARIMA_detect.append(
            model_workflow.ARIMA_detect(
                df, sensor[i], site_params[j][i],
                rules=False, plots=False, summary=False, output=True, site=site
                ))
    print('ARIMA detection complete.\n')

    # LSTM BASED DETECTION #
    #########################################

    # DATA: univariate,  MODEL: vanilla #
    model_type = 'vanilla'
    LSTM_detect_univar = []
    for i in range(0, len(sensor)):
        df = sensor_array[sensor[i]]
        LSTM_detect_univar.append(
            model_workflow.LSTM_detect_univar(
                df, sensor[i], site_params[j][i], model_type,
                rules=False, plots=False, summary=False, output=True, site=site
            ))

    # DATA: univariate,  MODEL: bidirectional #
    model_type = 'bidirectional'
    LSTM_detect_univar_bidir = []
    for i in range(0, len(sensor)):
        df = sensor_array[sensor[i]]
        LSTM_detect_univar_bidir.append(
            model_workflow.LSTM_detect_univar(
                df, sensor[i], site_params[j][i], model_type,
                rules=False, plots=False, summary=False, output=True, site=site
                ))

    # DATA: multivariate,  MODEL: vanilla #
    model_type = 'vanilla'
    LSTM_detect_multivar = \
        model_workflow.LSTM_detect_multivar(
            sensor_array, sensor, site_params[j], model_type,
            rules=False, plots=False, summary=False, output=True, site=site
            )

    # DATA: multivariate,  MODEL: bidirectional #
    model_type = 'bidirectional'
    LSTM_detect_multivar_bidirectional = \
        model_workflow.LSTM_detect_multivar(
            sensor_array, sensor, site_params[j], model_type,
            rules=False, plots=False, summary=False, output=True, site=site
            )

    #########################################

    print("Finished processing data: " + sites[j])
