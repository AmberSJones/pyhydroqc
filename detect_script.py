#####################################
# ANOMALY DETECTION #
#####################################
# This script performs anomaly detection for multiple variables. Parameters are imported.
# The complete workflow for model development and anomaly detection is carried out.
# Model types include ARIMA and LSTM (univariate/multivariate and vanilla/bidirectional).

import anomaly_utilities
import model_workflow
import rules_detect
from parameters import site_params

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
    df_full, sensor_array = anomaly_utilities.get_data(sites[j], sensor, year, path="LRO_data/")


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
