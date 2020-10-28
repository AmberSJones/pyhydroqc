#####################################
# ANOMALY DETECTION: FRANKLIN BASIN #
#####################################
# This script performs anomaly detection for multiple variables at Franklin Basin site. Parameters are defined.
# The complete workflow for model development and anomaly detection is carried out.
# Model types include ARIMA and LSTM (univariate/multivariate and vanilla/bidirectional).

import anomaly_utilities
import model_workflow
import rules_detect
import pandas as pd

#####################################
# SITE SPECIFIC SETTINGS #
#####################################

# RETRIEVE DATA #
#########################################
site = 'FranklinBasin'
sensor = ['temp', 'cond', 'ph', 'do']
year = [2014, 2015, 2016, 2017, 2018, 2019]
df_full, sensor_array = anomaly_utilities.get_data(site, sensor, year, path="LRO_data/")

# RULES DETECTION PARAMETERS #
#########################################
maximum = [13, 380, 9.2, 13]
minimum = [-2, 120, 7.5, 8]
length = [6, 3, 3, 3]

# RULES BASED ANOMALY DETECTION #
#########################################
# size = []
range_count = []
persist_count = []
for i in range(0, len(sensor_array)):
    sensor_array[sensor[i]], r_c = rules_detect.range_check(sensor_array[sensor[i]], maximum[i], minimum[i])
    range_count.append(r_c)
    sensor_array[sensor[i]], p_c = rules_detect.persistence(sensor_array[sensor[i]], length[i])
    persist_count.append(p_c)
    # s = rules_detect.group_size(sensor_array[sensor[i]])
    # size.append(s)
    sensor_array[sensor[i]] = rules_detect.add_labels(sensor_array[sensor[i]], -9999)
    sensor_array[sensor[i]] = rules_detect.interpolate(sensor_array[sensor[i]])
    # print(str(sensor[i]) + ' maximum detected group length = ' + str(size[i]))
print('Rules based detection complete.\n')


# Results Plot #
plt.figure()
plt.plot(sensor_array[sensor[i]]['raw'], 'b', label='original data')
plt.plot(sensor_array[sensor[i]]['raw'][sensor_array[sensor[i]]['anomaly']], 'r+', label='machine detected anomalies')
plt.legend()
plt.ylabel(sensor)
plt.show()




# ANOMALY DETECTION PARAMETERS #
#########################################
window_sz = [30, 30, 30, 30]
alpha = [0.0001, 0.0001, 0.00001, 0.0001]
min_range = [0.25, 4, 0.02, 0.15]
wf = [1, 1, 1, 1]

# ARIMA PARAMETERS #
#########################################
# Need to use an automated method to generalize getting p,d,q parameters
# These are the results of using auto.ARIMA to determine p,d,q parameters in R
sites = {'BlackSmithFork': 0,
         'FranklinBasin': 1,
         'MainStreet': 2,
         'Mendon': 3,
         'TonyGrove': 4,
         'WaterLab': 5}
sensors = {'cond': 0,
           'do': 1,
           'ph': 2,
           'temp': 3,
           'turb': 4}
pdqParams = [
    [[0, 0, 5], [0, 0, 5], [0, 1, 4], [1, 1, 0], [9, 1, 5]],  # BlackSmithFork
    [[10, 1, 3], [0, 1, 5], [10, 1, 1], [6, 1, 4], [0, 1, 5]],  # FranklinBasin
    [[1, 1, 5], [1, 1, 1], [3, 1, 1], [0, 0, 0], [1, 1, 5]],  # MainStreet
    [[9, 1, 4], [10, 1, 3], [0, 1, 2], [3, 1, 1], [9, 1, 1]],  # Mendon
    [[6, 1, 2], [10, 1, 0], [8, 1, 4], [10, 1, 0], [10, 1, 5]],  # TonyGrove
    [[7, 1, 0], [1, 1, 1], [10, 1, 0], [0, 1, 5], [1, 1, 3]]  # WaterLab
]
pdqParam = pd.DataFrame(pdqParams, columns=sensors, index=sites)

# LSTM PARAMETERS #
#########################################
time_steps = 10
samples = 20000
cells = 128
dropout = 0.2
patience = 6


##############################################
# MODEL AND ANOMALY DETECTION IMPLEMENTATION #
##############################################

# ARIMA BASED DETECTION #
#########################################
ARIMA_detect = []
for i in range(0, len(sensor)):
    p, d, q = pdqParam[sensor[i]][site]
    print('sensor: ' + str(sensor[i]) + ', p, d, q = (' + str(p) + ', ' + str(d) + ', ' + str(q) + ')')
    df = sensor_array[sensor[i]]
    ARIMA_detect.append(
        model_workflow.ARIMA_detect(
            df, sensor[i], p, d, q,
            minimum[i], maximum[i], length[i],
            window_sz[i], alpha[i], min_range[i], wf[i],
            rules=False, plots=False, summary=False, output=True, site=site
            ))

# LSTM BASED DETECTION #
#########################################

# DATA: univariate,  MODEL: vanilla #
model_type = 'vanilla'
LSTM_detect_univar = []
for i in range(0, len(sensor)):
    df = sensor_array[sensor[i]]
    LSTM_detect_univar.append(
        model_workflow.LSTM_detect_univar(
            df, sensor[i],
            minimum[i], maximum[i], length[i],
            model_type, time_steps, samples, cells, dropout, patience,
            window_sz[i], alpha[i], min_range[i], wf[i],
            rules=False, plots=False, summary=False, output=True, site=site
        ))

# DATA: univariate,  MODEL: bidirectional #
model_type = 'bidirectional'
LSTM_detect_univar_bidir = []
for i in range(0, len(sensor)):
    df = sensor_array[sensor[i]]
    LSTM_detect_univar_bidir.append(
        model_workflow.LSTM_detect_univar(
            df, sensor[i],
            minimum[i], maximum[i], length[i],
            model_type, time_steps, samples, cells, dropout, patience,
            window_sz[i], alpha[i], min_range[i], wf[i],
            rules=False, plots=False, summary=False, output=True, site=site
            ))

# DATA: multivariate,  MODEL: vanilla #
model_type = 'vanilla'
LSTM_detect_multivar = \
    model_workflow.LSTM_detect_multivar(
        sensor_array, sensor,
        minimum, maximum, length,
        model_type, time_steps, samples, cells, dropout, patience,
        window_sz, alpha, min_range, wf,
        rules=False, plots=False, summary=False, output=True, site=site
        )

# DATA: multivariate,  MODEL: bidirectional #
model_type = 'bidirectional'
LSTM_detect_multivar_bidirectional = \
    model_workflow.LSTM_detect_multivar(
        sensor_array, sensor,
        minimum, maximum, length,
        model_type, time_steps, samples, cells, dropout, patience,
        window_sz, alpha, min_range, wf,
        rules=False, plots=False, summary=False, output=True, site=site
        )

#########################################
