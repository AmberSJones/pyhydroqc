#####################################################
### SINGLE SITE ANOMALY DETECTION AND CORRECTION
#####################################################

#### Import Libraries and Functions

from PyHydroQC import anomaly_utilities
from PyHydroQC import model_workflow
from PyHydroQC import rules_detect
from PyHydroQC import ARIMA_correct
from PyHydroQC import modeling_utilities
from PyHydroQC.model_workflow import ModelType

# Parameters may be specified in a parameters file or in the same script
from Examples.FB_parameters import site_params, LSTM_params, calib_params

#### Retrieve data
#########################################
site = 'FranklinBasin'
sensors = ['temp', 'cond', 'ph', 'do']
sensor_array = anomaly_utilities.get_data(sensors, filename='FB2017.csv', path="LRO_data/")

#### Rules Based Anomaly Detection
#########################################
range_count = dict()
persist_count = dict()
rules_metrics = dict()
for snsr in sensor_array:
    sensor_array[snsr], range_count[snsr] = \
        rules_detect.range_check(sensor_array[snsr], site_params[site][snsr]['max_range'], site_params[site][snsr]['min_range'])
    sensor_array[snsr], persist_count[snsr] = \
        rules_detect.persistence(sensor_array[snsr], site_params[site][snsr]['persist'], output_grp=True)
    sensor_array[snsr] = rules_detect.interpolate(sensor_array[snsr])
print('Rules based detection complete.\n')

#### Detect Calibration Events
#########################################
calib_sensors = sensors[1:4]
input_array = dict()
for snsr in calib_sensors:
    input_array[snsr] = sensor_array[snsr]
all_calib, all_calib_dates, df_all_calib, calib_dates_overlap = \
    rules_detect.calib_overlap(calib_sensors, input_array, calib_params)

#### Model Based Anomaly Detection
#########################################

##### ARIMA Detection
#########################################
all_pdq = dict()
for snsr in sensors:
    all_pdq[snsr] = modeling_utilities.pdq(sensor_array[snsr]['observed'])
    print(snsr + ' (p, d, q) = ' + str(all_pdq[snsr]))
    site_params[site][snsr]['pdq'] = all_pdq[snsr]

ARIMA = dict()
for snsr in sensors:
    ARIMA[snsr] = model_workflow.ARIMA_detect(sensor_array[snsr], snsr, site_params[site][snsr],
                                              rules=False, plots=False, summary=False, compare=False)
print('ARIMA detection complete.\n')

##### LSTM Detection
#########################################
###### DATA: univariate, MODEL: vanilla

LSTM_univar = dict()
for snsr in sensors:
    LSTM_univar[snsr] = model_workflow.LSTM_detect_univar(
            sensor_array[snsr], snsr, site_params[site][snsr], LSTM_params, model_type=ModelType.VANILLA,
            rules=False, plots=False, summary=False, compare=False, model_output=False, model_save=False)

###### DATA: univariate,  MODEL: bidirectional

LSTM_univar_bidir = dict()
for snsr in sensors:
    LSTM_univar_bidir[snsr] = model_workflow.LSTM_detect_univar(
            sensor_array[snsr], snsr, site_params[site][snsr], LSTM_params, model_type=ModelType.BIDIRECTIONAL,
            rules=False, plots=False, summary=False, compare=False, model_output=False, model_save=False)

###### DATA: multivariate,  MODEL: vanilla

LSTM_multivar = model_workflow.LSTM_detect_multivar(
        sensor_array, sensors, site_params[site], LSTM_params, model_type=ModelType.VANILLA,
        rules=False, plots=False, summary=False, compare=False, model_output=False, model_save=False)

###### DATA: multivariate,  MODEL: bidirectional

LSTM_multivar_bidir = model_workflow.LSTM_detect_multivar(
        sensor_array, sensors, site_params[site], LSTM_params, model_type=ModelType.BIDIRECTIONAL,
        rules=False, plots=False, summary=False, compare=False, model_output=False, model_save=False)

##### Aggregate Detections for All Models
#########################################

aggregate_results = dict()
for snsr in sensors:
    models = dict()
    models['ARIMA'] = ARIMA[snsr].df
    models['LSTM_univar'] = LSTM_univar[snsr].df_anomalies
    models['LSTM_univar_bidir'] = LSTM_univar_bidir[snsr].df_anomalies
    models['LSTM_multivar'] = LSTM_multivar.all_data[snsr]
    models['LSTM_multivar_bidir'] = LSTM_multivar_bidir.all_data[snsr]
    results_all = anomaly_utilities.aggregate_results(sensor_array[snsr], models, verbose=True, compare=True)
    aggregate_results[snsr] = results_all

#### Correction
#########################################

corrections = dict()
for snsr in sensors:
    corrections[snsr] = ARIMA_correct.generate_corrections(aggregate_results[snsr], 'observed', 'detected_event')

############ PLOTTING ##############

import matplotlib.pyplot as plt
plt.figure()
plt.plot(df['observed'], 'b', label='original data')
# plt.plot(df['cor'], 'c', label='technician corrected')
plt.plot(df['observed'][df['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
plt.plot(df['observed'][df['detected_event']], 'r+', mfc='none', label='machine detected anomalies')
plt.plot(df['det_cor'], 'm', label='determined_corrected')
plt.legend()
plt.ylabel('do')
plt.show()
