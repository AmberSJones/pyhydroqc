#####################################################
### SINGLE SITE ANOMALY DETECTION AND CORRECTION
#####################################################

#### Import Libraries and Functions

from pyhydroqc import anomaly_utilities
from pyhydroqc import model_workflow
from pyhydroqc import rules_detect
from pyhydroqc import arima_correct
from pyhydroqc import modeling_utilities
from pyhydroqc.model_workflow import ModelType

# Parameters may be specified in a parameters file or in the same script
from Examples.FB_parameters import site_params, LSTM_params, calib_params

#### Retrieve data
#########################################
site = 'FranklinBasin'
sensors = ['temp', 'cond', 'ph', 'do']
sensor_array = anomaly_utilities.get_data(sensors=sensors, filename='FB2017.csv', path="LRO_data/")

#### Rules Based Anomaly Detection
#########################################
range_count = dict()
persist_count = dict()
rules_metrics = dict()
for snsr in sensor_array:
    sensor_array[snsr], range_count[snsr] = rules_detect.range_check(
        df=sensor_array[snsr], maximum=site_params[site][snsr]['max_range'], minimum=site_params[site][snsr]['min_range'])
    sensor_array[snsr], persist_count[snsr] = rules_detect.persistence(
        df=sensor_array[snsr], length=site_params[site][snsr]['persist'], output_grp=True)
    sensor_array[snsr] = rules_detect.interpolate(df=sensor_array[snsr])
print('Rules based detection complete.\n')

#### Detect Calibration Events
#########################################
calib_sensors = sensors[1:4]
input_array = dict()
for snsr in calib_sensors:
    input_array[snsr] = sensor_array[snsr]
all_calib, all_calib_dates, df_all_calib, calib_dates_overlap = rules_detect.calib_overlap(
    sensor_names=calib_sensors, input_array=input_array, calib_params=calib_params)

#### Model Based Anomaly Detection
#########################################

##### ARIMA Detection
#########################################
all_pdq = dict()
for snsr in sensors:
    all_pdq[snsr] = modeling_utilities.pdq(data=sensor_array[snsr]['observed'])
    print(snsr + ' (p, d, q) = ' + str(all_pdq[snsr]))
    site_params[site][snsr]['pdq'] = all_pdq[snsr]

ARIMA = dict()
for snsr in sensors:
    ARIMA[snsr] = model_workflow.arima_detect(
        df=sensor_array[snsr], sensor=snsr, params=site_params[site][snsr],
        rules=False, plots=False, summary=False, compare=False)
print('ARIMA detection complete.\n')

##### LSTM Detection
#########################################
###### DATA: univariate, MODEL: vanilla

lstm_univar = dict()
for snsr in sensors:
    lstm_univar[snsr] = model_workflow.lstm_detect_univar(
            df=sensor_array[snsr], sensor=snsr, params=site_params[site][snsr], LSTM_params=LSTM_params, model_type=ModelType.VANILLA,
            rules=False, plots=False, summary=False, compare=False, model_output=False, model_save=False)

###### DATA: univariate,  MODEL: bidirectional

lstm_univar_bidir = dict()
for snsr in sensors:
    lstm_univar_bidir[snsr] = model_workflow.lstm_detect_univar(
            df=sensor_array[snsr], sensor=snsr, params=site_params[site][snsr], LSTM_params=LSTM_params, model_type=ModelType.BIDIRECTIONAL,
            rules=False, plots=False, summary=False, compare=False, model_output=False, model_save=False)

###### DATA: multivariate,  MODEL: vanilla

lstm_multivar = model_workflow.lstm_detect_multivar(
        sensor_array=sensor_array, sensors=sensors, params=site_params[site], LSTM_params=LSTM_params, model_type=ModelType.VANILLA,
        rules=False, plots=False, summary=False, compare=False, model_output=False, model_save=False)

###### DATA: multivariate,  MODEL: bidirectional

lstm_multivar_bidir = model_workflow.lstm_detect_multivar(
        sensor_array=sensor_array, sensors=sensors, params=site_params[site], LSTM_params=LSTM_params, model_type=ModelType.BIDIRECTIONAL,
        rules=False, plots=False, summary=False, compare=False, model_output=False, model_save=False)

##### Aggregate Detections for All Models
#########################################

results_all = dict()
for snsr in sensors:
    models = dict()
    models['ARIMA'] = ARIMA[snsr].df
    models['lstm_univar'] = lstm_univar[snsr].df_anomalies
    models['lstm_univar_bidir'] = lstm_univar_bidir[snsr].df_anomalies
    models['lstm_multivar'] = lstm_multivar.all_data[snsr]
    models['lstm_multivar_bidir'] = lstm_multivar_bidir.all_data[snsr]
    results_all[snsr] = anomaly_utilities.aggregate_results(
        df=sensor_array[snsr], models=models, verbose=True, compare=False)

#### Correction
#########################################

corrections = dict()
for snsr in sensors:
    corrections[snsr] = arima_correct.generate_corrections(
        df=results_all[snsr], observed='observed', anomalies='detected_event', savecasts=True)

############ PLOTTING ##############

import matplotlib.pyplot as plt
df = corrections[snsr]
plt.figure()
plt.plot(df['observed'], 'b', label='original data')
plt.plot(df['observed'][df['detected_event']], 'r+', mfc='none', label='machine detected anomalies')
plt.plot(df['det_cor'], 'm', label='determined_corrected')
plt.legend()
plt.ylabel(snsr)
plt.show()
