#####################################################
### SINGLE SITE ANOMALY DETECTION AND CORRECTION
#####################################################

#### Import Libraries and Functions

import anomaly_utilities
import model_workflow
import rules_detect
from parameters import site_params, LSTM_params, calib_params
import copy
import pickle
import pandas as pd

#### Retrieve data
#########################################

site = 'MS'
sensor = ['temp', 'cond', 'ph', 'do']
year = [2014, 2015, 2016, 2017, 2018, 2019]
df_full, sensor_array = anomaly_utilities.get_data(site, sensor, year, path="LRO_data/")

#### Rules Based Anomaly Detection
#########################################

range_count = []
persist_count = []
rules_metrics = []
for snsr in sensor_array:
    sensor_array[snsr], r_c = \
        rules_detect.range_check(sensor_array[snsr], site_params[site][snsr].max_range, site_params[site][snsr].min_range)
    range_count.append(r_c)
    sensor_array[snsr], p_c = \
        rules_detect.persistence(sensor_array[snsr], site_params[site][snsr].persist, output_grp=True)
    persist_count.append(p_c)
    # s = rules_detect.group_size(sensor_array[snsr])
    # size.append(s)
    sensor_array[snsr] = rules_detect.add_labels(sensor_array[snsr], -9999)
    sensor_array[snsr] = rules_detect.interpolate(sensor_array[snsr])
    # print(str(sensor) + ' longest detected group = ' + str(size))

    # metrics for rules based detection #
    df_rules_metrics = sensor_array[snsr]
    df_rules_metrics['labeled_event'] = anomaly_utilities.anomaly_events(df_rules_metrics['labeled_anomaly'], wf=0)
    df_rules_metrics['detected_event'] = anomaly_utilities.anomaly_events(df_rules_metrics['anomaly'], wf=0)
    anomaly_utilities.compare_events(df_rules_metrics, wf=0)

    rules_metrics_object = anomaly_utilities.metrics(df_rules_metrics)
    print('\nRules based metrics')
    print('Sensor: ' + snsr)
    anomaly_utilities.print_metrics(rules_metrics_object)
    rules_metrics.append(rules_metrics_object)

print('Rules based detection complete.\n')


#### Detect Calibration Events
#########################################

sensor_names = sensor[1:4]
input_array = []
for snsr in sensor_names:
    input_array.append(sensor_array[snsr])
input_array = dict(zip(sensor_names, input_array))

all_calib, all_calib_dates, df_all_calib, calib_dates_overlap = \
    rules_detect.calib_overlap(sensor_names, input_array, calib_params)


#### Perform Linear Drift Correction
#########################################

calib_dates = dict()
for snsr in sensor_names:
    calib_dates[snsr] = \
        pd.read_csv('LRO_data/' + site + '_' + snsr + '_calib_dates.csv', header=1, parse_dates=True, infer_datetime_format=True)

    for i in range(0, len(calib_dates[snsr])):
        result, sensor_array[snsr]['observed'] = rules_detect.lin_drift_cor(
                sensor_array[snsr]['observed'],
                calib_dates[snsr]['start'][i],
                calib_dates[snsr]['end'][i],
                calib_dates[snsr]['gap'][i],
                replace=True)


#### Model Based Anomaly Detection
#########################################

##### ARIMA Detection
#########################################

ARIMA = dict()
for snsr in sensor:
    ARIMA[snsr] = model_workflow.ARIMA_detect(
            sensor_array[snsr], snsr, site_params[site][snsr],
            rules=False, plots=False, summary=False, output=True, site=site
        )
print('ARIMA detection complete.\n')

##### LSTM Detection
#########################################
###### DATA: univariate, MODEL: vanilla

model_type = 'vanilla'
LSTM_univar = dict()
for snsr in sensor:
    name = site + '_' + snsr
    LSTM_univar[snsr] = model_workflow.LSTM_detect_univar(
            sensor_array[snsr], snsr, site_params[site][snsr], LSTM_params, model_type, name,
            rules=False, plots=False, summary=False, output=True, site=site, model_output=False, model_save=True
        )

###### DATA: univariate,  MODEL: bidirectional

model_type = 'bidirectional'
LSTM_univar_bidir = dict()
for snsr in sensor:
    name = site + '_' + snsr
    LSTM_univar_bidir[snsr] = model_workflow.LSTM_detect_univar(
            sensor_array[snsr], snsr, site_params[site][snsr], LSTM_params, model_type, name,
            rules=False, plots=False, summary=False, output=True, site=site, model_output=False, model_save=True
        )

###### DATA: multivariate,  MODEL: vanilla

model_type = 'vanilla'
name = site
LSTM_multivar = model_workflow.LSTM_detect_multivar(
        sensor_array, sensor, site_params[site], LSTM_params, model_type, name,
        rules=False, plots=False, summary=False, output=True, site=site, model_output=False, model_save=True
    )

###### DATA: multivariate,  MODEL: bidirectional

model_type = 'bidirectional'
name = site
LSTM_multivar_bidir = model_workflow.LSTM_detect_multivar(
        sensor_array, sensor, site_params[site], LSTM_params, model_type, name,
        rules=False, plots=False, summary=False, output=True, site=site, model_output=False, model_save=True
    )

##### Aggregate Detections for All Models
#########################################

aggregate_results = dict()
aggregate_metrics = dict()
for snsr in sensor:
    results_all, metrics = anomaly_utilities.aggregate_results(
            sensor_array[snsr],
            ARIMA[snsr].df,
            LSTM_univar[snsr].df_anomalies,
            LSTM_univar_bidir[snsr].df_anomalies,
            LSTM_multivar.df_array[snsr],
            LSTM_multivar_bidir.df_array[snsr]
        )

    print('\nOverall metrics')
    print('Sensor: ' + snsr)
    anomaly_utilities.print_metrics(metrics)
    aggregate_results[snsr] = results_all
    aggregate_metrics[snsr] = metrics


#### Saving Output
#########################################

for snsr in sensor:
    ARIMA[snsr].df.to_csv(r'saved/ARIMA_df_' + site + '_' + snsr + '.csv')
    ARIMA[snsr].threshold.to_csv(r'saved/ARIMA_threshold_' + site + '_' + snsr + '.csv')
    ARIMA[snsr].detections.to_csv(r'saved/ARIMA_detections_' + site + '_' + snsr + '.csv')
    LSTM_univar[snsr].threshold.to_csv(r'saved/LSTM_univar_threshold_' + site + '_' + snsr + '.csv')
    LSTM_univar[snsr].detections.to_csv(r'saved/LSTM_univar_detections_' + site + '_' + snsr + '.csv')
    LSTM_univar[snsr].df_anomalies.to_csv(r'saved/LSTM_univar_df_anomalies_' + site + '_' + snsr + '.csv')
    LSTM_univar_bidir[snsr].threshold.to_csv(r'saved/LSTM_univar_bidir_threshold_' + site + '_' + snsr + '.csv')
    LSTM_univar_bidir[snsr].detections.to_csv(
        r'saved/LSTM_univar_bidir_detections_' + site + '_' + snsr + '.csv')
    LSTM_univar_bidir[snsr].df_anomalies.to_csv(
        r'saved/LSTM_univar_bidir_df_anomalies_' + site + '_' + snsr + '.csv')
    LSTM_multivar.threshold[snsr].to_csv(r'saved/LSTM_multivar_threshold_' + site + '_' + snsr + '.csv')
    LSTM_multivar.detections_array[snsr].to_csv(r'saved/LSTM_multivar_detections_' + site + '_' + snsr + '.csv')
    LSTM_multivar.df_array[snsr].to_csv(r'saved/LSTM_multivar_df_' + site + '_' + snsr + '.csv')
    LSTM_multivar.threshold[snsr].to_csv(r'saved/LSTM_multivar_threshold_' + site + '_' + snsr + '.csv')
    LSTM_multivar_bidir.detections_array[snsr].to_csv(
        r'saved/LSTM_multivar_bidir_detections_' + site + '_' + snsr + '.csv')
    LSTM_multivar_bidir.df_array[snsr].to_csv(r'saved/LSTM_multivar_bidir_df_' + site + '_' + snsr + '.csv')
    aggregate_results[snsr].to_csv(r'saved/aggregate_results_' + site + '_' + snsr + '.csv')

for i in range(0, len(sensor)):
    pickle_out = open('saved/metrics_ARIMA_' + site + '_' + snsr, "wb")
    pickle.dump(ARIMA[snsr].metrics, pickle_out)
    pickle_out.close()
    pickle_out = open('saved/metrics_LSTM_univar_' + site + '_' + snsr, "wb")
    pickle.dump(LSTM_univar[snsr].metrics, pickle_out)
    pickle_out.close()
    pickle_out = open('saved/metrics_LSTM_univar_bidir' + site + '_' + snsr, "wb")
    pickle.dump(LSTM_univar_bidir[snsr].metrics, pickle_out)
    pickle_out.close()
    pickle_out = open('saved/metrics_LSTM_multivar' + site + '_' + snsr, "wb")
    pickle.dump(LSTM_multivar.metrics_array[snsr], pickle_out)
    pickle_out.close()
    pickle_out = open('saved/metrics_LSTM_multivar_bidir' + site + '_' + snsr, "wb")
    pickle.dump(LSTM_multivar_bidir.metrics_array[snsr], pickle_out)
    pickle_out.close()
    pickle_out = open('saved/metrics_aggregate' + site + '_' + snsr, "wb")
    pickle.dump(aggregate_metrics[snsr], pickle_out)
    pickle_out.close()
    pickle_out = open('saved/metrics_rules' + site + '_' + snsr, "wb")
    pickle.dump(rules_metrics[snsr], pickle_out)
    pickle_out.close()

print('Finished saving output.')
