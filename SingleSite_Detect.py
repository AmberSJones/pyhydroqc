#####################################################
### SINGLE SITE ANOMALY DETECTION AND CORRECTION
#####################################################

#### Import Libraries and Functions

import anomaly_utilities
import model_workflow
import rules_detect
from parameters import site_params, LSTM_params, calib_params
import pickle
import pandas as pd

#### Retrieve data
#########################################

site = 'MainStreet'
sensors = ['temp', 'cond', 'ph', 'do']
years = [2014, 2015, 2016, 2017, 2018, 2019]
df_full, sensor_array = anomaly_utilities.get_data(site, sensors, years, path="LRO_data/")

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
    sensor_array[snsr] = rules_detect.add_labels(sensor_array[snsr], -9999)
    sensor_array[snsr] = rules_detect.interpolate(sensor_array[snsr])
    # s = rules_detect.group_size(sensor_array[snsr])
    # size.append(s)
    # print(str(snsr) + ' longest detected group = ' + str(size))

    # metrics for rules based detection #
    df_rules_metrics = sensor_array[snsr]
    df_rules_metrics['labeled_event'] = anomaly_utilities.anomaly_events(df_rules_metrics['labeled_anomaly'], wf=0)
    df_rules_metrics['detected_event'] = anomaly_utilities.anomaly_events(df_rules_metrics['anomaly'], wf=0)
    anomaly_utilities.compare_events(df_rules_metrics, wf=0)
    rules_metrics[snsr] = anomaly_utilities.metrics(df_rules_metrics)
    print('\nRules based metrics')
    print('Sensor: ' + snsr)
    anomaly_utilities.print_metrics(rules_metrics[snsr])
    del(df_rules_metrics)

print('Rules based detection complete.\n')

#### Detect Calibration Events
#########################################

calib_sensors = sensors[1:4]
input_array = dict()
for snsr in calib_sensors:
    input_array[snsr] = sensor_array[snsr]
all_calib, all_calib_dates, df_all_calib, calib_dates_overlap = \
    rules_detect.calib_overlap(calib_sensors, input_array, calib_params)

#### Perform Linear Drift Correction
#########################################

calib_dates = dict()
for cal_snsr in calib_sensors:
    calib_dates[cal_snsr] = \
        pd.read_csv('LRO_data/' + site + '_' + cal_snsr + '_calib_dates.csv', header=1, parse_dates=True, infer_datetime_format=True)

    for i in range(0, len(calib_dates[cal_snsr])):
        result, sensor_array[cal_snsr]['observed'] = rules_detect.lin_drift_cor(
                sensor_array[cal_snsr]['observed'],
                calib_dates[cal_snsr]['start'][i],
                calib_dates[cal_snsr]['end'][i],
                calib_dates[cal_snsr]['gap'][i],
                replace=True)

#### Model Based Anomaly Detection
#########################################

##### ARIMA Detection
#########################################

ARIMA = dict()
for snsr in sensors:
    ARIMA[snsr] = model_workflow.ARIMA_detect(
            sensor_array[snsr], snsr, site_params[site][snsr],
            rules=False, plots=False, summary=False, output=True)
print('ARIMA detection complete.\n')

##### LSTM Detection
#########################################
###### DATA: univariate, MODEL: vanilla

model_type = 'vanilla'
LSTM_univar = dict()
for snsr in sensors:
    name = site + '_' + snsr
    LSTM_univar[snsr] = model_workflow.LSTM_detect_univar(
            sensor_array[snsr], snsr, site_params[site][snsr], LSTM_params, model_type, name,
            rules=False, plots=False, summary=False, output=True, model_output=False, model_save=True)

###### DATA: univariate,  MODEL: bidirectional

model_type = 'bidirectional'
LSTM_univar_bidir = dict()
for snsr in sensors:
    name = site + '_' + snsr
    LSTM_univar_bidir[snsr] = model_workflow.LSTM_detect_univar(
            sensor_array[snsr], snsr, site_params[site][snsr], LSTM_params, model_type, name,
            rules=False, plots=False, summary=False, output=True, model_output=False, model_save=True
        )

###### DATA: multivariate,  MODEL: vanilla

model_type = 'vanilla'
name = site
LSTM_multivar = model_workflow.LSTM_detect_multivar(
        sensor_array, sensors, site_params[site], LSTM_params, model_type, name,
        rules=False, plots=False, summary=False, output=True, model_output=False, model_save=True)

###### DATA: multivariate,  MODEL: bidirectional

model_type = 'bidirectional'
name = site
LSTM_multivar_bidir = model_workflow.LSTM_detect_multivar(
        sensor_array, sensors, site_params[site], LSTM_params, model_type, name,
        rules=False, plots=False, summary=False, output=True, model_output=False, model_save=True)

##### Aggregate Detections for All Models
#########################################

aggregate_results = dict()
aggregate_metrics = dict()
for snsr in sensors:
    results_all, metrics = anomaly_utilities.aggregate_results(
            sensor_array[snsr],
            ARIMA[snsr].df,
            LSTM_univar[snsr].df_anomalies,
            LSTM_univar_bidir[snsr].df_anomalies,
            LSTM_multivar.all_data[snsr],
            LSTM_multivar_bidir.all_data[snsr]
        )
    print('\nOverall metrics')
    print('Sensor: ' + snsr)
    anomaly_utilities.print_metrics(metrics)
    aggregate_results[snsr] = results_all
    aggregate_metrics[snsr] = metrics

#### Saving Output
#########################################

for snsr in sensors:
    ARIMA[snsr].df.to_csv(r'saved/ARIMA_df_' + site + '_' + snsr + '.csv')
    ARIMA[snsr].threshold.to_csv(r'saved/ARIMA_threshold_' + site + '_' + snsr + '.csv')
    ARIMA[snsr].detections.to_csv(r'saved/ARIMA_detections_' + site + '_' + snsr + '.csv')
    LSTM_univar[snsr].threshold.to_csv(r'saved/LSTM_univar_threshold_' + site + '_' + snsr + '.csv')
    LSTM_univar[snsr].detections.to_csv(r'saved/LSTM_univar_detections_' + site + '_' + snsr + '.csv')
    LSTM_univar[snsr].df_anomalies.to_csv(r'saved/LSTM_univar_df_anomalies_' + site + '_' + snsr + '.csv')
    LSTM_univar_bidir[snsr].threshold.to_csv(r'saved/LSTM_univar_bidir_threshold_' + site + '_' + snsr + '.csv')
    LSTM_univar_bidir[snsr].detections.to_csv(r'saved/LSTM_univar_bidir_detections_' + site + '_' + snsr + '.csv')
    LSTM_univar_bidir[snsr].df_anomalies.to_csv(r'saved/LSTM_univar_bidir_df_anomalies_' + site + '_' + snsr + '.csv')
    LSTM_multivar.threshold[snsr].to_csv(r'saved/LSTM_multivar_threshold_' + site + '_' + snsr + '.csv')
    LSTM_multivar.detections[snsr].to_csv(r'saved/LSTM_multivar_detections_' + site + '_' + snsr + '.csv')
    LSTM_multivar.all_data[snsr].to_csv(r'saved/LSTM_multivar_df_' + site + '_' + snsr + '.csv')
    LSTM_multivar_bidir.threshold[snsr].to_csv(r'saved/LSTM_multivar_bidir_threshold_' + site + '_' + snsr + '.csv')
    LSTM_multivar_bidir.detections[snsr].to_csv(r'saved/LSTM_multivar_bidir_detections_' + site + '_' + snsr + '.csv')
    LSTM_multivar_bidir.all_data[snsr].to_csv(r'saved/LSTM_multivar_bidir_df_' + site + '_' + snsr + '.csv')
    aggregate_results[snsr].to_csv(r'saved/aggregate_results_' + site + '_' + snsr + '.csv')

for snsr in sensors:
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
    pickle.dump(LSTM_multivar.metrics[snsr], pickle_out)
    pickle_out.close()
    pickle_out = open('saved/metrics_LSTM_multivar_bidir' + site + '_' + snsr, "wb")
    pickle.dump(LSTM_multivar_bidir.metrics[snsr], pickle_out)
    pickle_out.close()
    pickle_out = open('saved/metrics_aggregate' + site + '_' + snsr, "wb")
    pickle.dump(aggregate_metrics[snsr], pickle_out)
    pickle_out.close()
    pickle_out = open('saved/metrics_rules' + site + '_' + snsr, "wb")
    pickle.dump(rules_metrics[snsr], pickle_out)
    pickle_out.close()

print('Finished saving output.')

#### Correction
#########################################
