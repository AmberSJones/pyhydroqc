#####################################################
### SINGLE SITE ANOMALY DETECTION AND CORRECTION
#####################################################

#### Import Libraries and Functions

from PyHydroQC import anomaly_utilities
from PyHydroQC import model_workflow
from PyHydroQC import rules_detect
from PyHydroQC import ARIMA_correct
from PyHydroQC.parameters import site_params, LSTM_params, calib_params
from PyHydroQC.model_workflow import ModelType
import pickle
import pandas as pd

#### Retrieve data
#########################################

site = 'MainStreet'
sensors = ['temp', 'cond', 'ph', 'do']
years = [2015, 2016, 2017, 2018, 2019]
sensor_array = anomaly_utilities.get_data(sensors=sensors, site=site, years=years, path="./LRO_data/")

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
    sensor_array[snsr] = rules_detect.add_labels(df=sensor_array[snsr], value=-9999)
    sensor_array[snsr] = rules_detect.interpolate(df=sensor_array[snsr])
    # s = rules_detect.group_size(df=sensor_array[snsr])
    # size.append(s)
    # print(str(snsr) + ' longest detected group = ' + str(size))

# metrics for rules based detection #
for snsr in sensor_array:
    df_rules_metrics = sensor_array[snsr]
    df_rules_metrics['labeled_event'] = anomaly_utilities.anomaly_events(anomaly=df_rules_metrics['labeled_anomaly'], wf=0)
    df_rules_metrics['detected_event'] = anomaly_utilities.anomaly_events(anomaly=df_rules_metrics['anomaly'], wf=0)
    anomaly_utilities.compare_events(df=df_rules_metrics, wf=0)
    rules_metrics[snsr] = anomaly_utilities.metrics(df=df_rules_metrics)
    print('\nRules based metrics')
    print('Sensor: ' + snsr)
    anomaly_utilities.print_metrics(df=rules_metrics[snsr])
    del(df_rules_metrics)

print('Rules based detection complete.\n')

#### Detect Calibration Events
#########################################

calib_sensors = sensors[1:4]
input_array = dict()
for snsr in calib_sensors:
    input_array[snsr] = sensor_array[snsr]
all_calib, all_calib_dates, df_all_calib, calib_dates_overlap = rules_detect.calib_overlap(
    sensor_names=calib_sensors, input_array=input_array, calib_params=calib_params)

#### Perform Linear Drift Correction
#########################################

calib_sensors = sensors[1:4]
calib_dates = dict()
for cal_snsr in calib_sensors:
    calib_dates[cal_snsr] = pd.read_csv(
        './LRO_data/' + site + '_' + cal_snsr + '_calib_dates.csv', header=1, parse_dates=True, infer_datetime_format=True)
    calib_dates[cal_snsr]['start'] = pd.to_datetime(calib_dates[cal_snsr]['start'])
    calib_dates[cal_snsr]['end'] = pd.to_datetime(calib_dates[cal_snsr]['end'])
    calib_dates[cal_snsr] = calib_dates[cal_snsr].loc[(calib_dates[cal_snsr]['start'] > min(sensor_array[cal_snsr].index)) &
                                                      (calib_dates[cal_snsr]['start'] < max(sensor_array[cal_snsr].index))]
    if len(calib_dates[cal_snsr]) > 0:
        for i in range(min(calib_dates[cal_snsr].index), max(calib_dates[cal_snsr].index)):
            result, sensor_array[cal_snsr]['observed'] = rules_detect.lin_drift_cor(
                                                            observed=sensor_array[cal_snsr]['observed'],
                                                            start=calib_dates[cal_snsr]['start'][i],
                                                            end=calib_dates[cal_snsr]['end'][i],
                                                            gap=calib_dates[cal_snsr]['gap'][i],
                                                            replace=True)

#### Model Based Anomaly Detection
#########################################

##### ARIMA Detection
#########################################

ARIMA = dict()
for snsr in sensors:
    ARIMA[snsr] = model_workflow.ARIMA_detect(
            df=sensor_array[snsr], sensor=snsr, params=site_params[site][snsr],
            rules=False, plots=False, summary=False, compare=True)
print('ARIMA detection complete.\n')

##### LSTM Detection
#########################################
###### DATA: univariate, MODEL: vanilla

LSTM_univar = dict()
for snsr in sensors:
    name = site + '_' + snsr
    LSTM_univar[snsr] = model_workflow.LSTM_detect_univar(
            df=sensor_array[snsr], sensor=snsr, params=site_params[site][snsr], LSTM_params=LSTM_params, model_type=ModelType.VANILLA, name=name,
            rules=False, plots=False, summary=False, compare=True, model_output=False, model_save=False)

###### DATA: univariate,  MODEL: bidirectional

LSTM_univar_bidir = dict()
for snsr in sensors:
    name = site + '_' + snsr
    LSTM_univar_bidir[snsr] = model_workflow.LSTM_detect_univar(
            df=sensor_array[snsr], sensor=snsr, params=site_params[site][snsr], LSTM_params=LSTM_params, model_type=ModelType.BIDIRECTIONAL, name=name,
            rules=False, plots=False, summary=False,compare=True, model_output=False, model_save=False)

###### DATA: multivariate,  MODEL: vanilla

name = site
LSTM_multivar = model_workflow.LSTM_detect_multivar(
        sensor_array=sensor_array, sensors=sensors, params=site_params[site], LSTM_params=LSTM_params, model_type=ModelType.VANILLA, name=name,
        rules=False, plots=False, summary=False, compare=True, model_output=False, model_save=False)

###### DATA: multivariate,  MODEL: bidirectional

name = site
LSTM_multivar_bidir = model_workflow.LSTM_detect_multivar(
        sensor_array=sensor_array, sensors=sensors, params=site_params[site], LSTM_params=LSTM_params, model_type=ModelType.BIDIRECTIONAL, name=name,
        rules=False, plots=False, summary=False, compare=True, model_output=False, model_save=False)

##### Aggregate Detections for All Models
#########################################

results_all = dict()
metrics_all = dict()
for snsr in ['temp','do']:
    models = dict()
    models['ARIMA'] = ARIMA[snsr].df
    models['LSTM_univar'] = LSTM_univar[snsr].df_anomalies
    models['LSTM_univar_bidir'] = LSTM_univar_bidir[snsr].df_anomalies
    models['LSTM_multivar'] = LSTM_multivar.all_data[snsr]
    models['LSTM_multivar_bidir'] = LSTM_multivar_bidir.all_data[snsr]
    results_all[snsr], metrics_all[snsr] = anomaly_utilities.aggregate_results(
        df=sensor_array[snsr], models=models, verbose=True, compare=True)
    print('\nOverall metrics')
    print('Sensor: ' + snsr)
    anomaly_utilities.print_metrics(metrics_all[snsr])

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
    results_all[snsr].to_csv(r'saved/aggregate_results_' + site + '_' + snsr + '.csv')

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
    pickle.dump(results_all[snsr], pickle_out)
    pickle_out.close()
    pickle_out = open('saved/metrics_rules' + site + '_' + snsr, "wb")
    pickle.dump(rules_metrics[snsr], pickle_out)
    pickle_out.close()

print('Finished saving output.')

#### Correction
#########################################

corrections = dict()
for snsr in sensors:
    corrections[snsr] = ARIMA_correct.generate_corrections(
        df=results_all[snsr], observed='observed', anomalies='detected_event', savecasts=True)

# Saving corrections
for snsr in sensors:
    corrections[snsr].to_csv(r'Examples/' + site + '_' + snsr + '_corrections.csv')


# Plotting corrections

import matplotlib.pyplot as plt

df = corrections[snsr]
plt.figure()
plt.plot(df['observed'], 'b', label='original data')
# plt.plot(df['cor'], 'c', label='technician corrected')
plt.plot(df['det_cor'], 'c', label='predicted values')
plt.plot(df['forecasts'], 'g', label='forecasted')
plt.plot(df['backcasts'], 'r', label='backcasted')
# plt.plot(df['observed'][df['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
plt.legend()
plt.ylabel(snsr)
plt.show()