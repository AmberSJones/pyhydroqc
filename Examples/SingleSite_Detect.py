#####################################################
### SINGLE SITE ANOMALY DETECTION AND CORRECTION
#####################################################

#### Import Libraries and Functions
from PyHydroQC import anomaly_utilities
from PyHydroQC import model_workflow
from PyHydroQC import rules_detect
from PyHydroQC import ARIMA_correct
from PyHydroQC import calibration
from PyHydroQC.parameters import site_params, LSTM_params, calib_params
from PyHydroQC.model_workflow import ModelType
import math
import pickle
import pandas as pd
import matplotlib.pyplot as plt

#### Retrieve data
#########################################
site = 'MainStreet'
sensors = ['temp', 'cond', 'ph', 'do']
years = [2017]
sensor_array = anomaly_utilities.get_data(sensors=sensors, site=site, years=years, path="./LRO_data/")

#### Rules Based Anomaly Detection
#########################################
range_count = dict()
persist_count = dict()
rules_metrics = dict()
for snsr in sensor_array:
    sensor_array[snsr], range_count[snsr] = rules_detect.range_check(df=sensor_array[snsr],
                                                                     maximum=site_params[site][snsr].max_range,
                                                                     minimum=site_params[site][snsr].min_range)
    sensor_array[snsr], persist_count[snsr] = rules_detect.persistence(df=sensor_array[snsr],
                                                                       length=site_params[site][snsr].persist,
                                                                       output_grp=True)
    sensor_array[snsr] = rules_detect.interpolate(df=sensor_array[snsr])

# # metrics for rules based detection #
# for snsr in sensor_array:
#     df_rules_metrics = sensor_array[snsr]
#     df_rules_metrics['labeled_event'] = anomaly_utilities.anomaly_events(anomaly=df_rules_metrics['labeled_anomaly'], wf=0)
#     df_rules_metrics['detected_event'] = anomaly_utilities.anomaly_events(anomaly=df_rules_metrics['anomaly'], wf=0)
#     anomaly_utilities.compare_events(df=df_rules_metrics, wf=0)
#     rules_metrics[snsr] = anomaly_utilities.metrics(df=df_rules_metrics)
#     print('\nRules based metrics')
#     print('Sensor: ' + snsr)
#     anomaly_utilities.print_metrics(metrics=rules_metrics[snsr])
#     del(df_rules_metrics)

print('Rules based detection complete.\n')

#### Detect Calibration Events
#########################################
# Subset of sensors that are calibrated
calib_sensors = sensors[1:4]

# Using concurrent persistence
input_array = dict()
for snsr in calib_sensors:
    input_array[snsr] = sensor_array[snsr]
all_calib, all_calib_dates, df_all_calib, calib_dates_overlap = calibration.calib_overlap(sensor_names=calib_sensors,
                                                                                          input_array=input_array,
                                                                                          calib_params=calib_params)
# Using edge detection
calib_candidates = dict()
edge_diff = dict()
# The threshold for each variable (the level of change for a difference to be identified as a calibration event)
# is set in the parameters. Finding the threshold can be an iterative process.
for snsr in calib_sensors:
    # Width is the window of time to consider in the edge detect difference.
    # 1 determines the difference between each point independently.
    # Higher numbers use the difference over a longer window.
    calib_candidates[snsr], edge_diff[snsr] = calibration.calib_edge_detect(observed=sensor_array[snsr]['observed'],
                                                                            width=1,
                                                                            calib_params=calib_params,
                                                                            threshold=site_params[site][snsr].calib_threshold)
### Find Gap Values
#########################################
# Subset of sensors that are calibrated
calib_sensors = sensors[1:4]
# Initialize data structures
calib_dates = dict()
gaps = dict()
shifts = dict()
tech_shifts = dict()

for cal_snsr in calib_sensors:
    # Import calibration dates
    calib_dates[cal_snsr] = pd.read_csv('./LRO_data/' + site + '_' + cal_snsr + '_calib_dates.csv',
                                        header=1, parse_dates=True, infer_datetime_format=True)
    calib_dates[cal_snsr]['start'] = pd.to_datetime(calib_dates[cal_snsr]['start'])
    calib_dates[cal_snsr]['end'] = pd.to_datetime(calib_dates[cal_snsr]['end'])
    # Ensure date range of calibrations correspond to imported data
    calib_dates[cal_snsr] = calib_dates[cal_snsr].loc[(calib_dates[cal_snsr]['start'] > min(sensor_array[cal_snsr].index)) &
                                                      (calib_dates[cal_snsr]['end'] < max(sensor_array[cal_snsr].index))]
    # Initialize dataframe to store determined gap values and associated dates
    gaps[cal_snsr] = pd.DataFrame(columns=['end', 'gap'],
                                  index=range(min(calib_dates[cal_snsr].index), max(calib_dates[cal_snsr].index)+1))
    if len(calib_dates[cal_snsr]) > 0:
        # Initialize data structures
        shifts[cal_snsr] = []
        tech_shifts[cal_snsr] = []
        # Loop through each calibration event date.
        for i in range(min(calib_dates[cal_snsr].index), max(calib_dates[cal_snsr].index)+1):
            # Apply find_gap routine, add to dataframe, add output of shifts to list.
            gap, end, shifted = calibration.find_gap(observed=sensor_array[cal_snsr]['observed'],
                                                      calib_date=calib_dates[cal_snsr]['end'][i],
                                                      hours=2,
                                                      show_shift=True)
            gaps[cal_snsr].loc[i]['end'] = end
            gaps[cal_snsr].loc[i]['gap'] = gap
            shifts[cal_snsr].append(shifted)

            # Create subsets to show the shifts of the technician selected gap values for comparison
            tech_subset = sensor_array[cal_snsr]['observed'].loc[
                                             pd.to_datetime(calib_dates[cal_snsr]['end'][i]) - pd.Timedelta(hours=2):
                                             pd.to_datetime(calib_dates[cal_snsr]['end'][i]) + pd.Timedelta(hours=2)]
            tech_shifted = tech_subset.loc[
                                   tech_subset.index[0]:
                                   pd.to_datetime(calib_dates[cal_snsr]['end'][i])] + calib_dates[cal_snsr]['gap'][i]
            tech_shifts[cal_snsr].append(tech_shifted)

    # Plotting to examine when calibrations occur and compare algorithm and technician gap values
    l = len(calib_dates[cal_snsr])
    ncol = math.ceil(math.sqrt(l))
    nrow = math.ceil(l/ncol)
    hours = 6

    fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(nrow, ncol+1), facecolor='w')
    for i, axi in enumerate(ax.flat):
        if i < l:
            axi.plot(sensor_array[cal_snsr]['observed'].loc[
                    pd.to_datetime(calib_dates[cal_snsr]['end'].iloc[i]) - pd.Timedelta(hours=hours):
                    pd.to_datetime(calib_dates[cal_snsr]['end'].iloc[i]) + pd.Timedelta(hours=hours)
                    ])
            axi.plot(shifts[cal_snsr][i], 'c')
            axi.plot(tech_shifts[cal_snsr][i], 'r')

# # Review gaps and make adjustments as needed before performing drift correction
# gaps['cond'].loc[3, 'gap'] = 4
# gaps['cond'].loc[4, 'gap'] = 10
# gaps['cond'].loc[21, 'gap'] = 0
# gaps['cond'].loc[39, 'gap'] = -5
# gaps['cond'].loc[41, 'gap'] = 4
# gaps['ph'].loc[33, 'gap'] = -0.04
# gaps['ph'].loc[43, 'gap'] = 0.12
# gaps['ph'].loc[43, 'end'] = '2019-08-15 15:00'

#### Perform Linear Drift Correction
#########################################
calib_sensors = sensors[1:4]
for snsr in calib_sensors:
    # Set start dates for drift correction at the previously identified calibration 
    # For the first calibration - use one month back or go to the first date in the series.
    gaps[snsr]['start'] = gaps[snsr]['end'].shift(1)
    # gaps[snsr]['start'].iloc[0] = sensor_array[snsr].index[0]
    gaps[snsr]['start'].iloc[0] = gaps[snsr]['end'].iloc[0] - pd.Timedelta(days=30)
    if len(gaps[snsr]) > 0:
        for i in range(min(gaps[snsr].index), max(gaps[snsr].index) + 1):
            result, sensor_array[snsr]['observed'] = calibration.lin_drift_cor(observed=sensor_array[snsr]['observed'],
                                                                                   start=gaps[snsr]['start'][i],
                                                                                   end=gaps[snsr]['end'][i],
                                                                                   gap=gaps[snsr]['gap'][i],
                                                                                   replace=True)

#### Model Based Anomaly Detection
#########################################

##### ARIMA Detection
#########################################
ARIMA = dict()
for snsr in sensors:
    ARIMA[snsr] = model_workflow.ARIMA_detect(
            df=sensor_array[snsr], sensor=snsr, params=site_params[site][snsr],
            rules=False, plots=False, summary=False, compare=False)
print('ARIMA detection complete.\n')

##### LSTM Detection
#########################################
###### DATA: univariate, MODEL: vanilla
LSTM_univar = dict()
for snsr in sensors:
    name = site + '_' + snsr
    LSTM_univar[snsr] = model_workflow.LSTM_detect_univar(
            df=sensor_array[snsr], sensor=snsr, params=site_params[site][snsr], LSTM_params=LSTM_params, model_type=ModelType.VANILLA, name=name,
            rules=False, plots=False, summary=False, compare=False, model_output=False, model_save=False)

###### DATA: univariate,  MODEL: bidirectional
LSTM_univar_bidir = dict()
for snsr in sensors:
    name = site + '_' + snsr
    LSTM_univar_bidir[snsr] = model_workflow.LSTM_detect_univar(
            df=sensor_array[snsr], sensor=snsr, params=site_params[site][snsr], LSTM_params=LSTM_params, model_type=ModelType.BIDIRECTIONAL, name=name,
            rules=False, plots=False, summary=False,compare=False, model_output=False, model_save=False)

###### DATA: multivariate,  MODEL: vanilla
name = site
LSTM_multivar = model_workflow.LSTM_detect_multivar(
        sensor_array=sensor_array, sensors=sensors, params=site_params[site], LSTM_params=LSTM_params, model_type=ModelType.VANILLA, name=name,
        rules=False, plots=False, summary=False, compare=False, model_output=False, model_save=False)

###### DATA: multivariate,  MODEL: bidirectional
name = site
LSTM_multivar_bidir = model_workflow.LSTM_detect_multivar(
        sensor_array=sensor_array, sensors=sensors, params=site_params[site], LSTM_params=LSTM_params, model_type=ModelType.BIDIRECTIONAL, name=name,
        rules=False, plots=False, summary=False, compare=False, model_output=False, model_save=False)

##### Aggregate Detections for All Models
#########################################
results_all = dict()
metrics_all = dict()
for snsr in sensors:
    models = dict()
    models['ARIMA'] = ARIMA[snsr].df
    models['LSTM_univar'] = LSTM_univar[snsr].df_anomalies
    models['LSTM_univar_bidir'] = LSTM_univar_bidir[snsr].df_anomalies
    models['LSTM_multivar'] = LSTM_multivar.all_data[snsr]
    models['LSTM_multivar_bidir'] = LSTM_multivar_bidir.all_data[snsr]
    results_all[snsr] = anomaly_utilities.aggregate_results(df=sensor_array[snsr],
                                                            models=models,
                                                            verbose=True,
                                                            compare=False)
    # results_all[snsr], metrics_all[snsr] = anomaly_utilities.aggregate_results(df=sensor_array[snsr],
    #                                                                            models=models,
    #                                                                            verbose=True,
    #                                                                            compare=True)
    # print('\nOverall metrics')
    # print('Sensor: ' + snsr)
    # anomaly_utilities.print_metrics(metrics_all[snsr])

# #### Saving Output
# #########################################
# for snsr in sensors:
#     ARIMA[snsr].df.to_csv(r'saved/ARIMA_df_' + site + '_' + snsr + '.csv')
#     ARIMA[snsr].threshold.to_csv(r'saved/ARIMA_threshold_' + site + '_' + snsr + '.csv')
#     ARIMA[snsr].detections.to_csv(r'saved/ARIMA_detections_' + site + '_' + snsr + '.csv')
#     LSTM_univar[snsr].threshold.to_csv(r'saved/LSTM_univar_threshold_' + site + '_' + snsr + '.csv')
#     LSTM_univar[snsr].detections.to_csv(r'saved/LSTM_univar_detections_' + site + '_' + snsr + '.csv')
#     LSTM_univar[snsr].df_anomalies.to_csv(r'saved/LSTM_univar_df_anomalies_' + site + '_' + snsr + '.csv')
#     LSTM_univar_bidir[snsr].threshold.to_csv(r'saved/LSTM_univar_bidir_threshold_' + site + '_' + snsr + '.csv')
#     LSTM_univar_bidir[snsr].detections.to_csv(r'saved/LSTM_univar_bidir_detections_' + site + '_' + snsr + '.csv')
#     LSTM_univar_bidir[snsr].df_anomalies.to_csv(r'saved/LSTM_univar_bidir_df_anomalies_' + site + '_' + snsr + '.csv')
#     LSTM_multivar.threshold[snsr].to_csv(r'saved/LSTM_multivar_threshold_' + site + '_' + snsr + '.csv')
#     LSTM_multivar.detections[snsr].to_csv(r'saved/LSTM_multivar_detections_' + site + '_' + snsr + '.csv')
#     LSTM_multivar.all_data[snsr].to_csv(r'saved/LSTM_multivar_df_' + site + '_' + snsr + '.csv')
#     LSTM_multivar_bidir.threshold[snsr].to_csv(r'saved/LSTM_multivar_bidir_threshold_' + site + '_' + snsr + '.csv')
#     LSTM_multivar_bidir.detections[snsr].to_csv(r'saved/LSTM_multivar_bidir_detections_' + site + '_' + snsr + '.csv')
#     LSTM_multivar_bidir.all_data[snsr].to_csv(r'saved/LSTM_multivar_bidir_df_' + site + '_' + snsr + '.csv')
#     results_all[snsr].to_csv(r'saved/aggregate_results_' + site + '_' + snsr + '.csv')
# for snsr in sensors:
#     pickle_out = open('saved/metrics_ARIMA_' + site + '_' + snsr, "wb")
#     pickle.dump(ARIMA[snsr].metrics, pickle_out)
#     pickle_out.close()
#     pickle_out = open('saved/metrics_LSTM_univar_' + site + '_' + snsr, "wb")
#     pickle.dump(LSTM_univar[snsr].metrics, pickle_out)
#     pickle_out.close()
#     pickle_out = open('saved/metrics_LSTM_univar_bidir' + site + '_' + snsr, "wb")
#     pickle.dump(LSTM_univar_bidir[snsr].metrics, pickle_out)
#     pickle_out.close()
#     pickle_out = open('saved/metrics_LSTM_multivar' + site + '_' + snsr, "wb")
#     pickle.dump(LSTM_multivar.metrics[snsr], pickle_out)
#     pickle_out.close()
#     pickle_out = open('saved/metrics_LSTM_multivar_bidir' + site + '_' + snsr, "wb")
#     pickle.dump(LSTM_multivar_bidir.metrics[snsr], pickle_out)
#     pickle_out.close()
#     pickle_out = open('saved/metrics_aggregate' + site + '_' + snsr, "wb")
#     pickle.dump(results_all[snsr], pickle_out)
#     pickle_out.close()
#     pickle_out = open('saved/metrics_rules' + site + '_' + snsr, "wb")
#     pickle.dump(rules_metrics[snsr], pickle_out)
#     pickle_out.close()
# print('Finished saving output.')

#### Correction
#########################################
corrections = dict()
for snsr in sensors:
    corrections[snsr] = ARIMA_correct.generate_corrections(df=results_all[snsr],
                                                           observed='observed',
                                                           anomalies='detected_event',
                                                           savecasts=True)
    print(snsr + 'correction complete')

# Saving corrections
for snsr in sensors:
    corrections[snsr].to_csv(r'Examples/' + site + '_' + snsr + '_corrections.csv')
