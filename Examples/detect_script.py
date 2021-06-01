#####################################
# ANOMALY DETECTION #
#####################################
# This script performs anomaly detection for multiple variables. Parameters are imported.
# The complete workflow for model development and anomaly detection is carried out.
# Model types include ARIMA and LSTM (univariate/multivariate and vanilla/bidirectional).


##### TODO: change i and j loops to dictionary loops


import copy
import pickle
from pyhydroqc import anomaly_utilities
from pyhydroqc import model_workflow
from pyhydroqc import rules_detect
from pyhydroqc.parameters import site_params


class MethodsOutput:
    pass


methods_output = MethodsOutput()


# RETRIEVE DATA #
#########################################

sites = ['FranklinBasin', 'TonyGrove','WaterLab', 'MainStreet', 'Mendon', 'BlackSmithFork']
year = [2014, 2015, 2016, 2017, 2018, 2019]
sensor = ['temp', 'cond', 'ph', 'do']

site_detect = []

rules_metrics = []

for j in range(0, len(sites)):
    site = sites[j]
    if site == 'BlackSmithFork': year.pop(0)
    print("\n\n###########################################\n#Processing data for site: "
          + sites[j] + ".\n###########################################")
    df_full, sensor_array = anomaly_utilities.get_data(sites[j], sensor, year, path="../LRO_data/")

    # RULES BASED ANOMALY DETECTION #
    #########################################
    range_count = []
    persist_count = []
    methods_output.rules_metrics = []
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

        # metrics for rules based detection #
        df_rules_metrics = sensor_array[sensor[i]]
        df_rules_metrics['labeled_event'] = anomaly_utilities.anomaly_events(df_rules_metrics['labeled_anomaly'], wf=0)
        df_rules_metrics['detected_event'] = anomaly_utilities.anomaly_events(df_rules_metrics['anomaly'], wf=0)
        anomaly_utilities.compare_events(df_rules_metrics, wf=0)

        rules_metrics_object = anomaly_utilities.metrics(df_rules_metrics)
        print('\nRules based metrics')
        print('Sensor: ' + sensor[i])
        anomaly_utilities.print_metrics(rules_metrics_object)
        methods_output.rules_metrics.append(rules_metrics_object)

    print('Rules based detection complete.\n')
    del persist_count
    del range_count

    ##############################################
    # MODEL AND ANOMALY DETECTION IMPLEMENTATION #
    ##############################################

    # ARIMA BASED DETECTION #
    # #########################################
    methods_output.ARIMA = []
    for i in range(0, len(sensor)):
        df = sensor_array[sensor[i]]
        methods_output.ARIMA.append(copy.deepcopy(
            model_workflow.arima_detect(
                df, sensor[i], site_params[j][i],
                rules=False, plots=False, summary=False, compare=True
            )))
    print('ARIMA detection complete.\n')
    del df

    # LSTM BASED DETECTION #
    #########################################

    # DATA: univariate,  MODEL: vanilla #
    model_type = 'vanilla'
    methods_output.lstm_univar = []
    for i in range(0, len(sensor)):
        df = sensor_array[sensor[i]]
        name = str(site) + '-' + str(sensor[i])
        method_object = model_workflow.lstm_detect_univar(
            df, sensor[i], site_params[j][i], model_type, name,
            rules=False, plots=False, summary=False, compare=True, model_output=False, model_save=True
            )
        methods_output.lstm_univar.append(copy.deepcopy(method_object))
    del df
    del method_object

    # DATA: univariate,  MODEL: bidirectional #
    model_type = 'bidirectional'
    methods_output.lstm_univar_bidir = []
    for i in range(0, len(sensor)):
        df = sensor_array[sensor[i]]
        name = str(site) + '-' + str(sensor[i])
        method_object = model_workflow.lstm_detect_univar(
                df, sensor[i], site_params[j][i], model_type, name,
                rules=False, plots=False, summary=False, compare=True, model_output=False, model_save=True
            )
        methods_output.lstm_univar_bidir.append(copy.deepcopy(method_object))
    del df
    del method_object

    # DATA: multivariate,  MODEL: vanilla #
    model_type = 'vanilla'
    name = str(site)
    methods_output.lstm_multivar = \
        model_workflow.lstm_detect_multivar(
            sensor_array, sensor, site_params[j], model_type, name,
            rules=False, plots=False, summary=False, compare=True, model_output=False, model_save=True
            )

    # DATA: multivariate,  MODEL: bidirectional #
    model_type = 'bidirectional'
    name = str(site)
    methods_output.lstm_multivar_bidir = \
        model_workflow.lstm_detect_multivar(
            sensor_array, sensor, site_params[j], model_type, name,
            rules=False, plots=False, summary=False, compare=True, model_output=False, model_save=True
            )

    # AGGREGATE DETECTIONS #
    #########################################
    methods_output.aggregate_results = dict()
    methods_output.aggregate_metrics = dict()
    for i in range(0, len(sensor)):
        results_all, metrics = \
            anomaly_utilities.aggregate_results(
                sensor_array[sensor[i]],
                methods_output.ARIMA[i].df,
                methods_output.lstm_univar[i].df_anomalies,
                methods_output.lstm_univar_bidir[i].df_anomalies,
                methods_output.lstm_multivar.df_array[i],
                methods_output.lstm_multivar_bidir.df_array[i]
                )

        print('\nOverall metrics')
        print('Sensor: ' + sensor[i])
        anomaly_utilities.print_metrics(metrics)
        methods_output.aggregate_results.append(copy.deepcopy(results_all))
        methods_output.aggregate_metrics.append(copy.deepcopy(metrics))
    del results_all
    del metrics

    #########################################

    site_detect.append(copy.deepcopy(methods_output))
    print("Finished processing data: " + sites[j])



# SAVING MODELS AND OUTPUT #
#########################################

# Save dataframes necessary to plotting as csv files

for j in range(0, len(sites)):
    for i in range(0, len(sensor)):
        site_detect[j].ARIMA[i].df.to_csv(r'saved/ARIMA_df_' + str(sites[j]) + '_' + str(sensor[i]) + '.csv')
        site_detect[j].ARIMA[i].threshold.to_csv(r'saved/ARIMA_threshold_' + str(sites[j]) + '_' + str(sensor[i]) + '.csv')
        site_detect[j].ARIMA[i].detections.to_csv(r'saved/ARIMA_detections_' + str(sites[j]) + '_' + str(sensor[i]) + '.csv')
        site_detect[j].lstm_univar[i].threshold.to_csv(r'saved/LSTM_univar_threshold_' + str(sites[j]) + '_' + str(sensor[i]) + '.csv')
        site_detect[j].lstm_univar[i].detections.to_csv(r'saved/LSTM_univar_detections_' + str(sites[j]) + '_' + str(sensor[i]) + '.csv')
        site_detect[j].lstm_univar[i].df_anomalies.to_csv(r'saved/LSTM_univar_df_anomalies_' + str(sites[j]) + '_' + str(sensor[i]) + '.csv')
        site_detect[j].lstm_univar_bidir[i].threshold.to_csv(r'saved/LSTM_univar_bidir_threshold_' + str(sites[j]) + '_' + str(sensor[i]) + '.csv')
        site_detect[j].lstm_univar_bidir[i].detections.to_csv(r'saved/LSTM_univar_bidir_detections_' + str(sites[j]) + '_' + str(sensor[i]) + '.csv')
        site_detect[j].lstm_univar_bidir[i].df_anomalies.to_csv(r'saved/LSTM_univar_bidir_df_anomalies_' + str(sites[j]) + '_' + str(sensor[i]) + '.csv')
        site_detect[j].lstm_multivar.threshold[i].to_csv(r'saved/LSTM_multivar_threshold_' + str(sites[j]) + '_' + str(sensor[i]) + '.csv')
        site_detect[j].lstm_multivar.detections_array[i].to_csv(r'saved/LSTM_multivar_detections_' + str(sites[j]) + '_' + str(sensor[i]) + '.csv')
        site_detect[j].lstm_multivar.df_array[i].to_csv(r'saved/LSTM_multivar_df_' + str(sites[j]) + '_' + str(sensor[i]) + '.csv')
        site_detect[j].lstm_multivar.threshold[i].to_csv(r'saved/LSTM_multivar_threshold_' + str(sites[j]) + '_' + str(sensor[i]) + '.csv')
        site_detect[j].lstm_multivar_bidir.detections_array[i].to_csv(r'saved/LSTM_multivar_bidir_detections_' + str(sites[j]) + '_' + str(sensor[i]) + '.csv')
        site_detect[j].lstm_multivar_bidir.df_array[i].to_csv(r'saved/LSTM_multivar_bidir_df_' + str(sites[j]) + '_' + str(sensor[i]) + '.csv')
        site_detect[j].aggregate_results[i].to_csv(r'saved/aggregate_results_' + str(sites[j]) + '_' + str(sensor[i]) + '.csv')

# Save metrics objects as pickles

for j in range(0, len(sites)):
    for i in range(0, len(sensor)):
        pickle_out = open('saved/metrics_ARIMA_' + str(sites[j]) + '_' + str(sensor[i]), "wb")
        pickle.dump(site_detect[j].ARIMA[i].metrics, pickle_out)
        pickle_out.close()
        pickle_out = open('saved/metrics_LSTM_univar_' + str(sites[j]) + '_' + str(sensor[i]), "wb")
        pickle.dump(site_detect[j].lstm_univar[i].metrics, pickle_out)
        pickle_out.close()
        pickle_out = open('saved/metrics_LSTM_univar_bidir' + str(sites[j]) + '_' + str(sensor[i]), "wb")
        pickle.dump(site_detect[j].lstm_univar_bidir[i].metrics, pickle_out)
        pickle_out.close()
        pickle_out = open('saved/metrics_LSTM_multivar' + str(sites[j]) + '_' + str(sensor[i]), "wb")
        pickle.dump(site_detect[j].lstm_multivar.metrics_array[i], pickle_out)
        pickle_out.close()
        pickle_out = open('saved/metrics_LSTM_multivar_bidir' + str(sites[j]) + '_' + str(sensor[i]), "wb")
        pickle.dump(site_detect[j].lstm_multivar_bidir.metrics_array[i], pickle_out)
        pickle_out.close()
        pickle_out = open('saved/metrics_aggregate' + str(sites[j]) + '_' + str(sensor[i]), "wb")
        pickle.dump(site_detect[j].aggregate_metrics[i], pickle_out)
        pickle_out.close()
        pickle_out = open('saved/metrics_rules' + str(sites[j]) + '_' + str(sensor[i]), "wb")
        pickle.dump(site_detect[j].rules_metrics[i], pickle_out)
        pickle_out.close()

print('Finished saving output.')
