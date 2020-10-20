################################
# MODELING WORKFLOW  #
################################

import anomaly_utilities
import modeling_utilities
import rules_detect
import matplotlib.pyplot as plt
import pandas as pd

class ModelWorkflow:
    pass
    """
    """


def ARIMA_detect(df, sensor, p, d, q, minimum, maximum, length, window_sz, alpha, min_range, wf):
    """
    """
    # RULES BASED DETECTION #
    df = rules_detect.range_check(df, maximum, minimum)
    df = rules_detect.persistence(df, length)
    size = rules_detect.group_size(df)
    df = rules_detect.interpolate(df)
    # MODEL CREATION #
    model_fit, residuals, predictions = modeling_utilities.build_arima_model(df['observed'], p, d, q, summary=True)
    # DETERMINE THRESHOLD AND DETECT ANOMALIES #
    threshold = anomaly_utilities.set_dynamic_threshold(residuals[0], window_sz, alpha, min_range)
    threshold.index = residuals.index
    plt.figure()
    plt.plot(residuals, 'b', label='residuals')
    plt.plot(threshold['low'], 'c', label='thresh_low')
    plt.plot(threshold['high'], 'm', mfc='none', label='thresh_high')
    plt.legend()
    plt.ylabel(sensor[0])
    plt.show()

    detections = anomaly_utilities.detect_anomalies(df['observed'], predictions, residuals, threshold, summary=True)

    # Use events function to widen and number anomalous events
    df['labeled_event'] = anomaly_utilities.anomaly_events(df['labeled_anomaly'], wf)
    df['detected_anomaly'] = detections['anomaly']
    df['all_anomalies'] = df.eval('detected_anomaly or anomaly')
    df['detected_event'] = anomaly_utilities.anomaly_events(df['all_anomalies'], wf)
    # DETERMINE METRICS #
    compare = anomaly_utilities.compare_labeled_detected(df)
    metrics = anomaly_utilities.metrics(df, compare.valid_detections, compare.invalid_detections)

    # OUTPUT RESULTS #
    print('\n\n\nScript report:\n')
    print('Model type: ARIMA')
    print('Sensor: ' + str(sensor))
    anomaly_utilities.print_metrics(metrics)
    print("\nARIMA script end.")

    # GENERATE PLOTS #
    plt.figure()
    plt.plot(df['raw'], 'b', label='original data')
    plt.plot(predictions, 'c', label='predicted values')
    plt.plot(df['raw'][df['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
    plt.plot(predictions[df['detected_event'] > 0], 'r+', label='machine detected anomalies')
    plt.legend()
    plt.ylabel(sensor[0])
    plt.show()

    ARIMA_detect = ModelWorkflow()
    ARIMA_detect.size = size
    ARIMA_detect.df = df
    ARIMA_detect.model_fit = model_fit
    ARIMA_detect.threshold = threshold
    ARIMA_detect.detections = detections
    ARIMA_detect.compare = compare
    ARIMA_detect.metrics = metrics

    return ARIMA_detect


def LSTM_detect_univar(df, sensor,
                minimum, maximum, length,
                model_type, time_steps, samples, cells, dropout, patience,
                window_sz, alpha, min_range,
                wf):
    """
    """
    # RULES BASED DETECTION #
    df = rules_detect.range_check(df, maximum, minimum)
    df = rules_detect.persistence(df, length)
    size = rules_detect.group_size(df)
    df = rules_detect.interpolate(df)

    # MODEL CREATION #
    if model_type == 'vanilla':
        model = modeling_utilities.LSTM_univar(df, time_steps, samples, cells, dropout, patience)
    else:
        model = modeling_utilities.LSTM_univar_bidir(df, time_steps, samples, cells, dropout, patience)
    plt.plot(model.history.history['loss'], label='Training Loss')
    plt.plot(model.history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

    # DETERMINE THRESHOLD AND DETECT ANOMALIES #
    threshold = anomaly_utilities.set_dynamic_threshold(model.test_residuals[0], window_sz, alpha, min_range)
    if model_type == 'vanilla':
        threshold.index = df[time_steps:].index
    else:
        threshold.index = df[time_steps:-time_steps].index
    residuals = pd.DataFrame(model.test_residuals)
    residuals.index = threshold.index
    plt.figure()
    plt.plot(residuals, 'b', label='residuals')
    plt.plot(threshold['low'], 'c', label='thresh_low')
    plt.plot(threshold['high'], 'm', mfc='none', label='thresh_high')
    plt.legend()
    plt.ylabel(sensor)
    plt.show()
    if model_type == 'vanilla':
        observed = df[['observed']][time_steps:]
    else:
        observed = df[['observed']][time_steps:-time_steps]
    detections = anomaly_utilities.detect_anomalies(observed, model.predictions, model.test_residuals,
                                                    threshold, summary=True)
    # Use events function to widen and number anomalous events
    df_anomalies = df.iloc[time_steps:]
    df_anomalies['labeled_event'] = anomaly_utilities.anomaly_events(df_anomalies['labeled_anomaly'], wf)
    df_anomalies['detected_anomaly'] = detections['anomaly']
    df_anomalies['all_anomalies'] = df_anomalies.eval('detected_anomaly or anomaly')
    df_anomalies['detected_event'] = anomaly_utilities.anomaly_events(df_anomalies['all_anomalies'], wf)

    # DETERMINE METRICS #
    compare = anomaly_utilities.compare_labeled_detected(df_anomalies)
    metrics = anomaly_utilities.metrics(df_anomalies, compare.valid_detections, compare.invalid_detections)

    # OUTPUT RESULTS #
    print('\n\n\nScript report:\n')
    print('Model type: LSTM univariate ' + str(model_type))
    print('Sensor: ' + str(sensor))
    anomaly_utilities.print_metrics(metrics)
    print("\n LSTM script end.")

    # GENERATE PLOTS #
    plt.figure()
    plt.plot(df['raw'], 'b', label='original data')
    plt.plot(detections['prediction'], 'c', label='predicted values')
    plt.plot(df['raw'][df['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
    plt.plot(detections['prediction'][df_anomalies['detected_event'] > 0], 'r+',
             label='machine detected anomalies')
    plt.legend()
    plt.ylabel(sensor)
    plt.show()

    LSTM_detect_univar = ModelWorkflow()
    LSTM_detect_univar.size = size
    LSTM_detect_univar.df = df
    LSTM_detect_univar.model = model
    LSTM_detect_univar.threshold = threshold
    LSTM_detect_univar.detections = detections
    LSTM_detect_univar.df_anomalies = df_anomalies
    LSTM_detect_univar.compare = compare
    LSTM_detect_univar.metrics = metrics

    return LSTM_detect_univar


def LSTM_detect_multivar(sensor_array, sensor,
                minimum, maximum, length,
                model_type, time_steps, samples, cells, dropout, patience,
                window_sz, alpha, min_range,
                wf):
    """

    """
    # RULES BASED DETECTION #
    size = []
    for i in range(0, len(sensor_array)):
        sensor_array[sensor[i]] = rules_detect.range_check(sensor_array[sensor[i]], maximum[i], minimum[i])
        sensor_array[sensor[i]] = rules_detect.persistence(sensor_array[sensor[i]], length)
        s = rules_detect.group_size(sensor_array[sensor[i]])
        size.append(s)
        sensor_array[sensor[i]] = rules_detect.interpolate(sensor_array[sensor[i]])

    # Create new data frames with raw  and observed (after applying rules) and preliminary anomaly detections for selected sensors
    df_raw = pd.DataFrame(index=sensor_array[sensor[0]].index)
    df_observed = pd.DataFrame(index=sensor_array[sensor[0]].index)
    df_anomaly = pd.DataFrame(index=sensor_array[sensor[0]].index)
    for i in range(0, len(sensor_array)):
        df_raw[str(sensor[i]) + '_raw'] = sensor_array[sensor[i]]['raw']
        df_observed[str(sensor[i]) + '_obs'] = sensor_array[sensor[i]]['observed']
        df_anomaly[str(sensor[i]) + '_anom'] = sensor_array[sensor[i]]['anomaly']
    print(df_raw.shape)
    print(df_observed.shape)
    print(df_anomaly.shape)

    # MODEL CREATION #
    if model_type == 'vanilla':
        model = modeling_utilities.LSTM_multivar(df_observed, df_anomaly, df_raw, time_steps, samples, cells,
                                                     dropout, patience)
    else:
        model = modeling_utilities.LSTM_multivar_bidir(df_observed, df_anomaly, df_raw, time_steps, samples, cells,
                                                       dropout, patience)
    # Plot Metrics and Evaluate the Model
    plt.plot(model.history.history['loss'], label='Training Loss')
    plt.plot(model.history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

    # DETERMINE THRESHOLD AND DETECT ANOMALIES #
    residuals = pd.DataFrame(model.test_residuals)
    if model_type == 'vanilla':
        residuals.index = df_observed[time_steps:].index
    else:
        residuals.index = df_observed[time_steps:-time_steps].index
    threshold = []
    for i in range(0, model.test_residuals.shape[1]):
        threshold_df = anomaly_utilities.set_dynamic_threshold(residuals.iloc[:, i], window_sz[i], alpha[i], min_range[i])
        threshold_df.index = residuals.index
        threshold.append(threshold_df)
        plt.figure()
        plt.plot(residuals.iloc[:, i], 'b', label='residuals')
        plt.plot(threshold[i]['low'], 'c', label='thresh_low')
        plt.plot(threshold[i]['high'], 'm', mfc='none', label='thresh_high')
        plt.legend()
        plt.ylabel(sensor[i])
        plt.show()

    if model_type == 'vanilla':
        observed = df_observed[time_steps:]
    else:
        observed = df_observed[time_steps:-time_steps]
    detections_array = []
    for i in range(0, observed.shape[1]):
        detections_df = anomaly_utilities.detect_anomalies(observed.iloc[:, i], model.predictions.iloc[:, i],
                                                           model.test_residuals.iloc[:, i], threshold[i], summary=True)
        detections_array.append(detections_df)

    # Use events function to widen and number anomalous events
    df_array = []
    for i in range(0, len(detections_array)):
        all_data = []
        all_data = sensor_array[sensor[i]].iloc[time_steps:]
        all_data['labeled_event'] = anomaly_utilities.anomaly_events(all_data['labeled_anomaly'], wf)
        all_data['detected_anomaly'] = detections_array[i]['anomaly']
        all_data['all_anomalies'] = all_data.eval('detected_anomaly or anomaly')
        all_data['detected_event'] = anomaly_utilities.anomaly_events(all_data['all_anomalies'], wf)
        df_array.append(all_data)

    # DETERMINE METRICS #
    compare_array = []
    metrics_array = []
    for i in range(0, len(df_array)):
        compare = anomaly_utilities.compare_labeled_detected(df_array[i])
        compare_array.append(compare)
        metrics = anomaly_utilities.metrics(df_array[i], compare_array[i].valid_detections, compare_array[i].invalid_detections)
        metrics_array.append(metrics)

    # OUTPUT RESULTS #
    for i in range(0, len(metrics_array)):
        print('\n\n\nMetrics report:\n')
        print('Model type: LSTM multivariate ' + str(model_type))
        print('Sensor: ' + str(sensor[i]))
        anomaly_utilities.print_metrics(metrics_array[i])

    # GENERATE PLOTS #
    for i in range(0, len(sensor)):
        plt.figure()
        plt.plot(df_raw[df_raw.columns[i]], 'b', label='original data')
        plt.plot(detections_array[i]['prediction'], 'c', label='predicted values')
        plt.plot(sensor_array[sensor[i]]['raw'][sensor_array[sensor[i]]['labeled_anomaly']], 'mo', mfc='none',
                 label='technician labeled anomalies')
        plt.plot(detections_array[i]['prediction'][detections_array[i]['anomaly']], 'r+',
                 label='machine detected anomalies')
        plt.plot(detections_array[i]['prediction'][df_array[i]['detected_event'] > 0], 'r+',
                 label='machine detected anomalies')
        plt.legend()
        plt.ylabel(sensor[i])
        plt.show()

    LSTM_detect_multivar = ModelWorkflow()
    LSTM_detect_multivar.size = size
    LSTM_detect_multivar.sensor_array = sensor_array
    LSTM_detect_multivar.df_observed = df_observed
    LSTM_detect_multivar.df_raw = df_raw
    LSTM_detect_multivar.df_anomaly = df_anomaly
    LSTM_detect_multivar.model = model
    LSTM_detect_multivar.threshold = threshold
    LSTM_detect_multivar.detections_array = detections_array
    LSTM_detect_multivar.df_array = df_array
    LSTM_detect_multivar.compare_array = compare_array
    LSTM_detect_multivar.metrics_array = metrics_array

    return LSTM_detect_univar
