################################
# MODELING WORKFLOW  #
################################

from pyhydroqc import anomaly_utilities
from pyhydroqc import modeling_utilities
from pyhydroqc import rules_detect
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum


class ModelWorkflow:
    pass
    """
    """


def arima_detect(df, sensor, params,
                 rules=False, plots=False, summary=True, compare=False, suppress_warnings=True):
    """
    """
    print('\nProcessing ARIMA detections.')
    # RULES BASED DETECTION #
    if rules:
        df = rules_detect.range_check(df, params.max_range, params.min_range)
        df = rules_detect.persistence(df, params.persist)
        size = rules_detect.group_size(df)
        df = rules_detect.interpolate(df)
        print(sensor + ' rules based detection complete. Longest detected group = ' + str(size))

    # MODEL CREATION #
    [p, d, q] = params.pdq
    model_fit, residuals, predictions = modeling_utilities.build_arima_model(df['observed'], p, d, q, summary, suppress_warnings)
    print(sensor + ' ARIMA model complete.')

    # DETERMINE THRESHOLD AND DETECT ANOMALIES #
    threshold = anomaly_utilities.set_dynamic_threshold(residuals[0], params.window_sz, params.alpha, params.threshold_min)
    threshold.index = residuals.index
    if plots:
        plt.figure()
        anomaly_utilities.plt_threshold(residuals, threshold, sensor)
        plt.show()
    print('Threshold determination complete.')
    detections = anomaly_utilities.detect_anomalies(df['observed'], predictions, residuals, threshold, summary=True)

    # WIDEN AND NUMBER ANOMALOUS EVENTS #
    df['detected_anomaly'] = detections['anomaly']
    df['all_anomalies'] = df.eval('detected_anomaly or anomaly')
    df['detected_event'] = anomaly_utilities.anomaly_events(df['all_anomalies'], params.widen)

    if compare:
        df['labeled_event'] = anomaly_utilities.anomaly_events(df['labeled_anomaly'], params.widen)
        # DETERMINE METRICS #
        anomaly_utilities.compare_events(df, params.widen)
        metrics = anomaly_utilities.metrics(df)
        e_metrics = anomaly_utilities.event_metrics(df)
        # OUTPUT RESULTS #
        print('Model type: ARIMA')
        print('Sensor: ' + sensor)
        anomaly_utilities.print_metrics(metrics)
        print('Event based calculations:')
        anomaly_utilities.print_metrics(e_metrics)
        print('Model report complete\n')

    # GENERATE PLOTS #
    if plots:
        plt.figure()
        anomaly_utilities.plt_results(
            raw=df['raw'],
            predictions=detections['prediction'],
            labels=df['labeled_event'],
            detections=df['detected_event'],
            sensor=sensor
        )
        plt.show()

    arima_detect = ModelWorkflow()
    arima_detect.df = df
    arima_detect.model_fit = model_fit
    arima_detect.threshold = threshold
    arima_detect.detections = detections
    if compare:
        arima_detect.metrics = metrics
        arima_detect.e_metrics = e_metrics

    return arima_detect


class ModelType(Enum):
    """
    """
    VANILLA = "vanilla"
    BIDIRECTIONAL = 'bidirectional'


def lstm_detect_univar(df, sensor, params, LSTM_params, model_type, name='',
                rules=False, plots=False, summary=True, compare=False, model_output=True, model_save=True):
    """
    """
    print('\nProcessing LSTM univariate ' + str(model_type) + ' detections.')
    # RULES BASED DETECTION #
    if rules:
        df = rules_detect.range_check(df, params.max_range, params.min_range)
        df = rules_detect.persistence(df, params.persist)
        size = rules_detect.group_size(df)
        df = rules_detect.interpolate(df)
        print(sensor + ' rules based detection complete. Maximum detected group length = '+str(size))

    # MODEL CREATION #
    if model_type == ModelType.VANILLA:
        model = modeling_utilities.lstm_univar(df, LSTM_params, summary, name, model_output, model_save)
    elif model_type == ModelType.BIDIRECTIONAL:
        model = modeling_utilities.lstm_univar_bidir(df, LSTM_params, summary, name, model_output, model_save)
    print(sensor + ' ' + str(model_type) + ' LSTM model complete.')
    if plots:
        plt.figure()
        plt.plot(model.history.history['loss'], label='Training Loss')
        plt.plot(model.history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.show()

    # DETERMINE THRESHOLD AND DETECT ANOMALIES #
    ts = LSTM_params.time_steps
    threshold = anomaly_utilities.set_dynamic_threshold(model.test_residuals[0], params.window_sz, params.alpha, params.threshold_min)
    if model_type == ModelType.VANILLA:
        threshold.index = df[ts:].index
    elif model_type == ModelType.BIDIRECTIONAL:
        threshold.index = df[ts:-ts].index
    residuals = pd.DataFrame(model.test_residuals)
    residuals.index = threshold.index
    if plots:
        plt.figure()
        anomaly_utilities.plt_threshold(residuals, threshold, sensor)
        plt.show()
    if model_type == ModelType.VANILLA:
        observed = df[['observed']][ts:]
    elif model_type == ModelType.BIDIRECTIONAL:
        observed = df[['observed']][ts:-ts]
    print('Threshold determination complete.')
    detections = anomaly_utilities.detect_anomalies(observed, model.predictions, model.test_residuals,
                                                    threshold, summary=True)

    # WIDEN AND NUMBER ANOMALOUS EVENTS #
    if model_type == ModelType.VANILLA:
        df_anomalies = df.iloc[ts:]
    elif model_type == ModelType.BIDIRECTIONAL:
        df_anomalies = df.iloc[ts:-ts]
    df_anomalies['detected_anomaly'] = detections['anomaly']
    df_anomalies['all_anomalies'] = df_anomalies.eval('detected_anomaly or anomaly')
    df_anomalies['detected_event'] = anomaly_utilities.anomaly_events(df_anomalies['all_anomalies'], params.widen)

    if compare:
        df_anomalies['labeled_event'] = anomaly_utilities.anomaly_events(df_anomalies['labeled_anomaly'], params.widen)
        # DETERMINE METRICS #
        anomaly_utilities.compare_events(df_anomalies, params.widen)
        metrics = anomaly_utilities.metrics(df_anomalies)
        e_metrics = anomaly_utilities.event_metrics(df_anomalies)
        # OUTPUT RESULTS #
        print('Model type: LSTM univariate ' + str(model_type))
        print('Sensor: ' + sensor)
        anomaly_utilities.print_metrics(metrics)
        print('Event based calculations:')
        anomaly_utilities.print_metrics(e_metrics)
        print('Model report complete\n')

    # GENERATE PLOTS #
    if plots:
        plt.figure()
        anomaly_utilities.plt_results(
            raw=df['raw'],
            predictions=detections['prediction'],
            labels=df['labeled_event'],
            detections=df_anomalies['detected_event'],
            sensor=sensor
            )
        plt.show()

    lstm_detect_univar = ModelWorkflow()
    lstm_detect_univar.df = df
    lstm_detect_univar.model = model
    lstm_detect_univar.threshold = threshold
    lstm_detect_univar.detections = detections
    lstm_detect_univar.df_anomalies = df_anomalies
    if compare:
        lstm_detect_univar.metrics = metrics
        lstm_detect_univar.e_metrics = e_metrics

    return lstm_detect_univar


def lstm_detect_multivar(sensor_array, sensors, params, LSTM_params, model_type, name='',
                rules=False, plots=False, summary=True, compare=False, model_output=True, model_save=True):
    """
    """
    print('\nProcessing LSTM multivariate ' + str(model_type) + ' detections.')
    # RULES BASED DETECTION #
    if rules:
        size = dict()
        for snsr in sensors:
            sensor_array[snsr], r_c = rules_detect.range_check(sensor_array[snsr], params[snsr].max_range, params[snsr].min_range)
            sensor_array[snsr], p_c = rules_detect.persistence(sensor_array[snsr], params[snsr].persist)
            size[snsr] = rules_detect.group_size(sensor_array[snsr])
            sensor_array[snsr] = rules_detect.interpolate(sensor_array[snsr])
            print(snsr + ' maximum detected group length = ' + str(size[snsr]))
        print('Rules based detection complete.\n')
    # Create new data frames with raw and observed (after applying rules) and preliminary anomaly detections for selected sensors
    df_raw = pd.DataFrame(index=sensor_array[sensors[0]].index)
    df_observed = pd.DataFrame(index=sensor_array[sensors[0]].index)
    df_anomaly = pd.DataFrame(index=sensor_array[sensors[0]].index)
    for snsr in sensors:
        df_raw[snsr + '_raw'] = sensor_array[snsr]['raw']
        df_observed[snsr + '_obs'] = sensor_array[snsr]['observed']
        df_anomaly[snsr + '_anom'] = sensor_array[snsr]['anomaly']
    print('Raw data shape: ' + str(df_raw.shape))
    print('Observed data shape: ' + str(df_observed.shape))
    print('Initial anomalies data shape: ' + str(df_anomaly.shape))

    # MODEL CREATION #
    if model_type == ModelType.VANILLA:
        model = modeling_utilities.lstm_multivar(df_observed, df_anomaly, df_raw, LSTM_params, summary, name, model_output, model_save)
    elif model_type == ModelType.BIDIRECTIONAL:
        model = modeling_utilities.lstm_multivar_bidir(df_observed, df_anomaly, df_raw, LSTM_params, summary, name, model_output, model_save)

    print('multivariate ' + str(model_type) + ' LSTM model complete.\n')
    # Plot Metrics and Evaluate the Model
    if plots:
        plt.figure()
        plt.plot(model.history.history['loss'], label='Training Loss')
        plt.plot(model.history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.show()

    # DETERMINE THRESHOLD AND DETECT ANOMALIES #
    ts = LSTM_params.time_steps
    residuals = pd.DataFrame(model.test_residuals)
    residuals.columns = sensors
    predictions = pd.DataFrame(model.predictions)
    predictions.columns = sensors
    if model_type == ModelType.VANILLA:
        residuals.index = df_observed[ts:].index
        predictions.index = df_observed[ts:].index
        observed = df_observed[ts:]
    elif model_type == ModelType.BIDIRECTIONAL:
        residuals.index = df_observed[ts:-ts].index
        predictions.index = df_observed[ts:-ts].index
        observed = df_observed[ts:-ts]

    threshold = dict()
    detections = dict()
    for snsr in sensors:
        threshold[snsr] = anomaly_utilities.set_dynamic_threshold(
            residuals[snsr], params[snsr].window_sz, params[snsr].alpha, params[snsr].threshold_min)
        threshold[snsr].index = residuals.index
        detections[snsr] = anomaly_utilities.detect_anomalies(
            observed[snsr+'_obs'], predictions[snsr], residuals[snsr], threshold[snsr], summary=True)
        if plots:
            plt.figure()
            anomaly_utilities.plt_threshold(residuals[snsr], threshold[snsr], sensors[snsr])
            plt.show()
    print('Threshold determination complete.')

    # WIDEN AND NUMBER ANOMALOUS EVENTS #
    all_data = dict()
    for snsr in sensors:
        if model_type == ModelType.VANILLA:
            all_data[snsr] = sensor_array[snsr].iloc[ts:]
        elif model_type == ModelType.BIDIRECTIONAL:
            all_data[snsr] = sensor_array[snsr].iloc[ts:-ts]
        all_data[snsr]['detected_anomaly'] = detections[snsr]['anomaly']
        all_data[snsr]['all_anomalies'] = all_data[snsr].eval('detected_anomaly or anomaly')
        all_data[snsr]['detected_event'] = anomaly_utilities.anomaly_events(all_data[snsr]['all_anomalies'], params[snsr].widen)

    # COMPARE AND DETERMINE METRICS #
    if compare:
        metrics = dict()
        e_metrics = dict()
        for snsr in sensors:
            all_data[snsr]['labeled_event'] = anomaly_utilities.anomaly_events(all_data[snsr]['labeled_anomaly'], params[snsr].widen)
            anomaly_utilities.compare_events(all_data[snsr], params[snsr].widen)
            metrics[snsr] = anomaly_utilities.metrics(all_data[snsr])
            e_metrics[snsr] = anomaly_utilities.event_metrics(all_data[snsr])
        # OUTPUT RESULTS #
            print('\nModel type: LSTM multivariate ' + str(model_type))
            print('Sensor: ' + snsr)
            anomaly_utilities.print_metrics(metrics[snsr])
            print('Event based calculations:')
            anomaly_utilities.print_metrics(e_metrics[snsr])
        print('Model report complete\n')

    # GENERATE PLOTS #
    if plots:
        for snsr in sensors:
            plt.figure()
            anomaly_utilities.plt_results(
                raw=sensor_array[snsr]['raw'],
                predictions=detections[snsr]['prediction'],
                labels=sensor_array[snsr]['labeled_event'],
                detections=all_data[snsr]['detected_event'],
                sensor=snsr
                )
            plt.show()

    lstm_detect_multivar = ModelWorkflow()
    lstm_detect_multivar.sensor_array = sensor_array
    lstm_detect_multivar.df_observed = df_observed
    lstm_detect_multivar.df_raw = df_raw
    lstm_detect_multivar.df_anomaly = df_anomaly
    lstm_detect_multivar.model = model
    lstm_detect_multivar.threshold = threshold
    lstm_detect_multivar.detections = detections
    lstm_detect_multivar.all_data = all_data
    if compare:
        lstm_detect_multivar.metrics = metrics
        lstm_detect_multivar.e_metrics = e_metrics

    return lstm_detect_multivar
