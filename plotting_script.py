

import anomaly_utilities
import matplotlib.pyplot as plt

i = 1

sensor = sensor[i]

# ARIMA
residuals = ARIMA_detect[i].detections['residual']
predictions = ARIMA_detect[i].detections['prediction']
threshold = ARIMA_detect[i].threshold
raw = ARIMA_detect[i].df['raw']
labels = ARIMA_detect[i].df['labeled_event']
detections = ARIMA_detect[i].df['detected_event']

# LSTM Univariate
residuals = LSTM_detect_univar[i].detections['residual']
predictions = LSTM_detect_univar[i].detections['prediction']
threshold = LSTM_detect_univar[i].threshold
raw = LSTM_detect_univar[i].df_anomalies['raw']
labels = LSTM_detect_univar[i].df_anomalies['labeled_event']
detections = LSTM_detect_univar[i].df_anomalies['detected_event']

# LSTM Multivariate
residuals = LSTM_detect_multivar.detections_array[i]['residual']
predictions = LSTM_detect_multivar.detections_array[i]['prediction']
threshold = LSTM_detect_multivar.threshold[i]
raw = LSTM_detect_multivar.df_raw[df_raw.columns[i]]
labels = LSTM_detect_multivar.df_array[sensor[i]]['labeled_event']
detections = LSTM_detect_multivar.df_array[sensor[i]]['detected_event']


# Threshold Plot #
plt.figure()
anomaly_utilities.plt_threshold(residuals, threshold, sensor)
plt.show()

# Results Plot #
plt.figure()
anomaly_utilities.plt_results(
    raw=raw,
    predictions=predictions,
    labels=labels,
    detections=detections,
    sensor=sensor
)
plt.show()



for i in range(0, len(sensor)):
    plt.figure()
    anomaly_utilities.plt_results(
        raw=raw[i],
        predictions=predictions[i],
        labels=labels[i],
        detections=detections[i],
        sensor=sensor[i]
    )
    plt.show()

for i in range(0, len(sensor)):
        plt.figure()
        anomaly_utilities.plt_threshold(residuals.iloc[:, i], threshold[i], sensor[i])
        plt.show()
