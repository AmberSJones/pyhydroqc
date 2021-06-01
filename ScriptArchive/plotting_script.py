

import anomaly_utilities
import matplotlib.pyplot as plt

i = 1

sensor = sensor[i]

# ARIMA
residuals = arima_detect[i].detections['residual']
predictions = arima_detect[i].detections['prediction']
threshold = arima_detect[i].threshold
raw = arima_detect[i].df['raw']
labels = arima_detect[i].df['labeled_event']
detections = arima_detect[i].df['detected_event']

# LSTM Univariate
residuals = lstm_detect_univar[i].detections['residual']
predictions = lstm_detect_univar[i].detections['prediction']
threshold = lstm_detect_univar[i].threshold
raw = lstm_detect_univar[i].df_anomalies['raw']
labels = lstm_detect_univar[i].df_anomalies['labeled_event']
detections = lstm_detect_univar[i].df_anomalies['detected_event']

# LSTM Multivariate
residuals = LSTM_detect_multivar_bidirectional.detections_array[i]['residual']
predictions = LSTM_detect_multivar_bidirectional.detections_array[i]['prediction']
threshold = LSTM_detect_multivar_bidirectional.threshold[i]
raw = LSTM_detect_multivar_bidirectional.df_array[i]['raw']
labels = LSTM_detect_multivar_bidirectional.df_array[i]['labeled_event']
detections = LSTM_detect_multivar_bidirectional.df_array[i]['detected_event']


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
    sensor='pH'
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
