################################
# ARIMA DEVELOP AND DIAGNOSTIC #
################################
# This code takes raw (and corrected if applicable) data, applies an ARIMA time series model, and identifies anomalies.

import rules_detect
import anomaly_utilities
import matplotlib.pyplot as plt
import pandas as pd

print("ARIMA exploration script begin.")

# DEFINE SITE and VARIABLE #
#########################################
# site = "BlackSmithFork"
# site = "FranklinBasin"
# site = "MainStreet"
site = "Mendon"
# site = "TonyGrove"
# site = "WaterLab"
# sensor = "temp"
sensor = ['cond']
# sensor = "ph"
# sensor = "do"
# sensor = "turb"
# sensor = "stage"
year = 2017

# PARAMETER SELECTION #
#########################################
# Need to use an automated method to generalize getting p,d,q parameters
# These are the results of using auto.ARIMA to determine p,d,q parameters in R
sites = {'BlackSmithFork': 0,
         'FranklinBasin': 1,
         'MainStreet': 2,
         'Mendon': 3,
         'TonyGrove': 4,
         'WaterLab': 5}
sensors = {'cond': 0,
           'do': 1,
           'ph': 2,
           'temp': 3,
           'turb': 4}
pdqParams = [
    [[0, 0, 5], [0, 0, 5], [0, 1, 4], [1, 1, 0], [9, 1, 5]],  # BlackSmithFork
    [[10, 1, 3], [0, 1, 5], [10, 1, 1], [6, 1, 4], [0, 1, 5]],  # FranklinBasin
    [[1, 1, 5], [1, 1, 1], [3, 1, 1], [0, 0, 0], [1, 1, 5]],  # MainStreet
    [[9, 1, 4], [10, 1, 3], [0, 1, 2], [3, 1, 1], [9, 1, 1]],  # Mendon
    [[6, 1, 2], [10, 1, 0], [8, 1, 4], [10, 1, 0], [10, 1, 5]],  # TonyGrove
    [[7, 1, 0], [1, 1, 1], [10, 1, 0], [0, 1, 5], [1, 1, 3]]  # WaterLab
]
pdqParam = pd.DataFrame(pdqParams, columns=sensors, index=sites)
print(pdqParam)

p, d, q = pdqParam[sensor[0]][site]
print("p: " + str(p))
print("d: " + str(d))
print("q: " + str(q))

# GET DATA #
#########################################
df_full, sensor_array = anomaly_utilities.get_data(site, sensor, year,path="/Users/amber/PycharmProjects/LRO-anomaly-detection/LRO_data/")
df = sensor_array[sensor[0]]

# RULES BASED DETECTION #
#########################################
# General sensor ranges for LRO data:
# Temp min: -5, max: 30
# SpCond min: 100, max: 900
# pH min: 7.5, max: 9.0
# do min: 2, max: 16
maximum = 900
minimum = 150
df = rules_detect.range_check(df, maximum, minimum)
length = 6
df = rules_detect.persistence(df, length)
size = rules_detect.group_size(df)
df = rules_detect.interpolate(df)

# MODEL CREATION #
#########################################
model_fit, residuals, predictions = anomaly_utilities.build_arima_model(df['det_cor'], p, d, q, summary=True)

# DETERMINE THRESHOLD AND DETECT ANOMALIES #
#########################################
dyn_threshold = anomaly_utilities.set_dynamic_threshold(residuals, 0.01, 75)
dyn_threshold.index = residuals.index

plt.figure()
# plt.plot(df['raw'], 'b', label='original data')
plt.plot(residuals, 'b', label='residuals')
plt.plot(dyn_threshold['low'], 'c', label='thresh_low')
plt.plot(dyn_threshold['high'], 'm', mfc='none', label='thresh_high')
plt.legend()
plt.ylabel(sensor)
plt.show()

detected_anomaly = anomaly_utilities.detect_dyn_anomalies(residuals, dyn_threshold, summary=True)

# Determine constant threshold and detect anomalies
cons_threshold = anomaly_utilities.set_cons_threshold(model_fit, 0.055)
cons_detected_anomaly = anomaly_utilities.detect_anomalies(residuals, cons_threshold, summary=True)

# Use events function to widen and number anomalous events
df['labeled_event'] = anomaly_utilities.anomaly_events(df['labeled_anomaly'])
df['detected_anomaly'] = detected_anomaly[0]
df['detected_event'] = anomaly_utilities.anomaly_events(detected_anomaly)

# DETERMINE METRICS #
#########################################
compare = anomaly_utilities.compare_labeled_detected(df)
metrics = anomaly_utilities.metrics(df, compare.valid_detections, compare.invalid_detections)

# OUTPUT RESULTS #
#########################################
print('\n\n\nScript report:\n')
print('Sensor: ' + sensor[0])
print('Year: ' + str(year))
print('Parameters: ARIMA(%i, %i, %i)' % (p, d, q))
print('PPV = %f' % metrics.PPV)
print('NPV = %f' % metrics.NPV)
print('Acc = %f' % metrics.ACC)
print('TP  = %i' % metrics.TruePositives)
print('TN  = %i' % metrics.TrueNegatives)
print('FP  = %i' % metrics.FalsePositives)
print('FN  = %i' % metrics.FalseNegatives)
print('F1 = %f' % metrics.f1)
print('F2 = %f' % metrics.f2)
print("\nTime Series ARIMA script end.")

# GENERATE PLOTS #
#########################################
plt.figure()
plt.plot(df['raw'], 'b', label='original data')
plt.plot(predictions, 'c', label='predicted values')
plt.plot(df['raw'][df['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
plt.plot(predictions[detected_anomaly], 'r+', label='machine detected anomalies')
plt.legend()
plt.ylabel(sensor)
plt.show()
