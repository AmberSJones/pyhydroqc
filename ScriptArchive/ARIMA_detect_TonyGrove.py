################################
# ARIMA DEVELOP AND DIAGNOSTIC #
################################
# This code takes raw (and corrected if applicable) data, applies an ARIMA time series model, and identifies anomalies.

import rules_detect
import anomaly_utilities
import modeling_utilities
import matplotlib.pyplot as plt
import pandas as pd

print("ARIMA exploration script begin.")

# DEFINE SITE and VARIABLE #
#########################################
# site = "BlackSmithFork"
# site = "FranklinBasin"
# site = "MainStreet"
# site = "Mendon"
site = "TonyGrove"
# site = "WaterLab"
sensor = ['temp', 'cond', 'ph', 'do']
year = [2014, 2015, 2016, 2017, 2018, 2019]

# GET DATA #
#########################################
df_full, sensor_array = anomaly_utilities.get_data(site, sensor, year, path="../LRO_data/")
temp_df = sensor_array[sensor[0]]
cond_df = sensor_array[sensor[1]]
ph_df = sensor_array[sensor[2]]
do_df = sensor_array[sensor[3]]

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

#########################################
#  TEMPERATURE #
#########################################

# RULES BASED DETECTION #
#########################################
maximum = 20
minimum = -2
temp_df = rules_detect.range_check(temp_df, maximum, minimum)
length = 6
temp_df = rules_detect.persistence(temp_df, length)
size = rules_detect.group_size(temp_df)
df = rules_detect.interpolate(temp_df)

# MODEL CREATION #
#########################################
p, d, q = pdqParam[sensor[0]][site]
print("p: " + str(p))
print("d: " + str(d))
print("q: " + str(q))
model_fit, residuals, predictions = modeling_utilities.build_arima_model(temp_df['observed'], p, d, q, summary=True)

# DETERMINE THRESHOLD AND DETECT ANOMALIES #
#########################################
threshold = anomaly_utilities.set_dynamic_threshold(residuals[0], 40, 0.0001, 0.25)
threshold.index = residuals.index

plt.figure()
# plt.plot(df['raw'], 'b', label='original data')
plt.plot(residuals, 'b', label='residuals')
plt.plot(threshold['low'], 'c', label='thresh_low')
plt.plot(threshold['high'], 'm', mfc='none', label='thresh_high')
plt.legend()
plt.ylabel(sensor[0])
plt.show()

detections = anomaly_utilities.detect_anomalies(temp_df['observed'], predictions, residuals, threshold, summary=True)

# Use events function to widen and number anomalous events
temp_df['labeled_event'] = anomaly_utilities.anomaly_events(temp_df['labeled_anomaly'], wf=2)
temp_df['detected_anomaly'] = detections['anomaly']
temp_df['all_anomalies'] = temp_df.eval('detected_anomaly or anomaly')
temp_df['detected_event'] = anomaly_utilities.anomaly_events(temp_df['all_anomalies'], wf=2)

# DETERMINE METRICS #
#########################################
anomaly_utilities.compare_events(temp_df, 2)
metrics = anomaly_utilities.metrics(temp_df)

# OUTPUT RESULTS #
#########################################
print('\n\n\nScript report:\n')
print('Sensor: ' + sensor[0])
print('Parameters: ARIMA(%i, %i, %i)' % (p, d, q))
print('PPV = %f' % metrics.prc)
print('NPV = %f' % metrics.npv)
print('Acc = %f' % metrics.acc)
print('TP  = %i' % metrics.true_positives)
print('TN  = %i' % metrics.true_negatives)
print('FP  = %i' % metrics.false_positives)
print('FN  = %i' % metrics.false_negatives)
print('F1 = %f' % metrics.f1)
print('F2 = %f' % metrics.f2)
print("\nTemperature ARIMA script end.")

# GENERATE PLOTS #
#########################################
plt.figure()
plt.plot(df['raw'], 'b', label='original data')
plt.plot(predictions, 'c', label='predicted values')
plt.plot(df['raw'][df['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
plt.plot(predictions[temp_df['detected_event'] > 0], 'r+', label='machine detected anomalies')
plt.legend()
plt.ylabel(sensor[0])
plt.show()

#########################################
#  SPECIFIC CONDUCTANCE #
#########################################

# RULES BASED DETECTION #
#########################################
maximum = 500
minimum = 175
cond_df = rules_detect.range_check(cond_df, maximum, minimum)
length = 6
cond_df = rules_detect.persistence(cond_df, length)
size = rules_detect.group_size(cond_df)
df = rules_detect.interpolate(cond_df)

# MODEL CREATION #
#########################################
p, d, q = pdqParam[sensor[1]][site]
print("p: " + str(p))
print("d: " + str(d))
print("q: " + str(q))
model_fit, residuals, predictions = modeling_utilities.build_arima_model(cond_df['observed'], p, d, q, summary=True)

# DETERMINE THRESHOLD AND DETECT ANOMALIES #
#########################################
threshold = anomaly_utilities.set_dynamic_threshold(residuals[0], 40, 0.0001, 5)
threshold.index = residuals.index

plt.figure()
# plt.plot(df['raw'], 'b', label='original data')
plt.plot(residuals, 'b', label='residuals')
plt.plot(threshold['low'], 'c', label='thresh_low')
plt.plot(threshold['high'], 'm', mfc='none', label='thresh_high')
plt.legend()
plt.ylabel(sensor[1])
plt.show()

detections = anomaly_utilities.detect_anomalies(cond_df['observed'], predictions, residuals, threshold, summary=True)

# Use events function to widen and number anomalous events
cond_df['labeled_event'] = anomaly_utilities.anomaly_events(cond_df['labeled_anomaly'], wf=2)
cond_df['detected_anomaly'] = detections['anomaly']
cond_df['all_anomalies'] = cond_df.eval('detected_anomaly or anomaly')
cond_df['detected_event'] = anomaly_utilities.anomaly_events(cond_df['all_anomalies'], wf=2)

# DETERMINE METRICS #
#########################################
anomaly_utilities.compare_events(cond_df, 2)
metrics = anomaly_utilities.metrics(cond_df)

# OUTPUT RESULTS #
#########################################
print('\n\n\nScript report:\n')
print('Sensor: ' + sensor[1])
print('Parameters: ARIMA(%i, %i, %i)' % (p, d, q))
print('PPV = %f' % metrics.prc)
print('NPV = %f' % metrics.npv)
print('Acc = %f' % metrics.acc)
print('TP  = %i' % metrics.true_positives)
print('TN  = %i' % metrics.true_negatives)
print('FP  = %i' % metrics.false_positives)
print('FN  = %i' % metrics.false_negatives)
print('F1 = %f' % metrics.f1)
print('F2 = %f' % metrics.f2)
print("\nSpecific Conductance ARIMA script end.")

# GENERATE PLOTS #
#########################################
plt.figure()
plt.plot(df['raw'], 'b', label='original data')
plt.plot(predictions, 'c', label='predicted values')
plt.plot(df['raw'][df['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
plt.plot(predictions[cond_df['detected_event'] > 0], 'r+', label='machine detected anomalies')
plt.legend()
plt.ylabel(sensor[1])
plt.show()

#########################################
#  pH #
#########################################

# RULES BASED DETECTION #
#########################################
maximum = 9.0
minimum = 8.0
ph_df = rules_detect.range_check(ph_df, maximum, minimum)
length = 6
ph_df = rules_detect.persistence(ph_df, length)
size = rules_detect.group_size(ph_df)
df = rules_detect.interpolate(ph_df)

# MODEL CREATION #
#########################################
p, d, q = pdqParam[sensor[2]][site]
print("p: " + str(p))
print("d: " + str(d))
print("q: " + str(q))
model_fit, residuals, predictions = modeling_utilities.build_arima_model(ph_df['observed'], p, d, q, summary=True)

# DETERMINE THRESHOLD AND DETECT ANOMALIES #
#########################################
threshold = anomaly_utilities.set_dynamic_threshold(residuals[0], 40, 0.0001, 0.01)
threshold.index = residuals.index

plt.figure()
# plt.plot(df['raw'], 'b', label='original data')
plt.plot(residuals, 'b', label='residuals')
plt.plot(threshold['low'], 'c', label='thresh_low')
plt.plot(threshold['high'], 'm', mfc='none', label='thresh_high')
plt.legend()
plt.ylabel(sensor[2])
plt.show()

detections = anomaly_utilities.detect_anomalies(ph_df['observed'], predictions, residuals, threshold, summary=True)

# Use events function to widen and number anomalous events
ph_df['labeled_event'] = anomaly_utilities.anomaly_events(ph_df['labeled_anomaly'], wf=2)
ph_df['detected_anomaly'] = detections['anomaly']
ph_df['all_anomalies'] = ph_df.eval('detected_anomaly or anomaly')
ph_df['detected_event'] = anomaly_utilities.anomaly_events(ph_df['all_anomalies'], wf=2)

# DETERMINE METRICS #
#########################################
anomaly_utilities.compare_events(ph_df, 2)
metrics = anomaly_utilities.metrics(ph_df)

# OUTPUT RESULTS #
#########################################
print('\n\n\nScript report:\n')
print('Sensor: ' + sensor[2])
print('Parameters: ARIMA(%i, %i, %i)' % (p, d, q))
print('PPV = %f' % metrics.prc)
print('NPV = %f' % metrics.npv)
print('Acc = %f' % metrics.acc)
print('TP  = %i' % metrics.true_positives)
print('TN  = %i' % metrics.true_negatives)
print('FP  = %i' % metrics.false_positives)
print('FN  = %i' % metrics.false_negatives)
print('F1 = %f' % metrics.f1)
print('F2 = %f' % metrics.f2)
print("\npH ARIMA script end.")

# GENERATE PLOTS #
#########################################
plt.figure()
plt.plot(df['raw'], 'b', label='original data')
plt.plot(predictions, 'c', label='predicted values')
plt.plot(df['raw'][df['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
plt.plot(predictions[ph_df['detected_event'] > 0], 'r+', label='machine detected anomalies')
plt.legend()
plt.ylabel(sensor[2])
plt.show()

#########################################
#  DISSOLVED OXYGEN #
#########################################

# RULES BASED DETECTION #
#########################################
maximum = 14
minimum = 7
do_df = rules_detect.range_check(do_df, maximum, minimum)
length = 6
do_df = rules_detect.persistence(do_df, length)
size = rules_detect.group_size(do_df)
df = rules_detect.interpolate(do_df)

# MODEL CREATION #
#########################################
p, d, q = pdqParam[sensor[3]][site]
print("p: " + str(p))
print("d: " + str(d))
print("q: " + str(q))
model_fit, residuals, predictions = modeling_utilities.build_arima_model(do_df['observed'], p, d, q, summary=True)

# DETERMINE THRESHOLD AND DETECT ANOMALIES #
#########################################
threshold = anomaly_utilities.set_dynamic_threshold(residuals[0], 40, 0.001, 0.01)
threshold.index = residuals.index

plt.figure()
# plt.plot(df['raw'], 'b', label='original data')
plt.plot(residuals, 'b', label='residuals')
plt.plot(threshold['low'], 'c', label='thresh_low')
plt.plot(threshold['high'], 'm', mfc='none', label='thresh_high')
plt.legend()
plt.ylabel(sensor[3])
plt.show()

detections = anomaly_utilities.detect_anomalies(do_df['observed'], predictions, residuals, threshold, summary=True)

# Use events function to widen and number anomalous events
do_df['labeled_event'] = anomaly_utilities.anomaly_events(do_df['labeled_anomaly'], wf=2)
do_df['detected_anomaly'] = detections['anomaly']
do_df['all_anomalies'] = do_df.eval('detected_anomaly or anomaly')
do_df['detected_event'] = anomaly_utilities.anomaly_events(do_df['all_anomalies'], wf=2)

# DETERMINE METRICS #
#########################################
anomaly_utilities.compare_events(do_df, 2)
metrics = anomaly_utilities.metrics(do_df)

# OUTPUT RESULTS #
#########################################
print('\n\n\nScript report:\n')
print('Sensor: ' + sensor[3])
print('Parameters: ARIMA(%i, %i, %i)' % (p, d, q))
print('PPV = %f' % metrics.prc)
print('NPV = %f' % metrics.npv)
print('Acc = %f' % metrics.acc)
print('TP  = %i' % metrics.true_positives)
print('TN  = %i' % metrics.true_negatives)
print('FP  = %i' % metrics.false_positives)
print('FN  = %i' % metrics.false_negatives)
print('F1 = %f' % metrics.f1)
print('F2 = %f' % metrics.f2)
print("\nDissolved Oxygen ARIMA script end.")

# GENERATE PLOTS #
#########################################
plt.figure()
plt.plot(df['raw'], 'b', label='original data')
plt.plot(predictions, 'c', label='predicted values')
plt.plot(df['raw'][df['labeled_anomaly']], 'mo', mfc='none', label='technician labeled anomalies')
plt.plot(predictions[do_df['detected_event'] > 0], 'r+', label='machine detected anomalies')
plt.legend()
plt.ylabel(sensor[3])
plt.show()

