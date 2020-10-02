################################
# ARIMA DEVELOP AND DIAGNOSTIC #
################################
# This code takes raw (and corrected if applicable) data, applies an ARIMA time series model, and identifies anomalies.

print("ARIMA exploration script begin.")

import anomaly_utilities
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as api


def build_arima_model(data, p, d, q, summary):
    """Builds an ARIMA model."""
    model = api.tsa.SARIMAX(data, order=(p, d, q))
    model_fit = model.fit(disp=0)
    residuals = pd.DataFrame(model_fit.resid)
    predict = model_fit.get_prediction()
    predictions = pd.DataFrame(predict.predicted_mean)
    residuals[0][0] = 0
    predictions[0][0] = data[0]

    # output summary
    if summary:
        print('\n\n')
        print(model_fit.summary())
        print('\n\nresiduals description:')
        print(residuals.describe())

    return model_fit, residuals, predictions


def set_threshold(model_fit, alpha_in):
    """Determines threshold for anomaly detection based on confidence interval and specified alpha value using
    SARIMAX model predict object."""
    predict = model_fit.get_prediction()
    predict_ci = predict.conf_int(alpha=alpha_in)
    predict_ci.columns = ["lower", "upper"]
    predict_ci["lower"][0] = predict_ci["lower"][1]

    # this gives a constant interval for all points. Considering methods to vary the threshold with data/model
    # variability include using the PI with a SSE over just a previous window or scaling this threshold to a
    # past window SSE.
    # Could also try a threshold to maximize F2, but that requires having labeled data. Could base on a portion of data?
    thresholds = predictions[0] - predict_ci["lower"]
    threshold = thresholds[-1]

    return threshold


def detect_anomalies(residuals, threshold, summary=True):
    """Compares residuals to threshold to identify anomalies. Can use set threshold level or threshold
    determined by set_threshold function."""
    # DETERMINE ANOMALIES
    detected_anomaly = np.abs(residuals) > threshold  # gives bools
    detected_anomaly[0][0] = False  # set 1st value to false
    # output summary
    if summary:
        print('\n\n')
        print('\nratio of detections: %f' % ((sum(detected_anomaly[0])/len(detected_anomaly))*100), '%')

    return detected_anomaly


#########################################
# IMPLEMENTATION AND FUNCTION EXECUTION #
#########################################

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
            [[0,0,5],[0,0,5],[0,1,4],[1,1,0],[9,1,5]],  # BlackSmithFork
            [[10,1,3],[0,1,5],[10,1,1],[6,1,4],[0,1,5]],  # FranklinBasin
            [[1,1,5],[1,1,1],[3,1,1],[0,0,0],[1,1,5]],  # MainStreet
            [[9,1,4],[10,1,3],[0,1,2],[3,1,1],[9,1,1]],  # Mendon
            [[6,1,2],[10,1,0],[8,1,4],[10,1,0],[10,1,5]],  # TonyGrove
            [[7,1,0],[1,1,1],[10,1,0],[0,1,5],[1,1,3]]  # WaterLab
            ]
pdqParam = pd.DataFrame(pdqParams, columns=sensors, index=sites)
print(pdqParam)

p, d, q = pdqParam[sensor[0]][site]
print("p: "+str(p))
print("d: "+str(d))
print("q: "+str(q))

# EXECUTE FUNCTIONS #
#########################################
# Get data
df_full, sensor_array = anomaly_utilities.get_data(site, sensor, year, path="/Users/amber/PycharmProjects/LRO-anomaly-detection/LRO_data/")
df = sensor_array[sensor[0]]

# Run model
model_fit, residuals, predictions = build_arima_model(df['raw'], p, d, q, summary=True)

# Determine threshold and detect anomalies

dyn_threshold = set_dynamic_threshold(residuals, 0.01, 50)
dyn_threshold.index = residuals.index




plt.figure()
# plt.plot(df['raw'], 'b', label='original data')
plt.plot(residuals, 'b', label='residuals')
plt.plot(dyn_threshold['low'], 'c', label='thesh_low')
plt.plot(dyn_threshold['high'], 'm', mfc='none', label='thresh_high')
plt.legend()
plt.ylabel(sensor)
plt.show()


detected_anomaly = np.abs(residuals) > threshold  # gives bools
detected_anomaly[0][0] = False  # set 1st value to false


cons_threshold = set_threshold(model_fit, 0.055)

detected_anomaly = detect_anomalies(residuals, threshold, summary=True)

# Use events function to widen and number anomalous events
df['labeled_event'] = anomaly_utilities.anomaly_events(df['labeled_anomaly'])
df['detected_anomaly'] = detected_anomaly[0]
df['detected_event'] = anomaly_utilities.anomaly_events(detected_anomaly[0])

# Determine Metrics
compare = anomaly_utilities.compare_labeled_detected(df)
metrics = anomaly_utilities.metrics(df, compare.valid_detections, compare.invalid_detections)

# OUTPUT RESULTS #
#########################################
print('\n\n\nScript report:\n')
print('Sensor: ' + sensor[0])
print('Year: ' + str(year))
print('Parameters: ARIMA(%i, %i, %i), Threshold = %f' %(p, d, q, threshold))
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
plt.plot(predictions[detected_anomaly[0]], 'r+', label='machine detected anomalies')
plt.legend()
plt.ylabel(sensor)
plt.show()

