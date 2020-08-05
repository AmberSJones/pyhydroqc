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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()
# plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})


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
sensor = "cond"
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

p, d, q = pdqParam[sensor][site]
print("p: "+str(p))
print("d: "+str(d))
print("q: "+str(q))

# EXECUTE FUNCTIONS #
#########################################
# Get data
df_full, df = anomaly_utilities.get_data(site, sensor, year, path="/Users/amber/PycharmProjects/LRO-anomaly-detection/LRO_data/")

# Run model
model_fit, residuals, predictions = build_arima_model(df['raw'], p, d, q, summary=True)

# Determine threshold and detect anomalies
threshold = set_threshold(model_fit, 0.025)
detected_anomaly = detect_anomalies(residuals, threshold, summary=True)

# Use events function to widen and number anomalous events
df['labeled_event'] = anomaly_utilities.anomaly_events(df['labeled_anomaly'])
df['detected_anomaly'] = detected_anomaly[0]
df['detected_event'] = anomaly_utilities.anomaly_events(detected_anomaly[0])

# Determine Metrics
labeled_in_detected, detected_in_labeled, valid_detections, invalid_detections = anomaly_utilities.compare_labeled_detected(df)
TruePositives, FalseNegatives, FalsePositives, TrueNegatives, PRC, PPV, NPV, ACC, RCL, f1, f2 \
    = anomaly_utilities.metrics(df, valid_detections, invalid_detections)

# OUTPUT RESULTS #
#########################################
print('\n\n\nScript report:\n')
print('Sensor: ' + sensor)
print('Year: ' + str(year))
print('Parameters: ARIMA(%i, %i, %i), Threshold = %f' %(p, d, q, threshold))
print('PPV = %f' % PPV)
print('NPV = %f' % NPV)
print('Acc = %f' % ACC)
print('TP  = %i' % TruePositives)
print('TN  = %i' % TrueNegatives)
print('FP  = %i' % FalsePositives)
print('FN  = %i' % FalseNegatives)
print('F1 = %f' % f1)
print('F2 = %f' % f2)
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


# EXPLORING SEASONAL ARIMA. Seems like seasonal would be useful given the diel fluctuation of ambient/aquatic variables.
# However, using 96 as the seasonal basis (4x24 hours) is computationally intensive.
# In parameter searching, seasonal variables were not significant?

# fig = api.graphics.tsa.plot_acf(srs.diff().dropna(), lags=700)
# plt.show()
# d = pm.arima.ndiffs(srs, test='adf')
# D = pm.arima.nsdiffs(srs, m=96, max_D=5)
# print("D = " + str(D)+"\nd = "+str(d)+"\n")
# model = pm.auto_arima(srs, start_p=1, start_q=1,
#     # test='adf', # use adftest to find optimal 'd'
#     max_p=3,
#     max_q=3,
#     seasonal=False,
#     # seasonal=True, # daily seasonality
#     # m=96, # number of samples per seasonal cycle, 4 samples/hr * 24hrs
#     d=0,
#     start_P=0,
#     # D=D,
#     trace=True,
#     error_action='ignore',
#     suppress_warnings=True,
#     stepwise=True)
# print(model.summary())

# initial exploration into parameter grid search
# p_vals = [0,1,2,3,4]
# d_vals = [0,1]
# q_vals = [0,1,2,3,4]
# for p in p_vals:
#   for d in d_vals:
#     for q in q_vals:
#       try:
#         order = (p,d,q)
#         model = ARIMA(srs, order)
#         model_fit = model.fit(disp=0)
#         print(model_fit)

# build ARIMA model


# Trying to use f score to maximize threshold
# def find_threshold(min, max, inc, normal_lbl, anomDetns):
#     f_score = []
#     thresholds = np.arange(min, max, inc) #set range and increments for threshold. will need to
#     generalize for other variables.
#     for threshold in thresholds:
#
#
#         f1 = metrics.f1_score(normal_lbl, anomDetns)
#         f_score.append(f1)
#
#     sns.lineplot(thresholds, f_score)
#     plt.xlabel("threshold")
#     plot.ylabel("f1 score")
#     opt_threshold =
#     return (thresholds, f_score, opt_threshold)
