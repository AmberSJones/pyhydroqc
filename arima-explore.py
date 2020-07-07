####################################
### ARIMA DEVELOP AND DIAGNOSTIC ###
####################################
# This code takes raw data and corrected data, applies an ARIMA time series model, identifies anomalies, outputs metrics

# print("ARIMA exploration script begin.")
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import statsmodels.api as api
# import statsmodels.tsa as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import seaborn as sns

import pmdarima as pm
# register_matplotlib_converters()
# plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

### DEFINE SITE and VARIABLE ###
# Inputs: Raw Data, Corrected Data, Parameters (p,d,q, threshold)
# declarations: site, model parameters (threshold and (p, d, q))
# site = "BlackSmithFork"
# site = "FranklinBasin"
# site = "MainStreet"
# site = "Mendon"
# site = "TonyGrove"
site = "WaterLab"
# sensor = "temp"
sensor = "cond"
# sensor = "ph"
# sensor = "do"
# sensor = "turb"
# sensor = "stage"
year = 2014

### GET DATA ###
os.chdir("../")
cwd = os.getcwd()
# print(cwd)
data_dir = "/TS-ARIMA/LRO_data/"
# file_list = os.listdir(cwd + data_dir)
# print(file_list)
print('Importing data from ' + cwd + data_dir + site + str(year) + ".csv")
# df = pd.read_csv(cwd + data_dir + site + ".csv",
df = pd.read_csv(cwd + data_dir + site + str(year) + ".csv",
                  skipinitialspace=True,
                  engine='python',
                  header=0,
                  index_col=0,
                  parse_dates=True,
                  infer_datetime_format=True)
print(df.head())

# makes one-dimensional data frame of booleans based on qualifier column indicating normal (TRUE) or not (FALSE)
normal_lbl = df[sensor + "_qual"].isnull()

#generate series from dataframe - time indexed values
srs = pd.Series(df[sensor])

# plt.figure()
# plt.plot(srs)
# plt.show()


### PARAMETER SELECTION ###
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
            [[0,0,5],[0,0,5],[0,1,4],[1,1,0],[9,1,5]], #BlackSmithFork
            [[10,1,3],[0,1,5],[10,1,1],[6,1,4],[0,1,5]], #FranklinBasin
            [[1,1,5],[1,1,1],[3,1,1],[0,0,0],[1,1,5]],#MainStreet
            [[9,1,4],[10,1,3],[0,1,2],[3,1,1],[9,1,1]],#Mendon
            [[6,1,2],[10,1,0],[8,1,4],[10,1,0],[10,1,5]],#TonyGrove
            [[7,1,0],[1,1,1],[10,1,0],[0,1,5],[1,1,3]]#WaterLab
            ]
pdqParam = pd.DataFrame(pdqParams, columns=sensors, index=sites)
print(pdqParam)

p, d, q = pdqParam[sensor][site]
print("p: "+str(p))
print("d: "+str(d))
print("q: "+str(q))

# Seasonal exploration. Determined not to pursue at this point.
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

### BUILD MODEL ###
model = ARIMA(srs, order=(p, d, q))
# print(model.arimaresults.aic) # doesn't work
model_fit = model.fit(disp=0)

# output summary
print('\n\n')
print(model_fit.summary())
model_fit.plot_predict(dynamic=False)

# find residual errors
residuals = pd.DataFrame(model_fit.resid)
print('\n\nresiduals description:')
print(residuals.describe())

### DETERMINE THRESHOLD ###
# Need to determine prediction intervals
# could also try to maximize F2

threshold = 12

forecast, stderr, conf = model_fit.forecast(steps=5)


# calculate prediction interval
z = 5
SSE = np.sum(residuals**2)
n = len(residuals)
interval = z * np.sqrt(SSE/)
np.max(interval)
# If need upper/lower PI for plotting
# lower, upper = model_fit.predict() - interval, model_fit.predict() + interval



### DETERMINE ANOMALIES ###
anomDetn = np.abs(residuals) > threshold # gives bools
anomDetn[0][0] = False # correct 1st value
print('\nratio of detections: %f' % ((sum(anomDetn[0])/len(srs))*100), '%')

# determine labeled anomalies
# windowing
anomaly_count = 0
anomaly_events = []
anomaly_events.append(0)
for i in range(1, len(normal_lbl)):
  if not(normal_lbl[i]):
    if normal_lbl[i-1]:
      anomaly_count += 1
      anomaly_events[i-1] = anomaly_count
    anomaly_events.append(anomaly_count)
  else:
    if not(normal_lbl[i-1]):
      anomaly_events.append(anomaly_count)
    else:
      anomaly_events.append(0)


anomLbl = pd.DataFrame(data=anomaly_events, index=normal_lbl.index)

#fix missing values
anomDetn = anomDetn.reindex(anomLbl.index)
anomDetn[anomDetn.isnull()] = True

#determine detecteions
det_count = 0
det_events = []
det_events.append(0)
for i in range(1, len(anomDetn)):
  if anomDetn[0][i]: #is a detection
    if not(anomDetn[0][i-1]): #prev is not detection
      det_count += 1
      det_events[i-1] = det_count
    det_events.append(det_count)#append detection number
  else: #is not a detection
    if anomDetn[0][i-1]: #prev is a detection
      det_events.append(det_count)# append det number
    else:
      det_events.append(0) #not a detection


anomDetns = pd.DataFrame(data=det_events, index=normal_lbl.index)

#generate lists of detected anomalies and valid detections
detected_anomalies = [0]
valid_detections = [0]
for i in range(0, len(anomDetn)):
  if anomDetn[0][i]: # anomaly detected
    if 0!=anomLbl[0][i]: # labeled as anomaly
      if not(detected_anomalies[-1] == anomLbl[0][i]):#if not already in list of detected anomalies
        detected_anomalies.append(anomLbl[0][i])
      if not(valid_detections[-1] == anomDetns[0][i]):#if not already in list of valid detections
        valid_detections.append(anomDetns[0][i])


detected_anomalies.pop(0)
valid_detections.pop(0)
invalid_detections = []
det_ind = 0
for i in range(1, max(anomDetns[0])):
  if (det_ind < len(valid_detections)):
    if i == valid_detections[det_ind]:
      det_ind += 1
    else:
      invalid_detections.append(i)

#generate plots
anom_srs = model_fit.predict()
# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)
# ax1.plot(srs[anomDetn[0]], 'r+', label='anomalies')
# fig.plot(srs[anomDetn[0]], 'r+')
# model_fit.plot_predict(dynamic=False)
# plt.plot(srs[anomDetn[0]], 'r+', label='anomalies')
# ax1.set_ylabel('anomalies, forcast, Specific Conductance')
# legend = ax1.legend()

# odo_srs = srs
# odo_anom_srs = anom_srs
# odo_normal_lbl = normal_lbl
# odo_anomDetn = anomDetn

# ph_srs = srs
# ph_anom_srs = anom_srs
# ph_normal_lbl = normal_lbl
# ph_anomDetn = anomDetn

# spc_srs = srs
# spc_anom_srs = anom_srs
# spc_normal_lbl = normal_lbl
# spc_anomDetn = anomDetn

# fig = plt.figure(figsize=(10,10))

# ax1 = fig.add_subplot(3, 1, 1)
# ax1.plot(odo_srs, 'b', label='original data')
# ax1.plot(odo_anom_srs, 'c', label='predicted values')
# ax1.plot(odo_srs[~odo_normal_lbl], 'mo', mfc='none', label='labeled anomalies')
# ax1.plot(odo_anom_srs[odo_anomDetn[0]], 'r+', label='detected anomalies')
# # ax1.legend()
# ax1.ylabel('Dissolved Oxygen')

# ax2 = fig.add_subplot(3, 1, 3)
# ax2.plot(ph_srs, 'b', label='original data')
# ax2.plot(ph_anom_srs, 'c', label='predicted values')
# ax2.plot(ph_srs[~ph_normal_lbl], 'mo', mfc='none', label='labeled anomalies')
# ax2.plot(ph_anom_srs[ph_anomDetn[0]], 'r+', label='detected anomalies')
# # ax2.legend()
# ax2.ylabel('pH')

# ax3 = fig.add_subplot(3, 1, 2)
# ax3.plot(spc_srs, 'b', label='original data')
# ax3.plot(spc_anom_srs+spc_srs, 'c', label='predicted values')
# ax3.plot(spc_srs[~spc_normal_lbl], 'mo', mfc='none', label='labeled anomalies')
# ax3.plot(spc_anom_srs[spc_anomDetn[0]]+spc_srs[spc_anomDetn[0]], 'r+', label='detected anomalies')
# # ax3.legend()
# ax3.ylabel('Specific Conductance')

# plt.figure()
# plt.plot(srs, 'b', label='original data')
# plt.plot(anom_srs, 'c', label='predicted values')
# plt.plot(srs[~normal_lbl], 'mo', mfc='none', label='labeled anomalies')
# plt.plot(anom_srs[anomDetn[0]], 'r+', label='detected anomalies')
# plt.legend()
# plt.show()


### METRICS AND EVALUATION ###
#generate confusion matrix
TruePositives = sum(anomLbl[0].value_counts()[detected_anomalies])
FalseNegatives= len(anomDetn) - anomLbl[0].value_counts()[0] - TruePositives
FalsePositives= sum(anomDetns[0].value_counts()[invalid_detections])
TrueNegatives = len(anomDetn) - TruePositives - FalseNegatives - FalsePositives

PPV = TruePositives/(TruePositives+FalsePositives)
NPV = TrueNegatives/(TrueNegatives+FalseNegatives)
ACC = (TruePositives+TrueNegatives)/len(anomDetn)
F2 = 5*TruePositives/(5*TruePositives+4*FalseNegatives+FalsePositives)
# ACC = (TruePositives+TrueNegatives)/(TruePositives+TrueNegatives+FalsePositives+FalseNegatives)

# Trying to estimate a threshold based on optimizing F score
f_score = []
thresholds = np.arange(0, 20, 0.5) #set range and increments for threshold. will need to generalize for other variables.
for threshold in thresholds:
    f1 = metrics.f1_score(normal_lbl, anomDetns)
    f_score.append(f1)

sns.lineplot(thresholds, f_score)
plt.xlabel("threshold")
plot.ylabel("f1 score")


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
print('F2 = %i' % F2)

print("\nTime Series ARIMA script end.")
