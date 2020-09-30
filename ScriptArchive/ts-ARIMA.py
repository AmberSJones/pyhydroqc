print("Time Series ARIMA script begin.")
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import statsmodels.api as api
#import statsmodels.tsa as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
# register_matplotlib_converters()
# plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

#declarations: sensor, year, model parameters (threshold and (p, d, q))
sensor = 'ODO' #options: 'ODO', 'pH', 'SpCond', 'Stage', 'TurbMed', 'WaterTemp_EXO'
year = '2015' #options: '2015', '2016', '2017', '2018', '2019'
threshold = 0.5
p = 1
d = 0
q = 1
#Tanner's choice parameters:
# ODO threshold = 0.5, (p, d, q) = (1, 0, 1)
# pH threshold = 0.06, (p, d, q) = (1, 0, 4)
# SpCond threshold = 12, (p, d, q) = (1, 1, 2)
# Stage threshold = 15, (p, d, q) = (1, 0, 1)
# TurbMed threshold = ?, (p, d, q) = (?, ?, ?)
# WaterTemp_EXO threshold = ?, (p, d, q) = (?, ?, ?)

#data file structure 
dir_raw = "LRO_RawData"
dir_corr = "LRO_CorrectedData"
file_list = os.listdir(dir_raw)
# print(file_list)
#Get corrected/labeled data
print('\n\n\nImporting corrected data...\n')
df_lbld = pd.read_csv(dir_corr + "/LR_Mendon_AA_" + sensor + "_SourceID_1_QC_1.csv", header=0, index_col=0, parse_dates=True, infer_datetime_format=True)
print(df_lbld.head())

normal_lbl = df_lbld.QualifierCode.isnull()
file_ind = 0
if '2015' == year:
  normal_lbl = normal_lbl['2015-01-01 00:00:00':'2015-12-31 23:45:00']
  file_ind = 1
elif '2016' == year:
  normal_lbl = normal_lbl['2016-01-01 00:00:00':'2016-12-31 23:45:00']
  file_ind = 2
elif '2017' == year:
  normal_lbl = normal_lbl['2017-01-01 00:00:00':'2017-12-31 23:45:00']
  file_ind = 0
elif '2018' == year:
  normal_lbl = normal_lbl['2018-01-01 00:00:00':'2018-12-31 23:45:00']
  file_ind = 3
elif '2019' == year:
  normal_lbl = normal_lbl['2019-01-01 00:00:00':'2019-12-31 23:45:00']
  file_ind = 4

print('\n\n\nImporting raw data...\n')
df = pd.read_csv(dir_raw + "/" + file_list[file_ind], header=0, index_col=0, parse_dates=True, infer_datetime_format=True)
print(df.head())

#generate series from dataframe - time indexed values
srs = pd.Series(df[sensor])
model = ARIMA(srs, order=(p,d,q))
model_fit = model.fit(disp=0)

print('\n\n')
print(model_fit.summary())
# model_fit.plot_predict(dynamic=False)

#find residual errors
residuals = pd.DataFrame(model_fit.resid)
print('\n\nresiduals description:')
print(residuals.describe())

# determine anomalies
anomDetn = np.abs(residuals) > threshold # gives bools
anomDetn[0][0] = False # correct 1st value
print('\nratio of detections: %f' % ((sum(anomDetn[0])/len(srs))*100), '%')

#determine labeled anomalies
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

#determine detectedions
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

plt.plot(srs, 'b', label='original data')
plt.plot(anom_srs, 'c', label='predicted values')
plt.plot(srs[~normal_lbl], 'mo', mfc='none', label='labeled anomalies')
plt.plot(anom_srs[anomDetn[0]], 'r+', label='detected anomalies')
plt.legend()
plt.show()

#generate confusion matrix
TruePositives = sum(anomLbl[0].value_counts()[detected_anomalies])
FalseNegatives= len(anomDetn) - anomLbl[0].value_counts()[0] - TruePositives
FalsePositives= sum(anomDetns[0].value_counts()[invalid_detections])
TrueNegatives = len(anomDetn) - TruePositives - FalseNegatives - FalsePositives

PPV = TruePositives/(TruePositives+FalsePositives)
NPV = TrueNegatives/(TrueNegatives+FalseNegatives)
ACC = (TruePositives+TrueNegatives)/len(anomDetn)
# ACC = (TruePositives+TrueNegatives)/(TruePositives+TrueNegatives+FalsePositives+FalseNegatives)

print('\n\n\nScript report:\n')
print('Sensor: ' + sensor)
print('Year: ' + year)
print('Parameters: ARIMA(%i, %i, %i), Threshold = %f' %(p, d, q, threshold))
print('PPV = %f' % PPV)
print('NPV = %f' % NPV)
print('Acc = %f' % ACC)
print('TP  = %i' % TruePositives)
print('TN  = %i' % TrueNegatives)
print('FP  = %i' % FalsePositives)
print('FN  = %i' % FalseNegatives)

print("\nTime Series ARIMA script end.")
