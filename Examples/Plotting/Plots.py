#########################################
# Time Series Figures
#########################################

#### Import Libraries and Functions
from pyhydroqc import anomaly_utilities, rules_detect, calibration
from pyhydroqc.parameters import site_params
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np
import os

colors = ['#0C7BDC', '#F3870D', '#24026A', '#AF3C31']

# FIGURES 3 (gap values and drift correction), 4 (threshold), C1 (detection example), C2 (long labeled event),
# C3 (model detection for calibration events)
# These figures all use data from Main Street.

#### Retrieve data
#########################################
site = 'MainStreet'
sensors = ['temp', 'cond', 'ph', 'do']
years = [2014, 2015, 2016, 2017, 2018, 2019]
sensor_array = anomaly_utilities.get_data(sensors=sensors, site=site, years=years, path="./LRO_data/")

#### Rules Based Anomaly Detection
#########################################
range_count = dict()
persist_count = dict()
rules_metrics = dict()
for snsr in sensor_array:
    sensor_array[snsr], range_count[snsr] = rules_detect.range_check(df=sensor_array[snsr],
                                                                     maximum=site_params[site][snsr]['max_range'],
                                                                     minimum=site_params[site][snsr]['min_range'])
    sensor_array[snsr], persist_count[snsr] = rules_detect.persistence(df=sensor_array[snsr],
                                                                       length=site_params[site][snsr]['persist'],
                                                                       output_grp=True)
    sensor_array[snsr] = rules_detect.interpolate(df=sensor_array[snsr])
print('Rules based detection complete.\n')

### Find Gap Values
#########################################
# Subset of sensors that are calibrated
calib_sensors = sensors[1:4]

# Initialize data structures
calib_dates = dict()
gaps = dict()
shifts = dict()
tech_shifts = dict()

for cal_snsr in calib_sensors:
    # Import calibration dates
    calib_dates[cal_snsr] = pd.read_csv('./LRO_data/' + site + '_' + cal_snsr + '_calib_dates.csv',
                                        header=1, parse_dates=True, infer_datetime_format=True)
    calib_dates[cal_snsr]['start'] = pd.to_datetime(calib_dates[cal_snsr]['start'])
    calib_dates[cal_snsr]['end'] = pd.to_datetime(calib_dates[cal_snsr]['end'])
    # Ensure date range of calibrations correspond to imported data
    calib_dates[cal_snsr] = calib_dates[cal_snsr].loc[(calib_dates[cal_snsr]['start'] > min(sensor_array[cal_snsr].index)) &
                                                      (calib_dates[cal_snsr]['end'] < max(sensor_array[cal_snsr].index))]
    # Initialize dataframe to store determined gap values and associated dates
    gaps[cal_snsr] = pd.DataFrame(columns=['end', 'gap'],
                                  index=range(min(calib_dates[cal_snsr].index), max(calib_dates[cal_snsr].index)+1))
    if len(calib_dates[cal_snsr]) > 0:
        # Initialize data structures
        shifts[cal_snsr] = []
        # Loop through each calibration event date.
        for i in range(min(calib_dates[cal_snsr].index), max(calib_dates[cal_snsr].index)+1):
            # Apply find_gap routine, add to dataframe, add output of shifts to list.
            gap, end = calibration.find_gap(observed=sensor_array[cal_snsr]['observed'],
                                                      calib_date=calib_dates[cal_snsr]['end'][i],
                                                      hours=2,
                                                      show_shift=False)
            gaps[cal_snsr].loc[i]['end'] = end
            gaps[cal_snsr].loc[i]['gap'] = gap
print('Gap value determination complete.\n')

# Review gaps and make adjustments as needed before performing drift correction
gaps['cond'].loc[3, 'gap'] = 4
gaps['cond'].loc[4, 'gap'] = 10
gaps['cond'].loc[21, 'gap'] = 0
gaps['cond'].loc[39, 'gap'] = -5
gaps['cond'].loc[41, 'gap'] = 4
gaps['ph'].loc[33, 'gap'] = -0.04
gaps['ph'].loc[43, 'gap'] = 0.12
gaps['ph'].loc[43, 'end'] = '2019-08-15 15:00'

#### Perform Linear Drift Correction
#########################################
calib_sensors = sensors[1:4]
for cal_snsr in calib_sensors:
    # Set start dates for drift correction at the previously identified calibration (one month back for the first calibration.)
    gaps[cal_snsr]['start'] = gaps[cal_snsr]['end'].shift(1)
    gaps[cal_snsr]['start'][0] = gaps[cal_snsr]['end'][0] - pd.Timedelta(days=30)
    if len(gaps[cal_snsr]) > 0:
        for i in range(min(gaps[cal_snsr].index), max(gaps[cal_snsr].index) + 1):
            result, sensor_array[cal_snsr]['observed'] = calibration.lin_drift_cor(observed=sensor_array[cal_snsr]['observed'],
                                                                                   start=gaps[cal_snsr]['start'][i],
                                                                                   end=gaps[cal_snsr]['end'][i],
                                                                                   gap=gaps[cal_snsr]['gap'][i],
                                                                                   replace=True)
print('Linear drift correction complete.\n')

## FIGURE 3 ##
#########################################
# Compare calibration and drift correction to Observed data and to technician corrected.

cal_snsr = 'ph'
df = sensor_array[cal_snsr]
plt.figure(figsize=(10, 4))
plt.plot(df['raw'], colors[0], label='Observed data')
plt.plot(df['cor'], colors[1], label='Technician corrected')
plt.plot(df['observed'], colors[3], label='Algorithm corrected')
plt.xlim(datetime.datetime(2014, 7, 24), datetime.datetime(2014, 8, 1))  # Specify date range of plot
plt.ylim(7.6, 8.4)
plt.legend()
plt.ylabel('pH')
plt.xlabel('Date')
plt.show()
plt.savefig('Figures/Figure3.png', bbox_inches='tight')

## FIGURE 4 ##
#########################################
# Examine thresholds and model residuals

# set working directory for importing model results.
os.chdir('Examples/Plotting')

ARIMA_detections = pd.read_csv('ARIMA_detections_MainStreet_cond.csv',
                               header=0,
                               index_col=0,
                               parse_dates=True,
                               infer_datetime_format=True)
ARIMA_threshold = pd.read_csv('ARIMA_threshold_MainStreet_cond.csv',
                              header=0,
                              index_col=0,
                              parse_dates=True,
                              infer_datetime_format=True)
plt.figure(figsize=(10, 4))
plt.plot(ARIMA_detections['residual'], 'b', label='Model residuals')
plt.plot(ARIMA_threshold['low'], 'c', label='Upper threshold')
plt.plot(ARIMA_threshold['high'], 'm', mfc='none', label='Lower threshold')
plt.xlim(datetime.datetime(2015, 7, 8), datetime.datetime(2015, 8, 14))  # Specify date range of plot
plt.ylim(-200, 150)
plt.xticks(pd.date_range(start='7/9/2015', end='8/14/2015', freq='5D'))  # Specify xticks at 5-day intervals
plt.legend()
plt.ylabel('Specific conductance, μS/cm')
plt.xlabel('Date')
plt.show()
plt.savefig('Figures/Figure4.png', bbox_inches='tight')

## FIGURE C1 ##
#########################################
# Detection example

LSTM_multivar_bidir_detections = pd.read_csv('LSTM_multivar_bidir_detections_MainStreet_cond.csv',
                                             header=0,
                                             index_col=0,
                                             parse_dates=True,
                                             infer_datetime_format=True)
LSTM_multivar_bidir_df = pd.read_csv('LSTM_multivar_bidir_df_MainStreet_cond.csv',
                                           header=0,
                                           index_col=0,
                                           parse_dates=True,
                                           infer_datetime_format=True)

raw = LSTM_multivar_bidir_df['raw']
predictions = LSTM_multivar_bidir_detections['prediction']
labels = LSTM_multivar_bidir_df['labeled_event']
detections = LSTM_multivar_bidir_df['detected_event']

plt.figure(figsize=(10, 4))
plt.plot(raw, color=colors[0], label='Observed data')
plt.plot(predictions, color=colors[2], label='Model prediction')
plt.plot(raw[labels > 0], 'o', color=colors[1], mfc='none', markersize=5, markeredgewidth=1, label='Technician labeled anomalies')
plt.plot(predictions[detections > 0], 'x', color=colors[3], markersize=6, markeredgewidth=2, label='Algorithm detected anomalies')
plt.xlim(datetime.datetime(2017, 12, 18), datetime.datetime(2017, 12, 27))  # Specify date range of plot
plt.ylim(-20, 1220)
plt.yticks(range(0, 1200, 200))
plt.xticks(pd.date_range(start='12/18/2017', end='12/27/2017', freq='2D'))  # Specify xticks at 2-day intervals
plt.ylabel('Specific conductance, μS/cm')
plt.xlabel('Date')
plt.legend(labelspacing=0.2, loc='upper left', ncol=2, fontsize=9, handletextpad=0.2, columnspacing=0.25)

plt.annotate('true\npositive\nevent', xy=(datetime.datetime(2017, 12, 18, 16, 0), 415), xycoords='data',
             xytext=(datetime.datetime(2017, 12, 19, 8, 0), 800), textcoords='data',
             arrowprops=dict(facecolor='black', width=1.5, headwidth=7),
             horizontalalignment='center', verticalalignment='top')
plt.annotate('true\npositive\nevent', xy=(datetime.datetime(2017, 12, 26, 12, 0), 450), xycoords='data',
             xytext=(datetime.datetime(2017, 12, 25, 20, 0), 850), textcoords='data',
             arrowprops=dict(facecolor='black', width=1.5, headwidth=7),
             horizontalalignment='center', verticalalignment='top')

plt.annotate('false\npositive\nevents', xy=(datetime.datetime(2017, 12, 20, 18, 0), 365), xycoords='data',
             xytext=(datetime.datetime(2017, 12, 22, 20, 0), 175), textcoords='data',
             arrowprops=dict(facecolor='black', width=1.5, headwidth=7),
             horizontalalignment='center', verticalalignment='top')
plt.annotate('', xy=(datetime.datetime(2017, 12, 22, 15, 0), 350), xycoords='data',
             xytext=(datetime.datetime(2017, 12, 22, 18, 0), 190), textcoords='data',
             arrowprops=dict(facecolor='black', width=1.5, headwidth=7),
             horizontalalignment='center', verticalalignment='top')
plt.annotate('    ', xy=(datetime.datetime(2017, 12, 23, 10, 0), 700), xycoords='data',
             xytext=(datetime.datetime(2017, 12, 22, 20, 0), 190), textcoords='data',
             arrowprops=dict(facecolor='black', width=1.5, headwidth=7),
             horizontalalignment='center', verticalalignment='top')
plt.annotate('        ', xy=(datetime.datetime(2017, 12, 25, 12, 0), 460), xycoords='data',
             xytext=(datetime.datetime(2017, 12, 22, 20, 0), 170), textcoords='data',
             arrowprops=dict(facecolor='black', width=1.5, headwidth=7),
             horizontalalignment='center', verticalalignment='top')

plt.savefig('Figures/FigureC1.png', bbox_inches='tight')

## FIGURE C2 ##
#########################################
# Compare technician and algorithm detections

LSTM_multivar_bidir_detections = pd.read_csv('LSTM_multivar_bidir_detections_MainStreet_ph.csv',
                                             header=0,
                                             index_col=0,
                                             parse_dates=True,
                                             infer_datetime_format=True)
LSTM_multivar_bidir_df = pd.read_csv('LSTM_multivar_bidir_df_MainStreet_ph.csv',
                                             header=0,
                                             index_col=0,
                                             parse_dates=True,
                                             infer_datetime_format=True)

raw = LSTM_multivar_bidir_df['raw']
predictions = LSTM_multivar_bidir_detections['prediction']
labels = LSTM_multivar_bidir_df['labeled_event']
detections = LSTM_multivar_bidir_df['detected_event']

plt.figure(figsize=(12, 4))
plt.plot(raw, color=colors[0], label='Observed data')
plt.plot(predictions, color=colors[2], label='Model prediction')
plt.plot(raw[labels > 0], 'o', color=colors[1], mfc='none', markersize=5, markeredgewidth=1, label='Technician labeled anomalies')
plt.plot(predictions[detections > 0], 'x', color=colors[3], markersize=6, markeredgewidth=2, label='Algorithm detected anomalies')
plt.xlim(datetime.datetime(2018, 6, 1), datetime.datetime(2018, 10, 30))  # Specify date range of plot
plt.ylim(8.25, 10.5)
plt.xticks(pd.date_range(start='6/1/2018', end='10/30/2018', freq='15D'))  # Specify xticks at 15-day intervals
plt.legend()
plt.ylabel('pH')
plt.xlabel('Date')
plt.show()
plt.savefig('Figures/FigureC2.png', bbox_inches='tight')

## FIGURE C3 ##
#########################################
# Examine calibration events

figC3 = plt.figure(figsize=(10, 6))
ax = figC3.add_subplot(2, 1, 1)
ax.plot(raw, color=colors[0], label='Observed data')
ax.plot(predictions, color=colors[2], label='Model prediction')
ax.plot(raw[labels > 0], 'o', color=colors[1], mfc='none', markersize=5, markeredgewidth=1, label='Technician labeled anomalies')
ax.plot(predictions[detections > 0], 'x', color=colors[3], markersize=6, markeredgewidth=2, label='Algorithm detected anomalies')
ax.set_xlim(datetime.datetime(2017, 10, 2), datetime.datetime(2017, 10, 5))  # Specify date range of plot
ax.set_ylim(8.35, 8.7)
ax.set_xticks(pd.date_range(start='10/2/2017', end='10/5/2017', freq='1D'))  # Specify xticks at 1-day intervals
ax.set_yticks(np.arange(8.4, 8.75, 0.1))
ax.legend()
ax.set_ylabel('pH')
ax.annotate('a', xy=(0.015, 0.9), xycoords='axes fraction', fontsize=15)
ax = figC3.add_subplot(2, 1, 2)
ax.plot(raw, color=colors[0], label='Observed data')
ax.plot(predictions, color=colors[2], label='Model prediction')
ax.plot(raw[labels > 0], 'o', color=colors[1], mfc='none', markersize=5, markeredgewidth=1, label='Technician labeled anomalies')
ax.plot(predictions[detections > 0], 'x', color=colors[3], markersize=6, markeredgewidth=2, label='Algorithm detected anomalies')
ax.set_xlim(datetime.datetime(2019, 8, 27), datetime.datetime(2019, 8, 30))  # Specify date range of plot
ax.set_ylim(8.1, 9.0)
ax.set_xticks(pd.date_range(start='8/27/2019', end='8/30/2019', freq='1D'))  # Specify xticks at 1-day intervals
ax.set_ylabel('pH')
ax.annotate('b', xy=(0.015, 0.9), xycoords='axes fraction', fontsize=15)
plt.xlabel('Date')

plt.savefig('Figures/FigureC3.png', bbox_inches='tight')


# FIGURES 5 (detection example), C4 (model comparison examples)
# These figures all use data from Tony Grove.

#### Retrieve data
#########################################
os.chdir('..')
os.chdir('..')
site = 'TonyGrove'
sensors = ['temp', 'cond', 'ph', 'do']
years = [2014, 2015, 2016, 2017, 2018, 2019]
sensor_array = anomaly_utilities.get_data(sensors=sensors, site=site, years=years, path="./LRO_data/")

os.chdir('Examples/Plotting/')

ARIMA_detections = pd.read_csv('ARIMA_detections_TonyGrove_cond.csv',
                               header=0,
                               index_col=0,
                               parse_dates=True,
                               infer_datetime_format=True)
ARIMA_df = pd.read_csv('ARIMA_df_TonyGrove_cond.csv',
                               header=0,
                               index_col=0,
                               parse_dates=True,
                               infer_datetime_format=True)
LSTM_univar_detections = pd.read_csv('LSTM_univar_detections_TonyGrove_cond.csv',
                               header=0,
                               index_col=0,
                               parse_dates=True,
                               infer_datetime_format=True)
LSTM_univar_df = pd.read_csv('LSTM_univar_df_anomalies_TonyGrove_cond.csv',
                               header=0,
                               index_col=0,
                               parse_dates=True,
                               infer_datetime_format=True)
LSTM_univar_bidir_detections = pd.read_csv('LSTM_univar_bidir_detections_TonyGrove_cond.csv',
                               header=0,
                               index_col=0,
                               parse_dates=True,
                               infer_datetime_format=True)
LSTM_univar_bidir_df = pd.read_csv('LSTM_univar_bidir_df_anomalies_TonyGrove_cond.csv',
                               header=0,
                               index_col=0,
                               parse_dates=True,
                               infer_datetime_format=True)
LSTM_multivar_detections = pd.read_csv('LSTM_multivar_detections_TonyGrove_cond.csv',
                               header=0,
                               index_col=0,
                               parse_dates=True,
                               infer_datetime_format=True)
LSTM_multivar_df = pd.read_csv('LSTM_multivar_df_TonyGrove_cond.csv',
                               header=0,
                               index_col=0,
                               parse_dates=True,
                               infer_datetime_format=True)
LSTM_multivar_bidir_detections = pd.read_csv('LSTM_multivar_bidir_detections_TonyGrove_cond.csv',
                               header=0,
                               index_col=0,
                               parse_dates=True,
                               infer_datetime_format=True)
LSTM_multivar_bidir_df = pd.read_csv('LSTM_multivar_bidir_df_TonyGrove_cond.csv',
                               header=0,
                               index_col=0,
                               parse_dates=True,
                               infer_datetime_format=True)

## FIGURE 5 ##
#########################################
# Detection example

raw = sensor_array['cond']['raw']
labels = ARIMA_df['labeled_event']
predictions = ARIMA_detections['prediction']
detections = ARIMA_df['detected_event']

plt.figure(figsize=(10, 4))
plt.plot(raw, color=colors[0], label='Observed data')
plt.plot(predictions, color=colors[2], label='Model prediction')
plt.plot(raw[labels > 0], 'o', color=colors[1], mfc='none', markersize=5, markeredgewidth=1, label='Technician labeled anomalies')
plt.plot(predictions[detections > 0], 'x', color=colors[3], markersize=6, markeredgewidth=2, label='Algorithm detected anomalies')
plt.xlim(datetime.datetime(2018, 11, 9), datetime.datetime(2018, 11, 16))  # Specify date range of plot
plt.ylim(330, 425)
plt.xticks(pd.date_range(start='11/9/2018', end='11/16/2018', freq='1D'))  # Specify xticks at 1-day intervals
plt.legend()
plt.ylabel('Specific conductance, μS/cm')
plt.xlabel('Date')
plt.annotate('false\npositive\nevent', xy=(datetime.datetime(2018, 11, 11, 11, 0), 359),  xycoords='data',
             xytext=(datetime.datetime(2018, 11, 11, 0, 0), 345), textcoords='data',
             arrowprops=dict(facecolor='black', width=1.5, headwidth=7),
             horizontalalignment='center', verticalalignment='top')
plt.annotate('true\npositive\nevents', xy=(datetime.datetime(2018, 11, 12, 15, 0), 356),  xycoords='data',
             xytext=(datetime.datetime(2018, 11, 13, 2, 0), 345), textcoords='data',
             arrowprops=dict(facecolor='black', width=1.5, headwidth=7),
             horizontalalignment='center', verticalalignment='top')
plt.annotate('         ', xy=(datetime.datetime(2018, 11, 13, 10, 0), 380),  xycoords='data',
             xytext=(datetime.datetime(2018, 11, 13, 2, 0), 345), textcoords='data',
             arrowprops=dict(facecolor='black', width=1.5, headwidth=7),
             horizontalalignment='center', verticalalignment='top')
plt.annotate('false\nnegative\nevent', xy=(datetime.datetime(2018, 11, 14, 17, 0), 357),  xycoords='data',
             xytext=(datetime.datetime(2018, 11, 15, 5, 0), 345), textcoords='data',
             arrowprops=dict(facecolor='black', width=1.5, headwidth=7),
             horizontalalignment='center', verticalalignment='top')
plt.show()
plt.savefig('Figures/Figure5.png', bbox_inches='tight')


## FIGURE C4 ##
#########################################
# Model comparison

raw = sensor_array['cond']['raw']
labels = ARIMA_df['labeled_event']

predictions = dict()
detections = dict()
predictions['ARIMA'] = ARIMA_detections['prediction']
detections['ARIMA'] = ARIMA_df['detected_event']
predictions['LSTM_univar'] = LSTM_univar_detections['prediction']
detections['LSTM_univar'] = LSTM_univar_df['detected_event']
predictions['LSTM_univar_bidir'] = LSTM_univar_bidir_detections['prediction']
detections['LSTM_univar_bidir'] = LSTM_univar_bidir_df['detected_event']
predictions['LSTM_univar_bidir'] = LSTM_univar_bidir_detections['prediction']
detections['LSTM_univar_bidir'] = LSTM_univar_bidir_df['detected_event']
predictions['LSTM_multivar'] = LSTM_multivar_detections['prediction']
detections['LSTM_multivar'] = LSTM_multivar_df['detected_event']
predictions['LSTM_multivar_bidir'] = LSTM_multivar_bidir_detections['prediction']
detections['LSTM_multivar_bidir'] = LSTM_multivar_bidir_df['detected_event']

model_type = ['ARIMA', 'LSTM_univar', 'LSTM_univar_bidir', 'LSTM_multivar', 'LSTM_multivar_bidir']
model_text = ['ARIMA', 'LSTM univariate', 'LSTM univariate bidirectional', 'LSTM multivariate', 'LSTM multivariate bidirectional']

fig, ax = plt.subplots(nrows=5, ncols=3, figsize=(15, 8), sharex='col', sharey='col',
                    gridspec_kw={'width_ratios': [3, 1, 1], 'height_ratios': [1, 1, 1, 1, 1], 'hspace': 0, 'wspace': 0.11})
fig.text(0.08, 0.5, 'Specific conductance, μS/cm', va='center', rotation='vertical', fontsize=14)
fig.text(0.5, 0.05, 'Date', va='center', fontsize=14)
for i, mdl in enumerate(model_type):
    ax[i][0].plot(raw, color=colors[0], label='Observed data')
    ax[i][0].plot(predictions[mdl], color=colors[2], label='Model prediction')
    ax[i][0].plot(raw[labels > 0], 'o', color=colors[1], mfc='none', markersize=5, markeredgewidth=1, label='Technician labeled anomalies')
    ax[i][0].plot(predictions[mdl][detections[mdl] > 0], 'x', color=colors[3], markersize=6, markeredgewidth=2, label='Algorithm detected anomalies')
    ax[i][0].set_xlim(datetime.datetime(2014, 9, 18, 18), datetime.datetime(2014, 9, 20, 6))
    ax[i][0].set_ylim(295, 375)
    ax[i][0].set_yticks(ticks=[300, 320, 340, 360])
    ax[i][0].set_xticks(pd.date_range(start='9/18/2014 18:00', end='9/20/2014 6:00', freq='8H'))
    ax[i][1].plot(raw, color=colors[0], label='Observed data')
    ax[i][1].plot(predictions[mdl], color=colors[2], label='Model prediction')
    ax[i][1].plot(raw[labels > 0], 'o', color=colors[1], mfc='none', markersize=5, markeredgewidth=1, label='Technician labeled anomalies')
    ax[i][1].plot(predictions[mdl][detections[mdl] > 0], 'x', color=colors[3], markersize=6, markeredgewidth=2, label='Algorithm detected anomalies')
    ax[i][1].set_xlim(datetime.datetime(2015, 2, 5, 6, 0), datetime.datetime(2015, 2, 5, 18))
    ax[i][1].set_ylim(350, 440)
    ax[i][1].set_yticks(ticks=[350, 375, 400, 425])
    ax[i][1].set_xticks(pd.date_range(start='2/5/2015 8:00', end='2/5/2015 18:00', freq='6H'))
    ax[i][2].plot(raw, color=colors[0], label='Observed data')
    ax[i][2].plot(predictions[mdl], color=colors[2], label='Model prediction')
    ax[i][2].plot(raw[labels > 0], 'o', color=colors[1], mfc='none', markersize=5, markeredgewidth=1, label='Technician labeled anomalies')
    ax[i][2].plot(predictions[mdl][detections[mdl] > 0], 'x', color=colors[3], markersize=6, markeredgewidth=2, label='Algorithm detected anomalies')
    ax[i][2].set_xlim(datetime.datetime(2015, 7, 6), datetime.datetime(2015, 7, 30))
    ax[i][2].set_ylim(310, 385)
    ax[i][2].set_yticks(ticks=[325, 350, 375])
    ax[i][2].set_xticks(pd.date_range(start='7/8/2015', end='7/30/2015', freq='5D'))

    ax[i][0].annotate(model_text[i],
                      xy=(datetime.datetime(2014, 9, 18, 20), 365),
                      xytext=(datetime.datetime(2014, 9, 18, 20), 365),
                      annotation_clip=False, rotation=0,
                      ha='left', va='center', fontname='Arial Narrow',
                      horizontalalignment='right', verticalalignment='top')
    ax[4][0].legend(ncol=1, labelspacing=0.2, fontsize=9, handletextpad=0.2, columnspacing=0.25, loc='lower right')
plt.savefig('Figures/FigureC4.png', bbox_inches='tight')

###################################################
