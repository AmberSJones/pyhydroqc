#########################################
# PyHydroQC Time Series Figures
#########################################

#### Import Libraries and Functions
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np
import os

# set working directory for importing model results.
os.chdir('./Examples/Plotting')

colors = ['#0C7BDC',  # forecasted
          '#AF3C31',  # backcasted
          '#F3870D',  # observed
          '#24026A']  # corrected


MainStreet_temp = pd.read_csv('MainStreet_temp_corrections.csv',
                               header=0,
                               index_col=0,
                               parse_dates=True,
                               infer_datetime_format=True)
MainStreet_ph = pd.read_csv('MainStreet_ph_corrections.csv',
                               header=0,
                               index_col=0,
                               parse_dates=True,
                               infer_datetime_format=True)
TonyGrove_temp = pd.read_csv('TonyGrove_temp_corrections.csv',
                             header=0,
                             index_col=0,
                             parse_dates=True,
                             infer_datetime_format=True)
TonyGrove_ph = pd.read_csv('TonyGrove_ph_corrections.csv',
                               header=0,
                               index_col=0,
                               parse_dates=True,
                               infer_datetime_format=True)
TonyGrove_do = pd.read_csv('TonyGrove_do_corrections.csv',
                               header=0,
                               index_col=0,
                               parse_dates=True,
                               infer_datetime_format=True)
Mendon_cond = pd.read_csv('Mendon_cond_corrections.csv',
                               header=0,
                               index_col=0,
                               parse_dates=True,
                               infer_datetime_format=True)


WaterLab_temp = pd.read_csv('WaterLab_temp_corrections.csv',
                             header=0,
                             index_col=0,
                             parse_dates=True,
                             infer_datetime_format=True)


from PyHydroQC import anomaly_utilities
os.chdir('..')
os.chdir('..')
site = 'WaterLab'
sensors = ['temp']
years = [2015, 2016, 2017, 2018, 2019]
sensor_array = anomaly_utilities.get_data(sensors=sensors, site=site, years=years, path="./LRO_data/")

fig12 = plt.figure(figsize=(10, 10))

df = WaterLab_temp
df['corrected'] = np.where(np.isnan(df['forecasts']) & np.isnan(df['backcasts']), np.nan, df['det_cor'])
observed = df['observed']
corrected = df['corrected']
technician = sensor_array['temp']['cor']
tech_sub = technician['2018-08-30 12:15':'2018-08-30 14:00']
forecast = df['forecasts']
backcast = df['backcasts']

ax = fig12.add_subplot(3, 1, 1)
ax.plot(corrected, color=colors[3], label='Model corrected', linewidth=3)
ax.plot(tech_sub, color='red', label='Technician corrected')
ax.plot(observed, color=colors[2], label='Observed data')
ax.plot(forecast, color=colors[0], label='Forecasted', ls='--')
ax.plot(backcast, color=colors[1], label='Backcasted', ls='--')
ax.set_xlim(datetime.datetime(2018, 8, 30, 8), datetime.datetime(2018, 8, 30, 20))
ax.set_ylim(10.5, 13.5)
ax.set_xticks(pd.date_range(start='8/30/2018 8:00', end='8/30/2018 20:00', freq='4H'))
# ax.set_yticks(np.arange(8.4, 8.75, 0.1))
ax.legend(loc='upper left')
ax.set_ylabel('Temperature, deg C')

df = MainStreet_ph
df['corrected'] = np.where(np.isnan(df['forecasts']) & np.isnan(df['backcasts']), np.nan, df['det_cor'])
observed = df['observed']
corrected = df['corrected']
forecast = df['forecasts']
backcast = df['backcasts']

ax = fig12.add_subplot(3, 1, 2)
ax.plot(observed, color=colors[2], label='Observed data')
ax.plot(corrected, color=colors[3], label='Model corrected', linewidth=2)
ax.plot(forecast, color=colors[0], label='Forecasted', ls='--')
ax.plot(backcast, color=colors[1], label='Backcasted', ls='--')
ax.set_xlim(datetime.datetime(2018, 6, 28), datetime.datetime(2018, 7, 10))
ax.set_ylim(8.9, 10.0)
ax.set_xticks(pd.date_range(start='6/28/2018', end='7/10/2018', freq='2D'))
ax.set_ylabel('pH')

df = WaterLab_temp
df['corrected'] = np.where(np.isnan(df['forecasts']) & np.isnan(df['backcasts']), np.nan, df['det_cor'])
observed = df['observed']
corrected = df['corrected']
forecast = df['forecasts']
backcast = df['backcasts']

ax = fig12.add_subplot(3, 1, 3)
ax.plot(observed, color=colors[2], label='Observed data')
ax.plot(corrected, color=colors[3], label='Model corrected', linewidth=2)
ax.plot(forecast, color=colors[0], label='Forecasted', ls='--')
ax.plot(backcast, color=colors[1], label='Backcasted', ls='--')
ax.set_xlim(datetime.datetime(2017, 6, 27), datetime.datetime(2017, 7, 6))
ax.set_ylim(7, 13)
ax.set_xticks(pd.date_range(start='6/27/2017', end='7/6/2017', freq='2D'))
# ax.set_yticks(np.arange(8.4,8.75,0.1))
ax.set_ylabel('Temperature, deg C')
ax.set_xlabel('Date')

plt.savefig('Figures/Figure12.png', bbox_inches='tight')


fig13 = plt.figure(figsize=(10, 10))

df = TonyGrove_do
df['corrected'] = np.where(np.isnan(df['forecasts']) & np.isnan(df['backcasts']), np.nan, df['det_cor'])
observed = df['observed']
corrected = df['corrected']
forecast = df['forecasts']
backcast = df['backcasts']

ax = fig13.add_subplot(3, 1, 1)
ax.plot(observed, color=colors[2], label='Observed data')
ax.plot(corrected, color=colors[3], label='Model corrected', linewidth=2)
ax.plot(forecast, color=colors[0], label='Forecasted', ls='--')
ax.plot(backcast, color=colors[1], label='Backcasted', ls='--')
ax.set_xlim(datetime.datetime(2018, 7, 25), datetime.datetime(2018, 8, 3))
ax.set_ylim(7, 14)
ax.set_xticks(pd.date_range(start='7/25/2018', end='8/3/2018', freq='2D'))
ax.set_ylabel('Dissolved oxygen, mg/L')
ax.legend()

ax = fig13.add_subplot(3, 1, 2)
ax.plot(observed, color=colors[2], label='Observed data')
ax.plot(corrected, color=colors[3], label='Model corrected', linewidth=3)
ax.plot(forecast, color=colors[0], label='Forecasted', ls='--')
ax.plot(backcast, color=colors[1], label='Backcasted', ls='--')
ax.set_xlim(datetime.datetime(2016, 12, 27), datetime.datetime(2017, 1, 15))
ax.set_ylim(11, 12.4)
ax.set_xticks(pd.date_range(start='12/27/2016', end='1/15/2017', freq='4D'))
ax.set_ylabel('Dissolved oxygen, mg/L')

df = Mendon_cond
df['corrected'] = np.where(np.isnan(df['forecasts']) & np.isnan(df['backcasts']), np.nan, df['det_cor'])
observed = df['observed']
corrected = df['corrected']
forecast = df['forecasts']
backcast = df['backcasts']

ax = fig13.add_subplot(3, 1, 3)
ax.plot(observed, color=colors[2], label='Observed data')
ax.plot(corrected, color=colors[3], label='Model corrected', linewidth=3)
ax.plot(forecast, color=colors[0], label='Forecasted', ls='--')
ax.plot(backcast, color=colors[1], label='Backcasted', ls='--')
ax.set_xlim(datetime.datetime(2018, 6, 17), datetime.datetime(2018, 6, 22))
ax.set_ylim(190, 425)
ax.set_xticks(pd.date_range(start='6/17/2018', end='6/22/2018', freq='1D'))
# ax.set_yticks(np.arange(8.4,8.75,0.1))
ax.set_ylabel('Specific conductance, μS/cm')
ax.set_xlabel('Date')

# df = Mendon_cond
# df['corrected'] = np.where(np.isnan(df['forecasts']) & np.isnan(df['backcasts']), np.nan, df['det_cor'])
# observed = df['observed']
# corrected = df['corrected']
# forecast = df['forecasts']
# backcast = df['backcasts']
#
# ax = fig13.add_subplot(2, 2, 4)
# ax.plot(observed, color=colors[2], label='Observed data')
# ax.plot(corrected, color=colors[3], label='Model corrected', linewidth=3)
# ax.plot(forecast, color=colors[0], label='Forecasted', ls='--')
# ax.plot(backcast, color=colors[1], label='Backcasted', ls='--')
# ax.set_xlim(datetime.datetime(2019, 1, 30), datetime.datetime(2019, 3, 20))
# ax.set_ylim(435, 560)
# ax.set_xticks(pd.date_range(start='1/30/2019', end='3/20/2019', freq='7D'))
# ax.set_ylabel('Specific conductance, μS/cm')
# ax.set_xlabel('Date')

plt.savefig('Figures/Figure13.png', bbox_inches='tight')


fig14 = plt.figure(figsize=(10, 8))

df = Mendon_cond
df['corrected'] = np.where(np.isnan(df['forecasts']) & np.isnan(df['backcasts']), np.nan, df['det_cor'])
observed = df['observed']
corrected = df['corrected']
forecast = df['forecasts']
backcast = df['backcasts']

ax = fig14.add_subplot(2, 1, 1)
ax.plot(observed, color=colors[2], label='Observed data')
ax.plot(corrected, color=colors[3], label='Model corrected', linewidth=3)
ax.plot(forecast, color=colors[0], label='Forecasted', ls='--')
ax.plot(backcast, color=colors[1], label='Backcasted', ls='--')
ax.set_xlim(datetime.datetime(2018, 6, 17), datetime.datetime(2018, 6, 22))
ax.set_ylim(190, 425)
ax.set_xticks(pd.date_range(start='6/17/2018', end='6/22/2018', freq='1D'))
# ax.set_yticks(np.arange(8.4,8.75,0.1))
ax.legend()
ax.set_ylabel('Specific conductance, μS/cm')

df = TonyGrove_temp
#df['corrected'] = np.where(np.isnan(df['forecasts']) & np.isnan(df['backcasts']), np.nan, df['det_cor'])
observed = df['observed']
corrected = df['corrected']
forecast = df['forecasts']
backcast = df['backcasts']

ax = fig14.add_subplot(2, 1, 2)
ax.plot(observed, color=colors[2], label='Observed data')
ax.plot(corrected, color=colors[3], label='Model corrected', linewidth=3)
ax.plot(forecast, color=colors[0], label='Forecasted', ls='--')
ax.plot(backcast, color=colors[1], label='Backcasted', ls='--')
ax.set_xlim(datetime.datetime(2017, 1, 14), datetime.datetime(2017, 1, 25))
ax.set_ylim(-0.05, 3)
ax.set_xticks(pd.date_range(start='1/14/2017', end='1/25/2017', freq='2D'))
ax.set_ylabel('Temperature, deg C')
ax.set_xlabel('Date')

plt.savefig('Figures/Figure14.png', bbox_inches='tight')

########################################################