
import anomaly_utilities
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
matplotlib.use('TkAgg')
# matplotlib.use('Qt5Agg')
# matplotlib.use('Qt4Agg')
# import matplotlib.pyplot as plt
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

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

# Get data
df_full, df = anomaly_utilities.get_data(site, sensor, year, path="/home/tjones/Documents/School/ECE6930/LRO-anomaly-detection/LRO_data/")

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(df['raw'], 'b', label='original data')
matplotlib.pyplot.legend()
matplotlib.pyplot.ylabel(sensor)
matplotlib.pyplot.show()

# Prophet .fit(df) requires columns ds (dates) and y
df['y'] = df['raw']
df['ds'] = df.index

m = Prophet(changepoint_range=1.0, n_changepoints=150, changepoint_prior_scale=30,
            seasonality_prior_scale=35, growth='linear', holidays=None,
            daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False,
            ).add_seasonality(name="daily", period=1, fourier_order=5, prior_scale=10)
m.fit(df)
forecast = m.predict()

# data, model, and changepoint plot
fig1 = m.plot(forecast)
a = add_changepoints_to_plot(fig1.gca(), m, forecast)
fig1.show()

# component plot
fig2 = m.plot_components(forecast)
fig2.show()
