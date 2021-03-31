#### Import Libraries and Functions

from PyHydroQC import anomaly_utilities
import numpy as np
import matplotlib.pyplot as plt
from PyHydroQC import rules_detect
from PyHydroQC.parameters import site_params, calib_params
import pandas as pd
import math

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
    sensor_array[snsr], range_count[snsr] = rules_detect.range_check(
        df=sensor_array[snsr], maximum=site_params[site][snsr]['max_range'], minimum=site_params[site][snsr]['min_range'])
    sensor_array[snsr], persist_count[snsr] = rules_detect.persistence(
        df=sensor_array[snsr], length=site_params[site][snsr]['persist'], output_grp=True)
    sensor_array[snsr] = rules_detect.add_labels(df=sensor_array[snsr], value=-9999)
    sensor_array[snsr] = rules_detect.interpolate(df=sensor_array[snsr])

print('Rules based detection complete.\n')

#### Edge Detection for Calibration Events
#########################################

# Subset of sensors that are calibrated
calib_sensors = sensors[1:4]

# Initialize data structures
calib_candidates = dict()
edge_diff = dict()
threshold = dict()
threshold['cond'] = 60
threshold['ph'] = 0.1
threshold['do'] = 0.3

for cal_snsr in calib_sensors:
    calib_candidates[cal_snsr], edge_diff[cal_snsr] = calib_edge_detect(observed=sensor_array[cal_snsr]['observed'],
                                                                width=1,
                                                                calib_params=calib_params,
                                                                threshold=threshold[cal_snsr])
    plt.figure()
    plt.plot(edge_diff[cal_snsr], 'b')
    plt.axhline(y=threshold[cal_snsr], color='r', linestyle='-')
    plt.axhline(y=-threshold[cal_snsr], color='r', linestyle='-')
    plt.ylabel(cal_snsr)
    plt.show()



