#### Import Libraries and Functions

from PyHydroQC import anomaly_utilities
import matplotlib.pyplot as plt
from PyHydroQC import rules_detect, calibration
from PyHydroQC.parameters import site_params, calib_params

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

#### Edge Detection for Calibration Events
#########################################

# Subset of sensors that are calibrated
calib_sensors = sensors[1:4]

# Initialize data structures
calib_candidates = dict()
edge_diff = dict()
threshold = dict()
# Set threshold for each variable - the level of change for a difference to be identified as a calibration event.
# This can be an iterative process.
threshold['cond'] = 60
threshold['ph'] = 0.1
threshold['do'] = 0.4

for cal_snsr in calib_sensors:
    # Width is the window of time to consider in the edge detect difference.
    # 1 determines the difference between each point independently.
    # Higher numbers use the difference over a longer window.
    calib_candidates[cal_snsr], edge_diff[cal_snsr] = calibration.calib_edge_detect(observed=sensor_array[cal_snsr]['observed'],
                                                                                    width=1,
                                                                                    calib_params=calib_params,
                                                                                    threshold=threshold[cal_snsr])
    # Plot the differences and the thresholds. Include actual data to these plots to examine the events.
    fig, ax1 = plt.subplots()
    ax1.set_ylabel(cal_snsr, color='c')  # we already handled the x-label with ax1
    ax1.plot(sensor_array[cal_snsr]['observed'], color='c')
    ax1.tick_params(axis='y', labelcolor='c')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('threshold', color='b')
    ax2.plot(edge_diff[cal_snsr], color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.axhline(y=threshold[cal_snsr], color='r', linestyle='-')
    ax2.axhline(y=-threshold[cal_snsr], color='r', linestyle='-')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
