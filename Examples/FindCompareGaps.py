#### Import Libraries and Functions

from PyHydroQC import anomaly_utilities
import matplotlib.pyplot as plt
from PyHydroQC import rules_detect
from PyHydroQC.parameters import site_params
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
    calib_dates[cal_snsr] = pd.read_csv(
        './LRO_data/' + site + '_' + cal_snsr + '_calib_dates.csv', header=1, parse_dates=True, infer_datetime_format=True)
    calib_dates[cal_snsr]['start'] = pd.to_datetime(calib_dates[cal_snsr]['start'])
    calib_dates[cal_snsr]['end'] = pd.to_datetime(calib_dates[cal_snsr]['end'])
    # Subset calibrations to correspond to imported data
    calib_dates[cal_snsr] = calib_dates[cal_snsr].loc[(calib_dates[cal_snsr]['start'] > min(sensor_array[cal_snsr].index)) &
                                                      (calib_dates[cal_snsr]['end'] < max(sensor_array[cal_snsr].index))]
    # Initialize dataframe for gap values
    gaps[cal_snsr] = pd.DataFrame(columns=['end', 'gap'], index=range(min(calib_dates[cal_snsr].index), max(calib_dates[cal_snsr].index)+1))
    if len(calib_dates[cal_snsr]) > 0:
        # Initialize data structures
        shifts[cal_snsr] = []
        tech_shifts[cal_snsr] = []
        # Loop through each calibration event date.
        for i in range(min(calib_dates[cal_snsr].index), max(calib_dates[cal_snsr].index)+1):
            # Apply find_gap routine, add to dataframe, add output of shifts to list.
            gap, end, shifted = rules_detect.find_gap(observed=sensor_array[cal_snsr]['observed'],
                                                      calib_date=calib_dates[cal_snsr]['end'][i],
                                                      hours=2,
                                                      show_shift=True)
            gaps[cal_snsr].loc[i]['end'] = end
            gaps[cal_snsr].loc[i]['gap'] = gap
            shifts[cal_snsr].append(shifted)

            # Create subsets to show the shifts of the technician selected gap values for comparison
            tech_subset = sensor_array[cal_snsr]['observed'].loc[
             pd.to_datetime(calib_dates[cal_snsr]['end'][i]) - pd.Timedelta(hours=2):
             pd.to_datetime(calib_dates[cal_snsr]['end'][i]) + pd.Timedelta(hours=2)
             ]
            tech_shifted = tech_subset.loc[
                           tech_subset.index[0]:pd.to_datetime(calib_dates[cal_snsr]['end'][i])] \
                           + calib_dates[cal_snsr]['gap'][i]
            tech_shifts[cal_snsr].append(tech_shifted)

    # Plotting to examine when calibrations occur and compare algorithm and technician gap values
    l = len(calib_dates[cal_snsr])
    ncol = math.ceil(math.sqrt(l))
    nrow = ncol
    hours = 6

    fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(nrow, ncol+1), facecolor='w')
    for i, axi in enumerate(ax.flat):
        if i < l:
            axi.plot(sensor_array[cal_snsr]['observed'].loc[
                    pd.to_datetime(calib_dates[cal_snsr]['end'].loc[i]) - pd.Timedelta(hours=hours):
                    pd.to_datetime(calib_dates[cal_snsr]['end'].loc[i]) + pd.Timedelta(hours=hours)
                    ])
            axi.plot(shifts[cal_snsr][i], 'c')
            axi.plot(tech_shifts[cal_snsr][i], 'r')

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
calib_dates = dict()
for cal_snsr in calib_sensors:
    gaps[cal_snsr]['start'] = gaps[cal_snsr]['end'].shift(1)
    gaps[cal_snsr]['start'][0] = gaps[cal_snsr]['end'][0] - pd.Timedelta(days=30)
    if len(gaps[cal_snsr]) > 0:
        for i in range(min(gaps[cal_snsr].index), max(gaps[cal_snsr].index)):
            result, sensor_array[cal_snsr]['observed'] = rules_detect.lin_drift_cor(
                                                            observed=sensor_array[cal_snsr]['observed'],
                                                            start=gaps[cal_snsr]['start'][i],
                                                            end=gaps[cal_snsr]['end'][i],
                                                            gap=gaps[cal_snsr]['gap'][i],
                                                            replace=True)
