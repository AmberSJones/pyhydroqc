###################################
# RULES BASED ANOMALY DETECTION #
###################################
# This script includes functionality for rules based anomaly detection and preprocessing.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anomaly_utilities

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

# EXECUTE FUNCTIONS #
#########################################
# Get data
df_full, sensor_array = anomaly_utilities.get_data(site, sensor, year, path="/Users/amber/PycharmProjects/LRO-anomaly-detection/LRO_data/")
df = sensor_array[sensor[0]]

# RANGE CHECKS
# need to do some sort of look up table with the values for each sensor
# e.g., if sensor == 'cond':


def range_check(df, maximum, minimum):
    """Adds column to data frame with label if data are out of range.
    df is a data frame with a column 'raw' of raw data.
    maximum and minimum define the range outside of which a value is anomalous.
    Output is the dataframe with an added column 'anomoly' with boolean where 1 = anomalous."""
    df = df.eval('anomaly = raw > @maximum or raw < @minimum')

    return df


maximum = 900
minimum = 150
df = range_check(df, maximum, minimum)


def persistence(df, length):
    """Adds flag in data frame if data repeat for specified length.
    df is a data frame with a column 'raw' of raw data and a boolean column 'anomaly'.
    length is the duration of persistant/repeated values to be flagged.
    Output is the dataframe with column 'anomaly' modified."""
    # temp = df.copy(deep=True)
    temp = df[['raw', 'anomaly']].copy(deep=True)
    temp['value_grp'] = (temp.raw.diff(1) == 0)
    temp['value_grp'] = anomaly_utilities.anomaly_events(temp['value_grp'], 0, 1)
    for i in range(1, max(temp['value_grp']) + 1):
        if(len(temp['value_grp'][temp['value_grp'] == i]) >= length):
            temp['anomaly'][temp['value_grp'] == i] = True
    df['anomaly'] = temp['anomaly']

    return df


length = 8
df = persistence(df, length)






# toy dataset used to develop this file
df = pd.DataFrame(
    [
        [494.0, 494.0, 0],
        [493.5, 493.5, 0],
        [492.5, 492.5, 0],
        [492.0, 492.0, 0],
        [405.0, 487.0, 1],
        [405.0, 478.0, 1],
        [405.0, 470.0, 1],
        [464.0, 464.0, 0],
        [462.0, 462.0, 0],
        [485.0, 485.0, 0],
        [472.5, 472.5, 1],
        [405.0, 487.0, 1],
        [100.0, 478.0, 1],
        [405.0, 470.0, 0],
        [472.5, 472.5, 0],
        [405.0, 487.0, 1],
        [405.0, 478.0, 1],
        [405.0, 470.0, 1],
        [405.0, 472.5, 0],
        [470.0, 470.0, 1],
        [494.0, 494.0, 0],
        [950.0, 493.5, 1],
        [492.5, 492.5, 0],
        [492.5, 492.0, 0],
    ],
    index=[
        '2020-01-01T00:00:00.0',
        '2020-01-01T00:15:00.0',
        '2020-01-01T00:30:00.0',
        '2020-01-01T00:45:00.0',
        '2020-01-01T01:00:00.0',
        '2020-01-01T01:15:00.0',
        '2020-01-01T01:30:00.0',
        '2020-01-01T01:45:00.0',
        '2020-01-01T02:00:00.0',
        '2020-01-01T02:15:00.0',
        '2020-01-01T02:30:00.0',
        '2020-01-01T02:45:00.0',
        '2020-01-01T03:00:00.0',
        '2020-01-01T03:15:00.0',
        '2020-01-01T03:30:00.0',
        '2020-01-01T03:45:00.0',
        '2020-01-01T04:00:00.0',
        '2020-01-01T04:15:00.0',
        '2020-01-01T04:30:00.0',
        '2020-01-01T04:45:00.0',
        '2020-01-01T05:00:00.0',
        '2020-01-01T05:15:00.0',
        '2020-01-01T05:30:00.0',
        '2020-01-01T05:45:00.0',
    ],
    columns={'raw': 0, 'cor': 1, 'labeled_anomaly': 2})
