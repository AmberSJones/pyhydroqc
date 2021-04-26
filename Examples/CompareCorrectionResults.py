
#### Import Libraries and Functions
from PyHydroQC import anomaly_utilities
from PyHydroQC.parameters import site_params
import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('..')
os.chdir('..')

#### Retrieve data
#########################################
site = 'FranklinBasin'
sensors = ['temp', 'cond', 'ph', 'do']
years = [2014, 2015, 2016, 2017, 2018, 2019]
sensor_array = anomaly_utilities.get_data(sensors=sensors, site=site, years=years, path="./LRO_data/")

os.chdir('./Examples/Plotting')

corrections = dict()
correct_compare = dict()
tech_changed_pct = dict()
det_changed_pct = dict()
tech_changed_ct = dict()
det_changed_ct = dict()
print(site)
for snsr in sensors:
    corrections[snsr] = pd.read_csv(site + '_' + snsr + '_' + 'corrections.csv',
                                                 header=0,
                                                 index_col=0,
                                                 parse_dates=True,
                                                 infer_datetime_format=True)
    correct_compare[snsr] = pd.DataFrame(index=sensor_array[snsr].index)
    correct_compare[snsr]['raw'] = sensor_array[snsr]['raw']
    correct_compare[snsr]['tech_cor'] = sensor_array[snsr]['cor']
    correct_compare[snsr]['det_cor'] = corrections[snsr]['det_cor']

    df = correct_compare[snsr]
    # plt.plot(df['raw'], 'b', label='raw')
    # plt.plot(df['tech_cor'], 'r', label='technician')
    # plt.plot(df['det_cor'], 'g', label='algorithm')
    # plt.legend()

    min_range = site_params[site][snsr]['min_range']
    max_range = site_params[site][snsr]['max_range']

    tech_changed_valid = len(df[((df['raw'] - df['tech_cor']) != 0) &
                                (df['tech_cor'] < max_range) &
                                (df['tech_cor'] > min_range)])
    tech_changed_invalid = len(df[((df['raw'] - df['tech_cor']) != 0) &
                                  ((df['tech_cor'] > max_range) |
                                   (df['tech_cor'] < min_range))])
    det_changed_valid = len(df[((df['raw'] - df['det_cor']) != 0) &
                                (df['det_cor'] < max_range) &
                                (df['det_cor'] > min_range)])
    det_changed_invalid = len(df[((df['raw'] - df['det_cor']) != 0) &
                                 ((df['det_cor'] > max_range) |
                                  (df['det_cor'] < min_range))])

    tech_changed_ct[snsr] = tech_changed_invalid
    det_changed_ct[snsr] = det_changed_invalid
    tech_changed_pct[snsr] = tech_changed_invalid/(tech_changed_valid+tech_changed_invalid)
    det_changed_pct[snsr] = det_changed_invalid/(det_changed_valid+det_changed_invalid)

    print(snsr + ' technician changed total ' + str(tech_changed_valid+tech_changed_invalid))
    print(snsr + ' technician changed valid ' + str(tech_changed_valid))
    print(snsr + ' technician changed invalid ' + str(tech_changed_invalid))
    print(snsr + ' corrected changed total ' + str(det_changed_valid+det_changed_invalid))
    print(snsr + ' corrected changed valid ' + str(det_changed_valid))
    print(snsr + ' corrected changed invalid ' + str(det_changed_invalid))
