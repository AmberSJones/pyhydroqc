import numpy as np
import pandas as pd
import statsmodels.api as api
import anomaly_utilities


def ARIMA_forecast(x, l, p, d, q):
    """ correction_forecast is used to create predictions of data where anomalies occur
    x is an array of values from which to predict corrections
    l is used to determine how many predicted data points are needed
    Outputs:
    y is an array of length l of the corrected values as predicted by the model
    """
    model = api.tsa.SARIMAX(x, order=(p, d, q))
    model_fit = model.fit(disp=0)
    results = model_fit.get_forecast(l)
    y = results.prediction_results.results.forecasts[0]

    return y


def generate_corrections(df, p, d, q):
    """ df is a data frame with required columns:
    'raw': raw data
    'detected_anomaly': boolean array corresponding to classified anomalies
    Outputs:
    df with additional column: 'det_cor' of corrected data
    """

    # initialize array for algorithm determined corrections to determined anomalies
    det_cor = []
    # df['det_cor'] = df['raw']

    # group data points into strings of True and False values
    df = anomaly_utilities.group_bools(df)

    # for each anomaly grouping of data points
    for i in range(0, (max(df['groups']) + 1)):

        # if this is an anomalous group of points
        if(df.loc[df['groups'] == i]['detected_anomaly'][0]):
            # perform forecasting to generate corrected data points
            if (i != 0):
                # forecast in forward direction
                # create an array of corrected data for current anomalous group
                yfor = ARIMA_forecast(np.array(df.loc[df['groups'] == (i - 1)]['raw']),
                                           len(df.loc[df['groups'] == i]),
                                           p, d, q)

            if (i != max(df['groups'])):
                # forcast in reverse direction
                yrev = ARIMA_forecast(np.flip(np.array(df.loc[df['groups'] == (i + 1)]['raw'])),
                                           len(df.loc[df['groups'] == i]),
                                           p, d, q)
                ybac = np.flip(yrev)

            det_cor.extend(anomaly_utilities.xfade(yfor, ybac))

        # not an anomalous group of data points
        else:
            # append the raw data
            det_cor.extend(df.loc[df['groups'] == i]['raw'])

    # add column for determined corrections in dataframe
    df['det_cor'] = det_cor
    return df

# # toy dataset used to develop this file
# df = pd.DataFrame(
#     [
#         [494.0, 494.0, False],
#         [493.5, 493.5, False],
#         [492.5, 492.5, False],
#         [492.0, 492.0, False],
#         [405.0, 487.0, True],
#         [405.0, 478.0, True],
#         [405.0, 470.0, True],
#         [464.0, 464.0, False],
#         [462.0, 462.0, False],
#         [485.0, 485.0, False],
#         [472.5, 472.5, False],
#     ],
#     index=[
#         '2020-01-01T00:00:00.0',
#         '2020-01-01T00:15:00.0',
#         '2020-01-01T00:30:00.0',
#         '2020-01-01T00:45:00.0',
#         '2020-01-01T01:00:00.0',
#         '2020-01-01T01:15:00.0',
#         '2020-01-01T01:30:00.0',
#         '2020-01-01T01:45:00.0',
#         '2020-01-01T02:00:00.0',
#         '2020-01-01T02:15:00.0',
#         '2020-01-01T02:30:00.0',
#     ],
#     columns={'raw': 0, 'cor': 1, 'detected_anomaly': 2})
#
# cdata = generate_corrections(df, p, d, q)

