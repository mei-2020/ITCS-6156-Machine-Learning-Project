import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, mean_absolute_error, SCORERS
import sys, os, patsy
import statsmodels.api as sm
import matplotlib.pyplot as plt
from math import pi, floor, ceil
import seaborn as sns


def import_preprocess(station, yr, daysahead=(0, 1), fulldata=False, subset=True, preprocess=True,
                      actual_diff=False, fcst_diff=True, time_adjust=True):
    """
    This function is for Phase I.
    Parameters
    ----------
    station: weather station ID
    yr: number/list/tuple, which year do we want to include for analysis
    daysahead: what's the daysahead for analysis
    fulldata: whether to include full data (with weather) or not (temperature only)
    subset: subset year and horizon
    preprocess: False=only output actual, True=output both actual and fcst
    actual_diff: produce differencing of actual data
    fcst_diff: produce differencing of fcst data
    time_adjust: whether to apply time zone adjustment

    Returns
    -------
    """
    if fulldata:
        actual = pd.read_csv(f'weather data/{station}_actualfull.csv', index_col=False)  # index is 1st col
        fcst = pd.read_csv(f'weather data/{station}_fcstfull.csv', index_col=False)
    else:
        actual = pd.read_csv(f'weather data/{station}_actual.csv', index_col=False)  # index is 1st col
        fcst = pd.read_csv(f'weather data/{station}_fcst.csv', index_col=False)
    actual = actual.loc[:, ~actual.columns.str.contains('^Unnamed')]  # drop unnamed cols
    fcst = fcst.loc[:, ~fcst.columns.str.contains('^Unnamed')]

    # Convert datetime
    fcst['ept'] = pd.to_datetime(fcst.ept, format='%Y-%m-%d %H:%M:%S.%f')
    fcst['historydate'] = pd.to_datetime(fcst.historydate, format='%Y-%m-%d %H:%M:%S.%f')
    actual['ept'] = pd.to_datetime(actual.ept, format='%Y-%m-%d %H:%M:%S.%f')

    if preprocess:
        if time_adjust:  # DST: -5hrs; else -6hrs
            fcst['date'], fcst['year'] = fcst['ept'].dt.date, fcst['ept'].dt.year
            fcst = fcst[(fcst.year.isin(yr)) & (fcst.daysahead.isin(daysahead))]
            fcst['ept'] = np.where(((fcst['date'] >= date(2019, 3, 10)) & (fcst['date'] <= date(2019, 11, 3))) |
                                   ((fcst['date'] >= date(2018, 3, 11)) & (fcst['date'] <= date(2018, 11, 4))),  # DST
                                   fcst['ept'] - timedelta(hours=5), fcst['ept'] - timedelta(hours=6))
            actual['ept'] = actual['ept'] - timedelta(hours=6)  # all actual -6hrs

        # refresh the dates
        fcst['date'], fcst['rls_hr'], fcst['year'], fcst['month'], fcst['hour'] = \
            fcst['ept'].dt.date, fcst['historydate'].dt.hour, fcst['ept'].dt.year, fcst['ept'].dt.month, \
            fcst['ept'].dt.hour

        fcst['season'] = (fcst.month % 12 + 3) // 3  # convert month to season

        fcst = fcst[(fcst.rls_hr == 7) & fcst.daysahead.isin(daysahead)]  # forecasts released 7am and chosen daysahead

        # if subset:  # subset again and remove 2018
        #     fcst = fcst[(fcst.year >= yr - 1) & (fcst.daysahead.isin(daysahead))]

        # sort before remove dup/differencing. sort priority daysahead>ept>historydate. Sort historydate while keep the sorting of ept.
        fcst.sort_values(by=['daysahead', 'ept', 'historydate'], ascending=True, inplace=True)
        fcst.drop_duplicates(subset=['daysahead', 'ept', 'rls_hr'], keep='last', inplace=True)  # keep last ele in dups

        if fcst_diff:
            same_day, one_day = fcst[fcst.daysahead == 0].copy(), fcst[fcst.daysahead == 1].copy()  # get copy
            # same_day differencing within each date
            same_day.sort_values(by=['date', 'ept'], ascending=True, inplace=True)
            for i in range(9):
                # same_day forecast temp differencing
                colname = 'abs_f_dif' + str(i + 1)
                same_day[colname] = abs(same_day.groupby('date')['temp'].diff(periods=i + 1))
                colname = 'f_dif' + str(i + 1)
                same_day[colname] = same_day.groupby('date')['temp'].diff(periods=i + 1)

            # other days ahead forecast differencing, and actual differencing
            actual.sort_values(by='ept', ascending=True, inplace=True)
            one_day.sort_values(by='ept', ascending=True, inplace=True)
            for i in range(9):
                if actual_diff:
                    # actual temp differencing
                    colname = 'abs_a_dif' + str(i + 1)
                    actual[colname] = abs(actual.temp.diff(periods=i + 1))
                    colname = 'a_dif' + str(i + 1)
                    actual[colname] = actual.temp.diff(periods=i + 1)
                # forecast temp differencing
                colname = 'abs_f_dif' + str(i + 1)
                one_day[colname] = abs(one_day.temp.diff(periods=i + 1))
                colname = 'f_dif' + str(i + 1)
                one_day[colname] = one_day.temp.diff(periods=i + 1)

            fcst = pd.concat([same_day, one_day])  # combine same_day and one_day forecasts back to fcst

        # Merge
        df = fcst.merge(actual, how='left', on=['stationcode', 'ept'], suffixes=('_fcst', '_actual'))
        df['abs_err'], df['err'] = abs(df.temp_fcst - df.temp_actual), df.temp_fcst - df.temp_actual
        # DST indicator. DST time has longer Sunlight duration
        df['DST'] = np.where(((df['date'] >= date(2019, 3, 10)) & (df['date'] <= date(2019, 11, 3))) |  # or
                             ((df['date'] >= date(2018, 3, 11)) & (df['date'] <= date(2018, 11, 4))), 1, 0)
        # sunshine indicator. Sunshine is available during that hour:1. Not available: 0.
        if fulldata:
            df['sunshine_ind'] = np.where(df['sunshine_min_total_fcst'] == 0, 0, 1)
            df['water_equivalent_fcst'] = df.water_equivalent_fcst * 100  # convert rainfall to mm unit
            df['water_equivalent_actual'] = df.water_equivalent_actual * 100  # convert rainfall to mm unit

        if subset:  # subset again and leave only specific year after timezone adjustment
            df = df[(df.year.isin(yr)) & (df.daysahead.isin(daysahead))]

    return df if preprocess else actual


def mean_max_process(data):
    # take daily mean/max/sum for other weather statistics, this is for daily weather conditions analysis
    shcva = data[['daysahead', 'date', 'sunshine_pct_possible_fcst', 'rel_humidity_fcst', 'cloud_cover_fcst',
                  'visibility_fcst', 'abs_err']].copy()  # solar_humid_cloud_visibility_abserr, for avg
    temp1 = shcva.groupby(['daysahead', 'date'], as_index=False).mean().round(2)
    rainfall = data[['daysahead', 'date', 'water_equivalent_fcst']].copy()  # daily rainfall, for sum
    temp2 = rainfall.groupby(['daysahead', 'date'], as_index=False).sum().round(2)
    wind_precip = data[['daysahead', 'date', 'wind_speed_fcst', 'speed_gust_fcst',
                        'precip_probability_fcst']].copy()  # wind/gust speed, precipitation_prob, for max
    temp3 = wind_precip.groupby(['daysahead', 'date'], as_index=False).max().round(2)
    temp = temp1.merge(temp2, on=['daysahead', 'date'])
    return temp.merge(temp3, on=['daysahead', 'date'])


# MAE
def MAE(y_true, y_pred):
    try:
        y_true, y_pred = y_true.to_numpy().flatten(), y_pred.to_numpy().flatten()
    except AttributeError:
        y_true, y_pred = y_true.to_numpy().flatten(), y_pred.flatten()
    return np.mean(np.abs((y_true - y_pred)))


def linreg_classifier(X, y, thres, print_confusion=False):
    # model = sm.OLS(y_train, X_train).fit()  # use training period to fit model
    # model.summary()
    lr = LinearRegression()
    reg = lr.fit(X, y)
    mae = mean_absolute_error(y, reg.predict(X))
    print(f'MAE: {mae:.3}')  # gives MAE on training set
    # Calc confusion matrix
    try:
        df_confusion = pd.crosstab(np.where(y >= thres, 'bad', 'good').flatten(),
                                   np.where(reg.predict(X) >= thres, 'bad', 'good'),
                                   rownames=['Actual'], colnames=['Predicted'], margins=True)
    except Exception:
        df_confusion = pd.crosstab(np.where(y >= thres, 'bad', 'good').flatten(),
                                   np.where(reg.predict(X).flatten() >= thres, 'bad', 'good'),  # convert y_pred to 1-D
                                   rownames=['Actual'], colnames=['Predicted'], margins=True)
    if print_confusion:
        print(df_confusion)
    return df_confusion, mae


def class_metrics(cm, null_accuracy):
    try:  # if is dataframe
        TN = cm.loc['good', 'good']
        TP = cm.loc['bad', 'bad']
        FN = cm.loc['bad', 'good']  # actual=bad (postive), predicted=good
        FP = cm.loc['good', 'bad']  # actual=good (negative), predicted=bad
        # print(
        #     f'The correct classification rate is: {(TP + TN) / float(TP + TN + FP + FN):.3%}')  # in %, 3 digits
        acc = 1 - (FP + FN) / float(TP + TN + FP + FN)
        tpr = TP / float(TP + FN)
        tnr = TN / float(TN + FP)
        # print(f'False positive rate: {FP / float(TN + FP):.3%}')  # "False Positive=Good Rate"
        # print(f'False negative rate: {FN / float(TP + FN):.3%}')  # "False Negative=Bad Rate"
    except KeyError:  # if no bad is forecasted, will throw out "KeyError: 'bad'"
        acc = null_accuracy
        tpr = 0
        tnr = 1

    print(f'Accuracy: {acc:.3%}')
    print(f'TP/all Positive: {tpr:.3%}')  # "True Positive=Good Rate"
    print(f'TN/all Negative: {tnr:.3%}')  # "True Negative=Bad Rate"

    return acc, tpr, tnr