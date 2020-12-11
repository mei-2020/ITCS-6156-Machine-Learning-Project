# xx worst/median/best hours, Not Done yet

# %% import & process
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os
import seaborn as sns
from numpy.polynomial.polynomial import polyfit

# set pandas printing: https://stackoverflow.com/questions/11707586/how-do-i-expand-the-output-display-to-see-more-columns-of-a-pandas-dataframe
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

# noinspection PyUnresolvedReferences
from Methods import import_preprocess

station_list = ['KECG', 'KASJ', 'KMWK', 'KHBI', 'KEWN', 'KPGV', 'KIPJ', 'KGSB', 'KCPC', 'KORF', 'KGSP', 'KHNZ', 'KNCA',
                'KRWI', 'KIGX', 'KILM', 'KGEV', 'KEYF', 'KHSE', 'KPOB', 'KHKY', 'KMRH', 'KCLT', 'KMRN', 'KGSO', 'KAVL',
                'KAFP', 'KRDU']
# k_list = ['KAFP', 'KASJ', 'KAVL', 'KCLT', 'KCPC', 'KECG']
k_list = ['KAFP']
# k = 'KAFP'  # station id
y = 'abs_err'
x = 'abs_f_'
x_ax = 30  # limit for x-axis
year = [2019]
hour = 5  # inspect xx worst/best/middle hours
plot_other_weather = 0  # plot other weather statistics
# --------------------------------#
save_plot = 0
show_plot = 1
# --------------------------------------------------------------------------------------------------------------------#
if os.path.exists('Extreme hours statistics.xlsx'):  # if exist old xlsx, delete
    os.remove('Extreme hours statistics.xlsx')
for k in k_list:
    df = import_preprocess(station=k, yr=year, fulldata=True, actual_diff=True, fcst_diff=True)
    df.sort_values(by=['daysahead', 'abs_err'], ascending=True, inplace=True)  # sort ascending
    df.dropna(subset=['abs_err'], inplace=True)
    # don't use groupby cols as index, only set date as index
    worst = df.groupby(['daysahead']).tail(hour).reset_index(drop=True)
    best = df.groupby(['daysahead']).head(hour).reset_index(drop=True)
    # for middle, get smallest half first, then get largest 3 from the smallest half
    middle = df.groupby(['daysahead']).head(len(df) // 4).reset_index(drop=True)
    middle = middle.groupby(['daysahead']).tail(hour).reset_index(drop=True)

# ----------------------------------------Below are not done yet-------------------------------------------#

    cols = ['stationcode', 'daysahead', 'date', 'month', 'hour', 'temp_fcst', 'temp_actual', 'err', 'abs_err',
            'rel_humidity_fcst', 'rel_humidity_actual', 'wind_speed_fcst', 'wind_speed_actual', 'speed_gust_fcst',
            'speed_gust_actual', 'cloud_cover_fcst', 'cloud_cover_actual', 'visibility_fcst', 'visibility_actual',
            'sunshine_min_total_actual', 'sunshine_pct_possible_fcst', 'sunshine_pct_possible_actual',
            'precip_probability_fcst', 'precip_probability_actual', 'water_equivalent_fcst', 'water_equivalent_actual']


    def process_extreme_days(data, export=True):  # this process is to filter only the worst/best days
        name = [x for x in globals() if globals()[x] is data][0]  # take name first, before data manipulation
        data = data.drop(columns=['abs_err']).merge(df, how='left', on=['daysahead', 'date'])  # drop abs_err then merge
        stat = data.groupby(['daysahead', 'date'], as_index=False).mean()[['daysahead', 'date', 'abs_err']].round(2)
        if export:  # the stat is for print out.
            try:
                with pd.ExcelWriter('Extreme days statistics.xlsx', mode='a') as writer:  # if no error, append
                    stat.to_excel(writer, sheet_name=f'{k}_{name}')
            except FileNotFoundError:
                with pd.ExcelWriter('Extreme days statistics.xlsx', mode='w') as writer:  # if error, write first
                    stat.to_excel(writer, sheet_name=f'{k}_{name}')
        return data


    worst = process_extreme_days(worst)
    middle = process_extreme_days(middle)
    best = process_extreme_days(best)

    if plot_type == 'Avg':  # take mean of all statistics
        stat_worst = worst.groupby(['daysahead', 'date'], as_index=False).mean().round(2)
        stat_middle = middle.groupby(['daysahead', 'date'], as_index=False).mean().round(2)
        stat_best = best.groupby(['daysahead', 'date'], as_index=False).mean().round(2)
    else:  # plot_type == 'Max', take mean of abs_err but others are max
        def mean_max_process(data):
            abs_err = data.pop('abs_err')
            abs_err = data[['daysahead', 'date']].merge(abs_err, left_index=True, right_index=True)
            temp1 = data.groupby(['daysahead', 'date'], as_index=False).max().round(2)
            temp2 = abs_err.groupby(['daysahead', 'date'], as_index=False).mean().round(2)
            return temp1.merge(temp2, how='left', on=['daysahead', 'date'])


        stat_worst = mean_max_process(worst)
        stat_middle = mean_max_process(middle)
        stat_best = mean_max_process(best)

    # stat_cols = stat_worst.columns.tolist()

    for horizon in range(2):
        w = stat_worst[stat_worst.daysahead == horizon]  # worst
        m = stat_middle[stat_middle.daysahead == horizon]  # middle
        b = stat_best[stat_best.daysahead == horizon]  # best

        if not plot_other_weather:
            fig, ax = plt.subplots(nrows=3, ncols=3, sharex='col', sharey='row', figsize=(7, 7))
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

            if x == 'abs_f_':
                plt.xlabel(f'Daily {plot_type} Absolute Forecast Temp Change (°F)', fontsize=12)
            elif x == 'abs_a_':
                plt.xlabel(f'Daily {plot_type} Actual Temp Change (°F)', fontsize=12)
            elif x == 'f_':
                plt.xlabel(f'Daily {plot_type} Forecast Temp Change (°F)', fontsize=12)

            if y == 'err':
                plt.ylabel(f'Daily Avg Forecast Error (°F)', fontsize=12)
            else:
                plt.ylabel(f'Daily Avg Absolute Forecast Error (°F)', fontsize=12)

            if horizon == 0:
                title = f'Year={year}, Same-Day Forecast'
            elif horizon == 1:
                title = f'Year={year}, {horizon} Day Ahead Forecast'
            else:
                title = f'Year={year}, {horizon} Days Ahead Forecast'

            fig.suptitle(title)

            for i in range(3):  # col
                for j in range(3):  # row
                    colname = 'dif' + str(i + j * 3 + 1)

                    ax[j, i].plot(w[x + colname], w[y], c='red', alpha=.7, marker="o", linewidth=.7, markersize=4)
                    ax[j, i].plot(m[x + colname], m[y], c='orange', alpha=.7, marker="o", linewidth=.7, markersize=4)
                    ax[j, i].plot(b[x + colname], b[y], c='green', alpha=.7, marker="o", linewidth=.7, markersize=4)

                    if (y[:3] == 'abs') & (x[:3] == 'abs'):
                        ax[j, i].set(xlim=(-2, x_ax + 1), ylim=(-2, 21))
                        ax[j, i].xaxis.set_ticks(np.arange(0, x_ax + 1, x_ax // 2))
                        ax[j, i].yaxis.set_ticks(np.arange(0, 21, 5))
                    elif (y[:3] != 'abs') & (x[:3] == 'abs'):
                        ax[j, i].set(xlim=(-2, x_ax + 1), ylim=(-21, 21))
                        ax[j, i].xaxis.set_ticks(np.arange(0, x_ax + 1, x_ax // 2))
                        ax[j, i].yaxis.set_ticks(np.arange(-20, 21, 10))
                        ax[j, i].axhline(y=0, color='gray', linestyle=':', linewidth=1)
                    elif (y[:3] == 'abs') & (x[:3] != 'abs'):
                        ax[j, i].set(xlim=(-x_ax - 1, x_ax + 1), ylim=(-2, 21))
                        ax[j, i].xaxis.set_ticks(np.arange(-x_ax, x_ax + 1, x_ax))
                        ax[j, i].yaxis.set_ticks(np.arange(0, 21, 5))
                        ax[j, i].axvline(x=0, color='gray', linestyle=':', linewidth=1)
                    else:  # both y and x are non-abs values
                        ax[j, i].set(xlim=(-x_ax - 1, x_ax + 1), ylim=(-21, 21))
                        ax[j, i].xaxis.set_ticks(np.arange(-x_ax, x_ax + 1, x_ax))
                        ax[j, i].yaxis.set_ticks(np.arange(-20, 21, 10))
                        ax[j, i].axvline(x=0, color='gray', linestyle=':', linewidth=1)
                        ax[j, i].axhline(y=0, color='gray', linestyle=':', linewidth=1)

                    if (i == 0) & (j == 0):
                        ax[j, i].set_title(str(i + j * 3 + 1) + ' Hour Temp Change', fontsize=10)
                    else:
                        ax[j, i].set_title(str(i + j * 3 + 1) + ' Hours Temp Change', fontsize=10)
                    ax[j, i].title.set_position([.5, 1])  # adjust distance between title and plot

            if save_plot:
                plt.savefig(f'graphs/station={k}, worst&best daily {plot_type} {title}.png', bbox_inches='tight')

        if plot_other_weather:
            fig, ax = plt.subplots(nrows=3, ncols=3, sharex='col', sharey='row', figsize=(7, 7))
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            for i in range(3):  # col
                for j in range(3):  # row
                    colname = 'dif' + str(i + j * 3 + 1)

                    ax[j, i].plot(w[x + colname], w[y], c='red', alpha=.7, marker="o", linewidth=.7, markersize=4)
                    ax[j, i].plot(m[x + colname], m[y], c='orange', alpha=.7, marker="o", linewidth=.7, markersize=4)
                    ax[j, i].plot(b[x + colname], b[y], c='green', alpha=.7, marker="o", linewidth=.7, markersize=4)

if show_plot:
    plt.show()
