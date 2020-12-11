# 5 worst/median/best days

# %% import & process
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os

# set pandas printing: https://stackoverflow.com/questions/11707586/how-do-i-expand-the-output-display-to-see-more-columns-of-a-pandas-dataframe
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

# inserting the searching path of import to the search list
user = 'Administrator'
sys.path.insert(1, f'C:/Users/{user}/PycharmProjects/MyTest/NCEMC code')
os.chdir(f'C:/Users/{user}/PycharmProjects/MyTest/NCEMC code')
# noinspection PyUnresolvedReferences
from Methods import import_preprocess, mean_max_process

station_list = ['KECG', 'KASJ', 'KMWK', 'KHBI', 'KEWN', 'KPGV', 'KIPJ', 'KGSB', 'KCPC', 'KORF', 'KGSP', 'KHNZ', 'KNCA',
                'KRWI', 'KIGX', 'KILM', 'KGEV', 'KEYF', 'KHSE', 'KPOB', 'KHKY', 'KMRH', 'KCLT', 'KMRN', 'KGSO', 'KAVL',
                'KAFP', 'KRDU']
k_list = ['KAFP', 'KASJ', 'KAVL', 'KCLT', 'KCPC', 'KECG']
# k_list = ['KAFP']
k = 'KAFP'  # station id
y = 'abs_err'
x = 'abs_f_'
year = [2019]
day = 5  # inspect xx worst/best/middle day
plot_other_weather = 1  # plot other weather conditions (daily)
# --------------------------------#
save_plot = 0
show_plot = 1
# --------------------------------#
plot_type = 'Max'  # do Daily Avg Err vs. Daily 'Avg' Temp change or Daily 'Max' Temp change
x_ax = 30  # limit for x-axis, for Temp change plots, not for other weather conditions
# --------------------------------------------------------------------------------------------------------------------#
if os.path.exists('Extreme days statistics.xlsx'):  # if exist old xlsx, delete
    os.remove('Extreme days statistics.xlsx')
for k in k_list:
    df = import_preprocess(station=k, yr=year, fulldata=True, actual_diff=False, fcst_diff=True)

    # don't use groupby cols as index, only set date as index
    mae_daily = df.groupby(['daysahead', 'date'], as_index=False)['abs_err'].mean().set_index('date')
    worst = mae_daily.groupby(['daysahead'])['abs_err'].nlargest(day).reset_index()
    best = mae_daily.groupby(['daysahead'])['abs_err'].nsmallest(day).reset_index()
    # for middle, get smallest half first, then get largest 3 from the smallest half
    middle = mae_daily.groupby(['daysahead'])['abs_err'].nsmallest(len(mae_daily) // 4).reset_index().set_index('date')
    middle = middle.groupby(['daysahead'])['abs_err'].nlargest(day).reset_index()


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

    # the plot_type only effects plotting the Temp change
    if not plot_other_weather:  # preprocessing for plotting Temp change only
        if plot_type == 'Avg':  # take mean of all statistics, including Temp change
            stat_worst = worst.groupby(['daysahead', 'date'], as_index=False).mean().round(2)
            stat_middle = middle.groupby(['daysahead', 'date'], as_index=False).mean().round(2)
            stat_best = best.groupby(['daysahead', 'date'], as_index=False).mean().round(2)
        else:  # plot_type == 'Max', take mean of abs_err but others are max
            def mean_max_temp_change(data):
                abs_err = data.pop('abs_err')  # pop out one col for special treatment, then merge it back
                abs_err = data[['daysahead', 'date']].merge(abs_err, left_index=True, right_index=True)
                temp1 = data.groupby(['daysahead', 'date'], as_index=False).max().round(2)
                temp2 = abs_err.groupby(['daysahead', 'date'], as_index=False).mean().round(2)
                return temp1.merge(temp2, how='left', on=['daysahead', 'date'])


            stat_worst = mean_max_temp_change(worst)
            stat_middle = mean_max_temp_change(middle)
            stat_best = mean_max_temp_change(best)

    # preprocessing when plotting other weather statistics
    cols = ['stationcode', 'daysahead', 'date', 'month', 'hour', 'temp_fcst', 'temp_actual', 'err', 'abs_err',
            'rel_humidity_fcst', 'rel_humidity_actual', 'wind_speed_fcst', 'wind_speed_actual', 'speed_gust_fcst',
            'speed_gust_actual', 'cloud_cover_fcst', 'cloud_cover_actual', 'cloud_cover_type_fcst',
            'cloud_cover_type_actual', 'visibility_fcst', 'visibility_actual',
            'sunshine_min_total_actual', 'sunshine_pct_possible_fcst', 'sunshine_pct_possible_actual',
            'precip_probability_fcst', 'precip_probability_actual', 'water_equivalent_fcst', 'water_equivalent_actual']

    if plot_other_weather:
        stat_worst = mean_max_process(worst)
        stat_middle = mean_max_process(middle)
        stat_best = mean_max_process(best)
    # ----------------------------------------------------------------------------------------------------------------#
    for horizon in range(2):
        w = stat_worst[stat_worst.daysahead == horizon]  # worst
        m = stat_middle[stat_middle.daysahead == horizon]  # middle
        b = stat_best[stat_best.daysahead == horizon]  # best
        if x == 'abs_f_':
            x_label = f'Daily {plot_type} Absolute Forecast Temp Change (°F)'
        elif x == 'abs_a_':
            x_label = f'Daily {plot_type} Actual Temp Change (°F)'
        else:  # if x=='f_'
            x_label = f'Daily {plot_type} Forecast Temp Change (°F)'
        if y == 'err':
            y_label = f'Daily Avg Forecast Error (°F)'
        else:
            y_label = f'Daily Avg Absolute Forecast Error (°F)'
        if horizon == 0:
            title = f'Year={year}, Same-Day Forecast'
        elif horizon == 1:
            title = f'Year={year}, {horizon} Day Ahead Forecast'
        else:
            title = f'Year={year}, {horizon} Days Ahead Forecast'

        if not plot_other_weather:
            fig, ax = plt.subplots(nrows=3, ncols=3, sharex='col', sharey='row', figsize=(7, 7))
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel(x_label, fontsize=12)
            plt.ylabel(y_label, fontsize=12)
            fig.suptitle(title)

            for i in range(3):  # col
                for j in range(3):  # row
                    colname = 'dif' + str(i + j * 3 + 1)

                    ax[j, i].plot(w[x + colname], w[y], c='red', alpha=.7, marker="o", linewidth=.3, markersize=4)
                    ax[j, i].plot(m[x + colname], m[y], c='orange', alpha=.7, marker="o", linewidth=.3, markersize=4)
                    ax[j, i].plot(b[x + colname], b[y], c='green', alpha=.7, marker="o", linewidth=.3, markersize=4)

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
            fig, ax = plt.subplots(nrows=4, ncols=2, sharey='row', figsize=(5.5, 7.5))
            fig.add_subplot(111, frameon=False)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel(y_label, fontsize=12)
            fig.suptitle(title)

            j = 0
            for i in ['sunshine_pct_possible_fcst', 'rel_humidity_fcst', 'cloud_cover_fcst', 'visibility_fcst',
                      'water_equivalent_fcst', 'precip_probability_fcst', 'wind_speed_fcst', 'speed_gust_fcst']:
                x_num, y_num = j // 2, j % 2
                sub_title = ['Daily Avg Sunshine %', 'Daily Avg Humidity %', 'Daily Avg Cloud Cover %',
                             'Daily Avg Visibility', 'Daily Sum Rainfall', 'Daily Max Precip Prob %',
                             'Daily Max Wind Speed', 'Daily Max Gust Speed']
                ax[x_num, y_num].plot(w[i], w[y], c='red', alpha=.5, marker="o", linewidth=.2, markersize=4)
                ax[x_num, y_num].plot(m[i], m[y], c='orange', alpha=.5, marker="o", linewidth=.2, markersize=4)
                ax[x_num, y_num].plot(b[i], b[y], c='green', alpha=.5, marker="o", linewidth=.2, markersize=4)
                ax[x_num, y_num].set_title(sub_title[j], fontsize=10)
                ax[x_num, y_num].set(ylim=(-1, 12))
                ax[x_num, y_num].yaxis.set_ticks(np.arange(0, 12, 5))
                if k == 'KCPC':
                    ax[x_num, y_num].set(ylim=(-1, 16))
                    ax[x_num, y_num].yaxis.set_ticks(np.arange(0, 16, 5))
                j += 1

            if save_plot:
                plt.savefig(f'graphs/station={k}, worst&best weather conditions {title}.png', bbox_inches='tight')
if show_plot:
    plt.show()
