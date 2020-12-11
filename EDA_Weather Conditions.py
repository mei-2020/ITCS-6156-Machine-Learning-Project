# %% import & process
import matplotlib
#
# matplotlib.use('TkAgg')

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
from Methods import import_preprocess, mean_max_process

# station_list = ['KECG', 'KASJ', 'KMWK', 'KHBI', 'KEWN', 'KPGV', 'KIPJ', 'KGSB', 'KCPC', 'KORF', 'KGSP', 'KHNZ', 'KNCA',
#                 'KRWI', 'KIGX', 'KILM', 'KGEV', 'KEYF', 'KHSE', 'KPOB', 'KHKY', 'KMRH', 'KCLT', 'KMRN', 'KGSO', 'KAVL',
#                 'KAFP', 'KRDU']
# k_list = ['KAFP', 'KASJ', 'KAVL', 'KCLT', 'KCPC', 'KECG']
k_list = ['KCPC']  # 'KCPC' gives the best right "signs" of variables
y = 'abs_err'
# x = 'f_'
year = [2019]
reg_line = True
plot_daily = 0  # if plot hourly then 0; If want to plot daily data (w/ min/max/avg process), then 1.
# --------------------------------#
save_plot = 0
show_plot = 1
# --------------------------------------------------------------------------------------------------------------------#
for k in k_list:
    df = import_preprocess(station=k, yr=year, fulldata=True, actual_diff=False, fcst_diff=True)
    df.dropna(subset=['err', 'abs_err'], inplace=True)
    # df[['temp_fcst', 'rel_humidity_fcst', 'cloud_cover_fcst', 'visibility_fcst','sunshine_pct_possible_fcst', 'precip_probability_fcst', 'water_equivalent_fcst']].describe()

    if plot_daily:  # if we are plotting daily forecast
        df = mean_max_process(data=df)
    for horizon in range(1, 2):  # range(2)
        dh = df[df.daysahead == horizon]

        if y == 'err':
            if plot_daily:
                y_label = f'Daily Avg Forecast Error (°F)'
            else:
                y_label = f'Daily Avg Absolute Forecast Error (°F)'
        else:  # y==abs_err
            if plot_daily:
                y_label = f'Daily Avg Absolute Forecast Error (°F)'
            else:
                y_label = f'Absolute Forecast Error (°F)'
        if horizon == 0:
            title = f'Year={year}, Same-Day Forecast'
        elif horizon == 1:
            title = f'Year={year}, {horizon} Day Ahead Forecast'
        else:
            title = f'Year={year}, {horizon} Days Ahead Forecast'

        # fig, ax = plt.subplots(nrows=4, ncols=2, sharey='row', figsize=(5.5, 7.5)) # plot wind speed and gust also
        fig, ax = plt.subplots(nrows=2, ncols=3, sharey='row', figsize=(9, 6), dpi=200)
        fig.add_subplot(111, frameon=False)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.ylabel(y_label, fontsize=14)
        # fig.suptitle(title)

        j = 0
        for i in ['sunshine_pct_possible_fcst', 'rel_humidity_fcst', 'cloud_cover_fcst', 'visibility_fcst',
                  'water_equivalent_fcst', 'precip_probability_fcst',
                  # 'wind_speed_fcst', 'speed_gust_fcst'
                  ]:
            x_num, y_num = j // 3, j % 3
            if plot_daily:
                sub_title = ['Daily Avg Sunshine %', 'Daily Avg Humidity %', 'Daily Avg Cloud Cover %',
                             'Daily Avg Visibility', 'Daily Sum Rainfall', 'Daily Max Precip Prob %',
                             'Daily Max Wind Speed', 'Daily Max Gust Speed']
                sc = ax[x_num, y_num].scatter(dh[i], dh[y], s=8, alpha=.3, marker="H", edgecolor='none')
            else:  # hourly
                sub_title = ['Sunshine (%)', 'Relative Humidity (%)', 'Cloud Cover (%)',
                             'Visibility (mile)', 'Precipitation (mm)', 'Precipitation Probability (%)',
                             # 'Hourly Wind Speed', 'Hourly Gust Speed'
                             ]
                if i == 'sunshine_pct_possible_fcst':  # subset for only hours when sunshine is available (sunshine_ind=1)
                    freq = dh[dh.sunshine_ind == 1].groupby([i, y]).size().reset_index(name="occurrance")
                else:
                    freq = dh.groupby([i, y]).size().reset_index(name="occurrance")  # coordinate freq
                # freq = dh.groupby([i, y]).size().reset_index(name="occurrance")  # coordinate freq
                freq['occurrance'] = freq.occurrance.div(freq.occurrance.max() - freq.occurrance.min())  # normalize
                sc = ax[x_num, y_num].scatter(freq[i], freq[y], c=freq.occurrance, s=12, alpha=.4,
                                              marker="o", edgecolor='none', cmap='jet')

            ax[x_num, y_num].set_title(sub_title[j], fontsize=11)  # set titles for subplots
            # set y axis ticks
            if plot_daily:
                ax[x_num, y_num].set(ylim=(-2, 15))
                ax[x_num, y_num].yaxis.set_ticks(np.arange(0, 15, 5))
            else:
                ax[x_num, y_num].set(ylim=(-2, 21))
                ax[x_num, y_num].yaxis.set_ticks(np.arange(0, 21, 5))

            if y[:3] != 'abs':
                ax[x_num, y_num].set(ylim=(-21, 21))
                ax[x_num, y_num].yaxis.set_ticks(np.arange(-20, 21, 10))
                ax[x_num, y_num].axhline(y=0, color='gray', linestyle=':', linewidth=2)

            if reg_line:
                if i == 'sunshine_pct_possible_fcst':  # subset for only hours when sunshine is available (sunshine_ind=1)
                    temp_df = pd.concat([dh[dh.sunshine_ind == 1][i], dh[dh.sunshine_ind == 1][y]], axis=1)
                else:
                    temp_df = pd.concat([dh[i], dh[y]], axis=1)  # 只有用于画regression的两个col

                temp_df.dropna(inplace=True)  # 手动去除na before plotting regression
                x_ax = np.arange(-5, 100, 1)
                b, m = polyfit(temp_df.iloc[:, 0], temp_df.iloc[:, 1], 1)
                ax[x_num, y_num].plot(x_ax, b + m * x_ax, '-', c='black', linewidth=3)
                ax[x_num, y_num].set(xlim=(-2, max(temp_df.iloc[:, 0])))
                if j == 3:  # the 4th graph
                    ax[x_num, y_num].set(xlim=(-1, max(temp_df.iloc[:, 0]) + 1))
                if j == 4:  # the 4th graph
                    ax[x_num, y_num].set(xlim=(-0.05, max(temp_df.iloc[:, 0] + 0.1)))
                plt.text(.8, .9, f'm={round(m, 3)}', horizontalalignment='center', verticalalignment='center',
                         transform=ax[x_num, y_num].transAxes, fontsize=11, fontweight='bold')

            j += 1

        # ax[1][0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))  # set visibility to integer
        ax[1][0].xaxis.set_ticks(np.arange(0, 10.1, 2))  # set the ticks for visibility
        ax[1][2].xaxis.set_ticks(np.arange(0, 85, 20))  # set the ticks for Precip. Prob

        if save_plot:
            if plot_daily:
                plt.savefig(f'graphs/station={k}, daily weather conditions {title}.png', bbox_inches='tight')
            else:
                plt.savefig(f'graphs/station={k}, hourly weather conditions {title}.png', bbox_inches='tight')
if show_plot:
    plt.show()
