# Daily Temp change impact (Tmax - Tmin, daily average Temp change) to errors

# %% import & process
# import matplotlib
#
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
import seaborn as sns
from numpy.polynomial.polynomial import polyfit
import matplotlib.colors as mcolors

# set pandas printing: https://stackoverflow.com/questions/11707586/how-do-i-expand-the-output-display-to-see-more-columns-of-a-pandas-dataframe
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

# noinspection PyUnresolvedReferences
from Methods import import_preprocess

# ----------------------------------------------------------------------------------------------------------------#
station_list = ['KECG', 'KASJ', 'KMWK', 'KHBI', 'KEWN', 'KPGV', 'KIPJ', 'KGSB', 'KCPC', 'KORF', 'KGSP', 'KHNZ', 'KNCA',
                'KRWI', 'KIGX', 'KILM', 'KGEV', 'KEYF', 'KHSE', 'KPOB', 'KHKY', 'KMRH', 'KCLT', 'KMRN', 'KGSO', 'KAVL',
                'KAFP', 'KRDU']
# k_list = ['KAFP', 'KASJ', 'KAVL', 'KCLT', 'KCPC', 'KECG']
k_list = ['KASJ']
y = 'abs_err'
year = [2019]
reg_line = 1
daily_Temp_fcst_range = 0  # Abs forecast error vs. Tmax - Tmin
daily_avg_Temp_change = 1  # Abs forecast error vs. Daily Avg Temp Change
# --------------------------------#
save_plot = 0
show_plot = 1
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
# ----------------------------------------------------------------------------------------------------------------#

# %% Plot
for k in k_list:
    # preprocess before plot
    df = import_preprocess(station=k, yr=year, actual_diff=False, fcst_diff=True)
    # np.ptp returns the range (max-min) of an array
    df['daily_f_range'] = df.groupby(['daysahead', 'date'])['temp_fcst'].transform(np.ptp)
    for i in range(1, 10, 1):
        df[f'abs_f_dif{i}_avg'] = df.groupby(['daysahead', 'date'])[f'abs_f_dif{i}'].transform(np.mean)  # avg T change
    df['daily_abs_err'] = df.groupby(['daysahead', 'date'])['abs_err'].transform(np.mean)  # avg daily abs err
    # check NaN
    # np.sum(df.abs_err.isnull())
    # df[df.abs_err.isnull()]

    for horizon in range(1, 2):
        dh = df[df.daysahead == horizon]
        if horizon == 0:
            title = f'Year={year}, Same-Day Forecast'
        elif horizon == 1:
            title = f'Year={year}, {horizon} Day Ahead Forecast'
        else:
            title = f'Year={year}, {horizon} Days Ahead Forecast'

        if daily_Temp_fcst_range:  # Abs forecast error vs. Tmax - Tmin
            fig, ax = plt.subplots(nrows=1, ncols=1, sharex='col', figsize=(4, 3.8), dpi=200)  # nrows=2, figsize=(3.5, 6)
            fig.add_subplot(111, frameon=False)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

            # fig.suptitle(title, fontsize=10)

            freq = dh.groupby(['daily_f_range', y]).size().reset_index(name="occurrance")  # coordinate freq
            freq['occurrance'] = freq.occurrance.div(freq.occurrance.max() - freq.occurrance.min())  # normalize
            ax.scatter(freq['daily_f_range'], freq[y], c=freq.occurrance, s=20, alpha=.5, marker="o",
                       edgecolor='none', cmap='jet')  # ax[0]
            ax.set_ylabel('Absolute Forecast Error (°F)', fontsize=10)
            ax.set(xlim=(-2, 41), ylim=(-2, 21))
            ax.xaxis.set_ticks(np.arange(0, 41, 10))

            temp_df = dh.drop_duplicates(subset=['date', 'daily_f_range', f'daily_{y}'], keep='first').copy()

            # ax[1].scatter(temp_df['daily_f_range'], temp_df[f'daily_{y}'], s=18, alpha=.3, marker="o", c=cols[0],
            #               edgecolor='none')
            # ax[1].set_ylabel('Daily Avg Absolute Error (°F)', fontsize=10)
            # ax[1].set(xlim=(-2, 41), ylim=(-2, 11))
            # ax[1].yaxis.set_ticks(np.arange(0, 11, 5))

            plt.xlabel('Diurnal $T_{F}$ Variation (°F)', fontsize=10)

            if reg_line:
                # for ax[0], hourly error vs. forecast T range
                temp0 = pd.concat([dh['daily_f_range'], dh[y]], axis=1)  # 只有用于画regression的两个col
                temp0.dropna(inplace=True)  # 手动去除na before plotting regression
                x_ax = np.arange(-1, 41, 1)
                b, m = polyfit(temp0.iloc[:, 0], temp0.iloc[:, 1], 1)
                ax.plot(x_ax, b + m * x_ax, '-', c='black', linewidth=2)
                plt.text(.85, .9, f'm={round(m, 3)}', horizontalalignment='center', verticalalignment='center',
                         transform=ax.transAxes, fontsize=9, fontweight='bold')

                # for ax[1], daily avg error vs. forecast T range
                # temp1 = pd.concat([temp_df['daily_f_range'], temp_df[f'daily_{y}']], axis=1)  # 只有画regression的两col
                # temp1.dropna(inplace=True)  # 手动去除na before plotting regression
                # b, m = polyfit(temp1.iloc[:, 0], temp1.iloc[:, 1], 1)
                # ax[1].plot(x_ax, b + m * x_ax, '-', c='black', linewidth=2)
                # plt.text(.85, .9, f'm={round(m, 3)}', horizontalalignment='center', verticalalignment='center',
                #          transform=ax[1].transAxes, fontsize=7, fontweight='bold')

            if save_plot:
                plt.savefig(f'graphs/station={k}, Hourly&Daily Abs Err vs. Fcst Range {title}.png', bbox_inches='tight')
            if not show_plot:
                plt.close('all')  # close all after save to save memory

        if daily_avg_Temp_change:  # Abs forecast error vs. Daily Avg Temp Change

            for y_ax in [
                'abs_err']:  # ['abs_err', 'daily_abs_err'], daily_abs_err has the same slope, abs_err is hourly
                fig, ax = plt.subplots(nrows=3, ncols=3, sharex='col', sharey='row', figsize=(7, 7.2), dpi=200)
                fig.add_subplot(111, frameon=False)
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

                # fig.suptitle(title)

                plt.xlabel('Diurnal Average Absolute $T_{F}$ Variation (°F)', fontsize=12)
                if y_ax == 'abs_err':  # abs_err is hourly
                    plt.ylabel('Absolute Forecast Error (°F)', fontsize=12)
                else:
                    plt.ylabel('Daily Avg Absolute Error (°F)', fontsize=12)

                for i in range(3):  # col
                    for j in range(3):  # row
                        colname = f'abs_f_dif{i + j * 3 + 1}_avg'
                        freq = dh.groupby([colname, y]).size().reset_index(name="occurrance")  # coordinate freq
                        freq['occurrance'] = freq.occurrance.div(
                            freq.occurrance.max() - freq.occurrance.min())  # normalize
                        sc = ax[j, i].scatter(freq[colname], freq[y], c=freq.occurrance, s=10, alpha=.3,
                                              marker="o", edgecolor='none', cmap='jet')
                        # sc = ax[j, i].scatter(dh[colname], dh[y_ax], s=12, alpha=.1, marker="H", edgecolor='none',
                        #                       c=cols[0])
                        ax[j, i].set(xlim=(-2, 23), ylim=(-2, 21))
                        ax[j, i].xaxis.set_ticks(np.arange(0, 23, 10))
                        ax[j, i].yaxis.set_ticks(np.arange(0, 21, 5))
                        if y_ax == 'daily_abs_err':
                            ax[j, i].set(xlim=(-2, 23), ylim=(-2, 16))
                            ax[j, i].yaxis.set_ticks(np.arange(0, 16, 5))

                        temp_df = pd.concat([dh[colname], dh[y_ax]], axis=1)  # 只有用于画regression的两个col
                        temp_df.dropna(inplace=True)  # 手动去除na before plotting regression
                        # temp_pos = temp_df[temp_df.iloc[:, 0] > 0]  # x>=0, 画一四象限的regression line
                        # temp_neg = temp_df[temp_df.iloc[:, 0] < 0]  # x<=0, 画二三象限的regression line

                        if reg_line:
                            x_ax = np.arange(-1, 41, 1)
                            b, m = polyfit(temp_df.iloc[:, 0], temp_df.iloc[:, 1], 1)
                            ax[j, i].plot(x_ax, b + m * x_ax, '-', c='black', linewidth=2)
                            plt.text(.78, .9, f'm={round(m, 3)}', horizontalalignment='center',
                                     verticalalignment='center', transform=ax[j, i].transAxes, fontsize=8.5,
                                     fontweight='bold')

                        if (i == 0) & (j == 0):
                            ax[j, i].set_title(f'Within {str(i + j * 3 + 1)} Hour', fontsize=10)
                        else:
                            ax[j, i].set_title(f'Within {str(i + j * 3 + 1)} Hours', fontsize=10)
                        ax[j, i].title.set_position([.5, 1])  # adjust distance between title and plot

                if save_plot:
                    plt.savefig(f'graphs/station={k}, {y_ax} vs. Avg Temp change, {title}.png', bbox_inches='tight')
            if not show_plot:
                plt.close('all')  # close all after save to save memory
if show_plot:
    plt.show()
