"""
EDA of error by time dependent factors, such as month and hour
"""

# %% import & process
# import matplotlib
#
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os
from datetime import date
import seaborn as sns

# set pandas printing: https://stackoverflow.com/questions/11707586/how-do-i-expand-the-output-display-to-see-more-columns-of-a-pandas-dataframe
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
plt.rcParams.update({'figure.max_open_warning': 0})  # get rid of figure.max_open_warning

# noinspection PyUnresolvedReferences
from Methods import import_preprocess, mean_max_process

station_list = ['KECG', 'KASJ', 'KMWK', 'KHBI', 'KEWN', 'KPGV', 'KIPJ', 'KGSB', 'KCPC', 'KORF', 'KGSP', 'KHNZ', 'KNCA',
                'KRWI', 'KIGX', 'KILM', 'KGEV', 'KEYF', 'KHSE', 'KPOB', 'KHKY', 'KMRH', 'KCLT', 'KMRN', 'KGSO', 'KAVL',
                'KAFP', 'KRDU']
# k_list = ['KAFP', 'KASJ', 'KAVL', 'KCLT', 'KCPC', 'KECG']
k_list = ['KASJ']
year = [2019]
y = 'abs_err'
X_title = 'Hour'
X = X_title.lower()
by_X_boxplot = 1  # 1=plot by X_title plot
avg_temp_change_by_X = 0  # avg temperature change plot by X_title

DST = [True]  # [False, True] is all days, [True] is only DST days, [False] is only non-DST days
seasons = ['Winter', 'non-Winter']
# by_season = False  # True: plot by season error plots, this is to detect if winter is different than others. False: plot by DST error plot

if DST[0]:  # if DST, then highlight; otherwise, don't highlight
    highlight_X_0 = [7, 8, 9, 18]  # same-day forecast, highlight, within [4,18]
    highlight_X_1 = [6, 7, 8, 9, 17, 18, 19]  # 1-day-ahead forecast, highlight the hour/month on the plot
else:  # if DST[0]==False
    highlight_X_0 = [7, 8, 9, 17]  # same-day forecast, highlight, within [5,17]
    highlight_X_1 = [6, 7, 8, 9, 17, 18, 19]
# ------------------#
show_plot = 1
save_plot = 0
# --------------------------------------------------------------------------------------------------------------------#
for k in k_list:
    df = import_preprocess(station=k, yr=year, fulldata=True, actual_diff=False, fcst_diff=True)

    # plot for a day
    # dh = df[(df.daysahead == 1) & (df.month == 1) & (df.date == date(2019, 1, 2))]
    # plt.plot(dh.ept, dh[y])

    # for season_no, season in enumerate(seasons, 1):  # start season index is 1

    # By X boxplot
    if by_X_boxplot:
        sns.set_style("ticks")
        for horizon in range(1, 2):
            dh = df[(df.daysahead == horizon) & (df.DST.isin(DST))]
            # if season_no == 1:
            #     dh = df[(df.daysahead == horizon) & (df.season == season_no)]
            # else:
            #     dh = df[(df.daysahead == horizon) & (df.season != 1)]

            plt.figure(figsize=(8, 4), dpi=200)
            sns.boxplot(x=X, y=y, data=dh)
            plt.xlabel(X_title, fontsize=12)

            if horizon == 0:
                title = f'Year={year}, Same-Day Forecast Error (°F) by {X_title}, ' + r"$\bf{DST=" + str(DST) + "}$"
            elif horizon == 1:
                title = f'Year={year}, 1 Day Ahead Forecast Error (°F) by {X_title}, ' + \
                        r"$\bf{DST=" + str(DST) + "}$"
                # title = f'Year={year}, 1 Day Ahead Forecast Error (°F) by {X_title}, ' + \
                #         r"$\bf{Season=" + str(season) + "}$"
            else:
                title = f'Year={year}, {horizon} Days Ahead Forecast Error (°F) by {X_title}, ' + \
                        r"$\bf{DST=" + str(DST) + "}$"

            # plt.title(title)

            if y == 'err':
                plt.ylabel(f'Forecast Error (°F)', fontsize=12)
                plt.axhline(y=0, color='gray', linestyle=':', linewidth=1)
                plt.axhline(y=5, color='gray', linestyle=':', linewidth=1)
                plt.axhline(y=10, color='gray', linestyle=':', linewidth=1)
                plt.axhline(y=-5, color='gray', linestyle=':', linewidth=1)
                plt.axhline(y=-10, color='gray', linestyle=':', linewidth=1)
                ax = plt.gca()  # get current axis
                ax.set(ylim=(-21, 21))
                ax.yaxis.set_ticks(np.arange(-20, 21, 5))
            else:
                plt.ylabel('Absolute Forecast Error (°F)', fontsize=12)
                plt.axhline(y=5, color='gray', linestyle=':', linewidth=1)
                plt.axhline(y=10, color='gray', linestyle=':', linewidth=1)
                ax = plt.gca()  # get current axis
                ax.set(ylim=(0, 17))  # ylim=(0, 21)
                ax.yaxis.set_ticks(np.arange(0, 17, 5))  # np.arange(0, 21, 5)

            if save_plot:
                plt.savefig(f'graphs/station={k}, {y}, {title[:-17]} DST={DST}.png', bbox_inches='tight')
                # plt.savefig(f'graphs/station={k}, {y}, by {X} boxplot Season={season}.png', bbox_inches='tight')

    # plot average temp change by hour/month
    if avg_temp_change_by_X:
        for horizon in range(1, 2):
            dh = df[(df.daysahead == horizon) & (df.DST.isin(DST))]
            # if season_no == 1:
            #     dh = df[(df.daysahead == horizon) & (df.season == season_no)]
            # else:
            #     dh = df[(df.daysahead == horizon) & (df.season != 1)]
            plt.rcParams['xtick.labelsize'] = 9  # set tick size for xaxis
            fig, ax = plt.subplots(nrows=2, ncols=3, sharex='col', sharey='row',
                                   figsize=(8, 5.5), dpi=200)  # nrows=3, figsize=(7, 7)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

            if horizon == 0:
                title = f'Year={year}, Avg Absolute Same-Day Forecast Temp Change by {X_title}, ' + \
                        r"$\bf{DST=" + str(DST) + "}$"
            elif horizon == 1:
                title = f'Year={year}, Avg Absolute {horizon} Day Ahead Forecast Temp Change by {X_title}, ' + \
                        r"$\bf{DST=" + str(DST) + "}$"
                # title = f'Year={year}, Avg Absolute {horizon} Day Ahead Forecast Temp Change by {X_title}, ' + \
                #         r"$\bf{Season=" + str(season) + "}$"
            else:
                title = f'Year={year}, Avg Absolute {horizon} Days Ahead Forecast Temp Change by {X_title}, ' + \
                        r"$\bf{DST=" + str(DST) + "}$"

            # fig.suptitle(title)

            for i in range(3):  # col
                for j in range(2):  # row
                    colname = 'abs_f_dif' + str(i + j * 3 + 1)
                    temp_df = dh.groupby(X)[colname].mean()
                    ax[j, i].plot(temp_df.index, temp_df, marker="o")

                    # highlighted hours
                    if X_title == 'Hour':
                        if horizon == 0:
                            for XX in highlight_X_0:
                                if i + j * 3 + 1 <= 4:  # within the first 3 hours
                                    ax[j, i].axvline(x=XX, color='gray', linestyle=':', linewidth=1)  # linewidth=1.5
                                    # ax[j, i].axhline(y=temp_df[XX], color='gray', linestyle=':', linewidth=1.5)
                                    ax[j, i].plot(XX, temp_df[XX], marker="o", c='orange')  # change dot color
                        if horizon == 1:
                            for XX in highlight_X_1:
                                ax[j, i].axvline(x=XX, color='gray', linestyle=':', linewidth=1)  # linewidth=1.5
                                # ax[j, i].axhline(y=temp_df[XX], color='gray', linestyle=':', linewidth=1.5)
                                ax[j, i].plot(XX, temp_df[XX], marker="o", c='orange')  # change dot color

                    if X_title == 'Hour':
                        if horizon == 0:
                            ax[j, i].set(xlim=(2, 19))
                            ax[j, i].xaxis.set_ticks(np.arange(3, 19, 2))
                        else:  # 1-day-ahead has all 24 hours
                            ax[j, i].set(xlim=(-2, 25))
                            ax[j, i].xaxis.set_ticks(np.arange(0, 24, 3))
                    else:  # X_title=='Month'
                        ax[j, i].set(xlim=(0, 13))
                        ax[j, i].xaxis.set_ticks(np.arange(1, 13, 1))

                    if (i == 0) & (j == 0):
                        ax[j, i].set_title(f'Within {i + j * 3 + 1} Hour', fontsize=11)
                    else:
                        ax[j, i].set_title(f'Within {i + j * 3 + 1} Hours', fontsize=11)
                    ax[j, i].title.set_position([.5, 1])  # adjust distance between title and plot

            plt.xlabel(X_title, fontsize=12)
            plt.ylabel('Average Absolute $T_F$ Variation (°F)', fontsize=12)
            ax = plt.gca() # get current axis
            ax.get_yaxis().set_label_coords(-0.07, 0.5) # adjust position of y label

            if save_plot:
                plt.savefig(f'graphs/station={k}, {title[:-17]} DST={DST}.png', bbox_inches='tight')
                # plt.savefig(f'graphs/station={k}, avg temp change by {X} Season={season}.png', bbox_inches='tight')

if show_plot:
    plt.show()
