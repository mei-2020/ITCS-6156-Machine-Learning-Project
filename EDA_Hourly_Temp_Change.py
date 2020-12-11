# Hourly Temp change impact to errors & Error by month plot

# %% import & process
# import matplotlib
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
from Methods import import_preprocess

# ----------------------------------------------------------------------------------------------------------------#
station_list = ['KECG', 'KASJ', 'KMWK', 'KHBI', 'KEWN', 'KPGV', 'KIPJ', 'KGSB', 'KCPC', 'KORF', 'KGSP', 'KHNZ', 'KNCA',
                'KRWI', 'KIGX', 'KILM', 'KGEV', 'KEYF', 'KHSE', 'KPOB', 'KHKY', 'KMRH', 'KCLT', 'KMRN', 'KGSO', 'KAVL',
                'KAFP', 'KRDU']
# k_list = ['KAFP', 'KASJ', 'KAVL', 'KCLT', 'KCPC', 'KECG']
k_list = ['KAVL']
y = 'abs_err'
x = 'abs_f_'
year = [2019]  # year(s) in list
reg_line = True
plot_scatter = True  # plot forecast error vs. Hourly Temp change
by_month_boxplot = False  # plot by month boxplot
avg_temp_change_by_month = False
time_adjust = True
# --------------------------------#
check_actual_outlier_boxplot = 0  # check if there is any outliers/data issues in actual temperature data
check_daysahead_err_boxplot = 0  # check errors for each daysahead
# --------------------------------#
save_plot = 0
show_plot = 1
# zero_dif = 1  # find out when forecast Temp change is 0, what's the forecast error looks like
after_1000 = 0  # this is used for apple-to-apple comparison between same-day and 1-day-ahead. same-day only has forecasts after 1000am
highlight_month = []

# ----------------------------------------------------------------------------------------------------------------#

# %% Plot
for k in k_list:
    if plot_scatter:
        df = import_preprocess(station=k, yr=year)
        for horizon in range(1, 2):
            dh = df[df.daysahead == horizon]

            if after_1000:
                dh = df[(df.year.isin(year)) & (df.daysahead == horizon) & (df.hour > 10)]

            # show average error on plots
            avg_err = np.around(np.mean(df[(df.year.isin(year)) & (df.daysahead == horizon)].err), decimals=3)

            # if zero_dif:
            #     for i in range(3):  # col
            #         for j in range(3):  # row
            #             colname = 'dif' + str(i + j * 3 + 1)
            #             zero = dh[dh[x + colname] == 0]
            #             freq = zero.groupby([x + colname, y]).size().reset_index(name="occurrance")  # coordinate freq
            #             freq['occurrance'] = freq.occurrance.div(freq.occurrance.max() - freq.occurrance.min())  # normalize

            fig, ax = plt.subplots(nrows=3, ncols=3, sharex='col', sharey='row', figsize=(7, 7), dpi=200)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

            if x == 'abs_f_':
                plt.xlabel('Absolute $T_{F}$ Variation (°F)', fontsize=12)
            elif x == 'abs_a_':
                plt.xlabel('Absolute Actual Temperature Change (°F)', fontsize=12)
            elif x == 'f_':
                plt.xlabel('Forecast Temperature Change (°F)', fontsize=12)
            else:
                plt.xlabel('Actual Temperature Change (°F)', fontsize=12)

            if y == 'err':
                plt.ylabel(f'Forecast Error (°F), \u03BC={avg_err}', fontsize=12)
            else:
                plt.ylabel('Absolute Forecast Error (°F)', fontsize=12)

            if horizon == 0:
                title = f'Year={year}, Same-Day Forecast'
            elif horizon == 1:
                title = f'Year={year}, {horizon} Day Ahead Forecast'
            else:
                title = f'Year={year}, {horizon} Days Ahead Forecast'

            # fig.suptitle(title)

            for i in range(3):  # col
                for j in range(3):  # row
                    colname = 'dif' + str(i + j * 3 + 1)
                    freq = dh.groupby([x + colname, y]).size().reset_index(name="occurrance")  # coordinate freq
                    freq['occurrance'] = freq.occurrance.div(freq.occurrance.max() - freq.occurrance.min())  # normalize

                    sc = ax[j, i].scatter(freq[x + colname], freq[y], c=freq.occurrance, s=10, alpha=.5,
                                          marker="o", edgecolor='none', cmap='jet')

                    if (y[:3] == 'abs') & (x[:3] == 'abs'):
                        ax[j, i].set(xlim=(-2, 41), ylim=(-2, 21))
                        ax[j, i].xaxis.set_ticks(np.arange(0, 41, 20))
                        ax[j, i].yaxis.set_ticks(np.arange(0, 21, 5))
                    elif (y[:3] != 'abs') & (x[:3] == 'abs'):
                        ax[j, i].set(xlim=(-2, 41), ylim=(-21, 21))
                        ax[j, i].xaxis.set_ticks(np.arange(0, 41, 20))
                        ax[j, i].yaxis.set_ticks(np.arange(-20, 21, 10))
                        ax[j, i].axhline(y=0, color='gray', linestyle=':', linewidth=2)
                    elif (y[:3] == 'abs') & (x[:3] != 'abs'):
                        ax[j, i].set(xlim=(-41, 41), ylim=(-2, 21))
                        ax[j, i].xaxis.set_ticks(np.arange(-40, 41, 40))
                        ax[j, i].yaxis.set_ticks(np.arange(0, 21, 5))
                        ax[j, i].axvline(x=0, color='gray', linestyle=':', linewidth=2)
                    else:  # both y and x are non-abs values
                        ax[j, i].set(xlim=(-41, 41), ylim=(-21, 21))
                        ax[j, i].xaxis.set_ticks(np.arange(-40, 41, 40))
                        ax[j, i].yaxis.set_ticks(np.arange(-20, 21, 10))
                        ax[j, i].axvline(x=0, color='gray', linestyle=':', linewidth=2)
                        ax[j, i].axhline(y=0, color='gray', linestyle=':', linewidth=2)

                    temp_df = pd.concat([dh[x + colname], dh[y]], axis=1)  # 只有用于画regression的两个col
                    temp_df.dropna(inplace=True)  # 手动去除na before plotting regression
                    temp_pos = temp_df[temp_df.iloc[:, 0] > 0]  # x>=0, 画一四象限的regression line
                    temp_neg = temp_df[temp_df.iloc[:, 0] < 0]  # x<=0, 画二三象限的regression line

                    if reg_line:
                        if (x[:3] == 'abs'):  # if temp change is in absolute value, 1 reg line
                            x_ax = np.arange(-44, 45, 1)
                            b, m = polyfit(temp_df.iloc[:, 0], temp_df.iloc[:, 1], 1)
                            ax[j, i].plot(x_ax, b + m * x_ax, '-', c='black', linewidth=2)
                            plt.text(.78, .9, f'm={round(m, 3)}', horizontalalignment='center',
                                     verticalalignment='center', transform=ax[j, i].transAxes, fontsize=8.5,
                                     fontweight='bold')
                        else:  # if temp change is in with sign, 2 reg lines, one when x>=0 and one when x<0
                            x_ax = np.arange(0, 45, 1)
                            b, m = polyfit(temp_pos.iloc[:, 0], temp_pos.iloc[:, 1], 1)
                            ax[j, i].plot(x_ax, b + m * x_ax, '-', c='black', linewidth=2)

                            x_ax = np.arange(-45, 1, 1)
                            b, m = polyfit(temp_neg.iloc[:, 0], temp_neg.iloc[:, 1], 1)
                            ax[j, i].plot(x_ax, b + m * x_ax, '-', c='black', linewidth=2)

                    if (i == 0) & (j == 0):
                        ax[j, i].set_title(f'Within {str(i + j * 3 + 1)} Hour', fontsize=10)
                    else:
                        ax[j, i].set_title(f'Within {str(i + j * 3 + 1)} Hours', fontsize=10)
                    ax[j, i].title.set_position([.5, 1])  # adjust distance between title and plot

            if save_plot:
                if reg_line:
                    if (x[:3] == 'abs'):
                        plt.savefig(f'graphs/station={k}, {y} vs. {x} {title}.png', bbox_inches='tight')
                    else:
                        plt.savefig(f'graphs/station={k}, {y} vs. {x}, 2 Reg lines {title}.png', bbox_inches='tight')

    if by_month_boxplot:
        df = import_preprocess(station=k, yr=year)
        sns.set_style("ticks")
        for horizon in range(2):
            dh = df[df.daysahead == horizon]
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='month', y=y, data=dh)
            plt.xlabel('Month', fontsize=12)

            if horizon == 0:
                title = f'Year={year}, Same-Day Forecast Error (°F) by Month'
            elif horizon == 1:
                title = f'Year={year}, 1 Day Ahead Forecast Error (°F) by Month'
            else:
                title = f'Year={year}, {horizon} Days Ahead Forecast Error (°F) by Month'
            plt.title(title)

            if y == 'err':
                plt.ylabel(f'Forecast Error (°F)', fontsize=12)
                plt.axhline(y=0, color='gray', linestyle=':', linewidth=2)
                plt.axhline(y=5, color='gray', linestyle=':', linewidth=2)
                plt.axhline(y=10, color='gray', linestyle=':', linewidth=2)
                plt.axhline(y=-5, color='gray', linestyle=':', linewidth=2)
                plt.axhline(y=-10, color='gray', linestyle=':', linewidth=2)
                ax = plt.gca()  # get current axis
                ax.set(ylim=(-21, 21))
                ax.yaxis.set_ticks(np.arange(-20, 21, 5))
                if save_plot:
                    plt.savefig(f'graphs/station={k}, {y}, {title}.png', bbox_inches='tight')
            else:
                plt.ylabel('Absolute Forecast Error (°F)', fontsize=12)
                plt.axhline(y=5, color='gray', linestyle=':', linewidth=2)
                plt.axhline(y=10, color='gray', linestyle=':', linewidth=2)
                ax = plt.gca()  # get current axis
                ax.set(ylim=(0, 21))
                ax.yaxis.set_ticks(np.arange(0, 21, 5))
            if save_plot:
                plt.savefig(f'graphs/station={k}, {y}, {title}.png', bbox_inches='tight')

    if avg_temp_change_by_month:
        df = import_preprocess(station=k, yr=year)
        for horizon in range(2):
            dh = df[df.daysahead == horizon]
            plt.rcParams['xtick.labelsize'] = 8  # set tick size for xaxis
            fig, ax = plt.subplots(nrows=3, ncols=3, sharex='col', sharey='row', figsize=(7, 7))
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

            if horizon == 0:
                title = f'Year={year}, Avg Absolute Same-Day Forecast Temp Change by Month'
            elif horizon == 1:
                title = f'Year={year}, Avg Absolute {horizon} Day Ahead Forecast Temp Change by Month'
            else:
                title = f'Year={year}, Avg Absolute {horizon} Days Ahead Forecast Temp Change by Month'

            fig.suptitle(title)
            for i in range(3):  # col
                for j in range(3):  # row
                    colname = 'abs_f_dif' + str(i + j * 3 + 1)
                    temp_df = dh.groupby('month')[colname].mean()
                    ax[j, i].plot(temp_df.index, temp_df, marker="o")

                    # highlighted months
                    for mon in highlight_month:
                        ax[j, i].axvline(x=mon, color='gray', linestyle=':', linewidth=2)
                        ax[j, i].axhline(y=temp_df[mon], color='gray', linestyle=':', linewidth=2)

                    ax[j, i].set(xlim=(0, 13))
                    ax[j, i].xaxis.set_ticks(np.arange(1, 13, 1))

                    if (i == 0) & (j == 0):
                        ax[j, i].set_title(f'Within {i + j * 3 + 1} Hour', fontsize=10)
                    else:
                        ax[j, i].set_title(f'Within {i + j * 3 + 1} Hours', fontsize=10)
                    ax[j, i].title.set_position([.5, 1])  # adjust distance between title and plot

            plt.xlabel('Month', fontsize=12)
            plt.ylabel('Average Absolute Forecast Temp Change (°F)', fontsize=12)

            if save_plot:
                plt.savefig(f'graphs/station={k}, {title}.png', bbox_inches='tight')

if check_actual_outlier_boxplot:
    sns.set_style("whitegrid")
    df = pd.DataFrame()  # create an empty dataframe
    for k in station_list:
        actual = import_preprocess(station=k, yr=year, preprocess=False)
        actual.temp.rename(k)
        df = pd.concat([df, actual.temp.rename(k)], axis=1)

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df)
    plt.ylabel('Measured Temp (°F)', fontsize=12)
    plt.title(f'Measured Temp (°F) of Stations')

if check_daysahead_err_boxplot:  # not available on the fcst set when we only have 0 and 1
    sns.set(style="ticks")  # set stype for all plots
    for k in k_list:
        df = import_preprocess(station=k, yr=year, daysahead=(0, 1, 2, 3, 4, 5, 6, 7), fcst_diff=False)
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='daysahead', y='err', data=df)
        plt.xlabel('Days Ahead Forecast')
        plt.ylabel('Error (°F)')

        if time_adjust:
            title = f'Station={k}, Error Boxplot by daysahead, after time zone adjustment'
        else:
            title = f'Station={k}, Error Boxplot by daysahead, before time zone adjustment'
        plt.title(title)

        if save_plot:
            plt.savefig(f'graphs/station={k}, {title}.png', bbox_inches='tight')

if show_plot:
    plt.show()
