# %% Use linear regression to predict the quality (good/bad) of hourly Temp forecast
# import matplotlib
#
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
import pandas as pd
import sys, os, time, re
import patsy
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix, SCORERS, roc_curve, \
    plot_roc_curve, classification_report, roc_auc_score, accuracy_score, recall_score, average_precision_score, \
    precision_recall_curve
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=UserWarning)

# set pandas printing: https://stackoverflow.com/questions/11707586/how-do-i-expand-the-output-display-to-see-more-columns-of-a-pandas-dataframe
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

station_list = ['KECG', 'KASJ', 'KMWK', 'KHBI', 'KEWN', 'KPGV', 'KIPJ', 'KGSB', 'KCPC', 'KORF', 'KGSP', 'KHNZ', 'KNCA',
                'KRWI', 'KIGX', 'KILM', 'KGEV', 'KEYF', 'KHSE', 'KPOB', 'KHKY', 'KMRH', 'KCLT', 'KMRN', 'KGSO', 'KAVL',
                'KAFP', 'KRDU']

# %% ---------------------------------------------Parameter Setting----------------------------------------------------#

k_list = ['KAFP', 'KASJ', 'KAVL', 'KCLT', 'KCPC', 'KECG', 'KEWN', 'KEYF', 'KGEV', 'KGSB', 'KGSO', 'KGSP', 'KHBI',
          'KHKY', 'KHNZ', 'KHSE', 'KIGX', 'KILM', 'KIPJ', 'KMRH']
k_list = ['KASJ']

columns = ['station', 'method', 'null_acc', 'mae', 'acc', 'tpr', 'fpr', 'tnr', 'auroc', 'auprc', 'best_thres',
           'tpr_best', 'fpr_best', 'tnr_best', 'exec_time']
result_df = pd.DataFrame(columns=columns)  # create empty dataframe to store results
user = 'Administrator'  # Administrator or yli109
# method = ['LinReg', 'LogReg', 'NB', 'KNN', 'SVM', 'RF', 'GB', 'XGB']  # LinReg, LogReg, NB, KNN, SVM, RF, GB, XGB
method = ['LogReg', 'NB', 'KNN', 'SVM', 'RF', 'GB', 'XGB']
method = ['RF']
year = [2018]

boundary = 5  # if abs_err>=boundary, bad forecast; otherwise, good
n_jobs = 40

# --- Below are for graphing --- #
report_auroc = True  # True: output auroc value and roc curve; False: output auprc value and PR curve
plot_optimal_thres = False  # whether to plot the black optimal threshold dots on graph
show_thres_XGBoost = False  # whether to show threshold indicator on XGBoost
thres_ms = 50  # marker size for optimal decision threshold
curve_lw = 1.5  # roc/pr curve linewidth
curve_ms = 0  # roc/pr curve marker size
plot_roc = 0  # plot ROC/PR curve. 1: ROC, 0: PR
plot_roc_combine = 0  # combine ROC/PR curves in one plot
plot_title = False  # whether include title in plot
plot_save = False  # whether to save the plots
plot_confusion, alpha = True, 17.13 / 100  # use sklearn to plot confusion matrix, alpha is the threshold
plot_corr_matrix = False  # plot correlation matrix for variables

# --- Below are for experiment --- #
metric = 'roc_auc'  # roc_auc (AUROC), average_precision (F1 score, also AUPRC)
k_features_search_range = [-7, 0]  # if max feature num is 24, then search 17 - 24 vars
OverSampling = False  # this is to over sampling the training data
balanced_class = 1  # this is for LogReg, SVM, RF, XGBoost
use_1hot_trees = False  # whether or not to use one-hot encoding for ensemble methods
use_DST = True  # True: use DST to segregate; False: use winter to segregate
use_sunshine = True  # True: include 'sunshine_pct_possible_fcst' and 'sunshine_ind' in model. ind=1: daytime when sunshine available
result_df_save = False  # whether to save result_df

# --- Below are for debugging --- #
filter_before_GridSearch = 0  # whether conduct feature engineering before GridSearch
feature_discretization = False  # whether to discretize numerical features into categorical
show_sfs_var_selection = False  # whether to show best combination (MAE by 5-fold CV:  1.762), 15 out of 24 features selected...
print_GSearch_details = False  # whether to print GridSearchCV result details

# %% ---------------------------------------Modeling-------------------------------------- #
weather_hourly = 1  # if 1, include hourly weather vars
weather_daily = 0  # if 1, include daily weather vars (min/max/avg)
abs_f_dif_num = 4  # Forecast Temp change within X hours
fcst_avg_Temp_change_num = 2  # Forecast daily average Temp change within X(4) hours
lag_1hr_Temp_diff_num = 0  # Lagged Temp change within 1 hour. Increase the # will add more distant vars 1-hr Temp change to be used
hour_categorical = 1
interact_hr = 0  # add interaction effect for C(hour)
horizon = 1  # focus on 1-day-ahead
plot_feature_selection = 0  # plot features selection graph
seed = 1

base_model = 'fcst_quality ~ -1 + daily_f_range'  # keep intercept.
abs_f_dif_str, fcst_avg_Temp_change_str, lag_1hr_Temp_diff_str = '', '', ''  # initialize
W_var = ' + rel_humidity_fcst_X + C(cloud_cover_type_fcst) + visibility_fcst_X + water_equivalent_fcst_X + precip_probability_fcst_X '  # does not have solar

if use_sunshine:
    W_var += '+ sunshine_pct_possible_fcst_X '
if hour_categorical:
    base_model += ' + C(class_hr)'
## Temp change at the hour
for i in range(1, abs_f_dif_num + 1):
    abs_f_dif_str += f' + abs_f_dif{i}'
for i in range(1, fcst_avg_Temp_change_num + 1):
    fcst_avg_Temp_change_str += f' + abs_f_dif{i}_avg'
## Lag Temp change at the hour
for i in range(1, lag_1hr_Temp_diff_num + 1):
    lag_1hr_Temp_diff_str += f' + absf_dif1_lag{i}'
# ----------------------------------------------------------#
## Add weather variables
if weather_daily:
    base_model += W_var.replace('X', 'daily')
if weather_hourly:
    base_model += W_var.replace('X', 'hrly')

f = base_model + abs_f_dif_str + fcst_avg_Temp_change_str + lag_1hr_Temp_diff_str

if interact_hr:
    f = f.replace('abs_f_dif', 'C(class_hr) * abs_f_dif')

f_tree = f.replace('(', '').replace(')', '').replace('C', '').replace('cloud_cover_type_fcst', 'cloud_cover_fcst_hrly')
f_linreg = f.replace('fcst_quality', 'abs_err')  # this f_linreg is for LinearRegression only

# %% ----------------------------------Define Functions & extra parameter setting-------------------------------------#
# noinspection PyUnresolvedReferences
from Methods import import_preprocess, mean_max_process, class_metrics, linreg_classifier


def pre_filter_features(k_features, forward=False, floating=False):
    """conduct feature engineering before GridSearch."""
    sfs = SFS(clf, k_features=k_features, forward=forward, floating=floating, scoring=metric, cv=5,
              n_jobs=n_jobs)
    pipe = make_pipeline(StandardScaler(), sfs)
    pipe.fit(X_tr, y_tr)
    print(f'Initial Feature selection (AUC by 5-fold CV: {sfs.k_score_:.3}), {len(sfs.k_feature_idx_)} '
          f'out of {X_tr.shape[1]} features selected: {sfs.k_feature_idx_}')
    X_tr_sfs = sfs.transform(X_tr)
    return X_tr_sfs


def Step_GSearch(param_grid, pipe, X, print_GSearch_details=False):
    """Conduct step by step GridSearch. """
    gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring=metric, n_jobs=n_jobs, cv=5, verbose=0)
    gs = gs.fit(X, y_tr)  # use subset of features
    if print_GSearch_details:
        for i in range(len(gs.cv_results_['params'])):
            print(gs.cv_results_['params'][i], 'test acc.:', gs.cv_results_['mean_test_score'][i])
        print(f'Best parameters via GridSearch {gs.best_params_}, best score: {gs.best_score_:.3}')
    pipe = gs.best_estimator_  # update searched parameters
    return pipe


def print_store_results(method_name, class_report=False):
    """Print Classfication results."""
    auroc = roc_auc_score(y_test, model.predict_proba(X_test_sfs)[:, 1])  # area under ROC curve
    auprc = average_precision_score(y_test, model.predict_proba(X_test_sfs)[:, 1])  # area under PRC curve
    y_pred = model.predict(X_test_sfs)
    if class_report:  # show classification report
        print(classification_report(y_test, y_pred, target_names=['Good', 'Bad'], digits=4))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc = accuracy_score(y_test, y_pred)
    tpr = recall_score(y_test, y_pred)
    tnr = tn / (tn + fp)
    print(f'Accuracy : {acc:.4g}')
    print(f'Recall : {tpr:.4g}')
    print(f'Specificity : {tnr:.4g}')
    print(f'AU-ROC Score (test): {auroc:.3g}')
    print(f'AU-PRC Score (test): {auprc:.3g}')

    t2 = time.perf_counter()
    # Insert new row to results dataframe
    global result_df
    result_df = result_df.append(
        {'station': k, 'method': method_name, 'null_acc': null_accuracy, 'mae': float('nan'), 'acc': acc,
         'tpr': tpr, 'fpr': 1 - tnr, 'tnr': tnr, 'auroc': auroc, 'auprc': auprc,
         # insert later
         'best_thres': float('nan'), 'tpr_best': float('nan'), 'fpr_best': float('nan'), 'tnr_best': float('nan'),
         'exec_time': round((t2 - t1) / 60, 2)}, ignore_index=True)

    if report_auroc:  # if use auroc as metric
        auc = auroc
    else:
        auc = auprc

    return auc


def rename_cols(data):
    """a generic way of removing [, ] or < from pandas column names. This is to avoid error from XGBoost."""
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    data.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                    data.columns.values]
    return data


def locate_plot_optimal_thres(apply_label=False):
    """
    locate optimal threshold for each method, before plotting the threshold.
    Parameters
    ----------
    apply_label: whether to apply legend label for threshold. Should only apply once
    """
    global result_df, tpr, fpr, idx, plot_optimal_thres, thres_ms, show_thres_indicator
    idx = argmax(tpr - fpr)
    best_thres = deci_thres[idx]
    print(f'Best Threshold={best_thres:.3f}')
    result_df.iloc[-1, result_df.columns.get_loc('best_thres')] = best_thres  # insert best_thres to last row
    result_df.iloc[-1, result_df.columns.get_loc('tpr_best')] = tpr[idx]  # insert tpr_best to last row
    result_df.iloc[-1, result_df.columns.get_loc('fpr_best')] = fpr[idx]
    result_df.iloc[-1, result_df.columns.get_loc('tnr_best')] = 1 - fpr[idx]

    if plot_optimal_thres:
        if apply_label:  # apply label only on the last method, XGBoost, to avoid to many labels
            plt.scatter(fpr[idx], tpr[idx], marker='o', color='black', s=thres_ms, label='Optimal Threshold')
        else:
            plt.scatter(fpr[idx], tpr[idx], marker='o', color='black', s=thres_ms)


# %% -------------------------Training Loop starts-----------------------------#
for k in k_list:
    t_start = time.perf_counter()
    if any(x in ['LogReg', 'SVM', 'RF'] for x in method):  # balanced_class for 'LogReg', 'SVM', 'RF'
        print(
            f'Station={k}, year={year}, method={method}, boundary=\u00B1{boundary}, balanced_class={balanced_class}, OverSampling={OverSampling}.')
    else:
        print(f'Station={k}, year={year}, method={method}, boundary=\u00B1{boundary}, OverSampling={OverSampling}.')

    # ---------------------------------------Preprocessing-------------------------------------- #
    df = import_preprocess(station=k, yr=year, fulldata=True, actual_diff=False, fcst_diff=True)
    df.dropna(subset=['err', 'abs_err'], inplace=True)
    daily_weather = mean_max_process(data=df).drop('abs_err', axis=1)
    df = df.merge(daily_weather, how='left', on=['daysahead', 'date'], suffixes=('_hrly', '_daily'))
    df['fcst_quality'] = np.where(df['abs_err'] >= boundary, 1, 0)  # 1=bad=positive, 0=good=negative
    # np.ptp returns the range (max-min) of an array
    df['daily_f_range'] = df.groupby(['daysahead', 'date'])['temp_fcst'].transform(np.ptp)
    # Forecast average Temp change
    for i in range(1, 10, 1):
        df[f'abs_f_dif{i}_avg'] = df.groupby(['daysahead', 'date'])[f'abs_f_dif{i}'].transform(np.mean)  # avg T change
    # get lagged temperature change (within 1 hr)
    for i in range(1, lag_1hr_Temp_diff_num + 1):
        colname = 'absf_dif1_lag' + str(i)  # construct col name
        df[colname] = df['abs_f_dif1'].shift(i)

    # Categorical hours during non-DST/non-winter, [6, 7, 8, 9, 17, 18, 19], other non-DST/non-winter hours=1 and DST/winter hours=0
    if use_DST:
        df['class_hr'] = np.where(df.DST == 0, 0, df.hour)  # DST=0 -> class_hr=0
    else:
        df['class_hr'] = np.where(df.season == 1, 0, df.hour)  # season=1 -> winter -> class_hr=0
    df['class_hr'] = np.where((df.class_hr != 0) & (~df.hour.isin([6, 7, 8, 9, 17, 18, 19])), 1, df['class_hr'])

    # Generate design matrix
    y, X = patsy.dmatrices(f, df[df.daysahead == horizon], return_type='dataframe')  # for all other
    y_lr, X_lr = patsy.dmatrices(f_linreg, df[df.daysahead == horizon], return_type='dataframe')  # for linreg
    _, X_es = patsy.dmatrices(f_tree, df[df.daysahead == horizon], return_type='dataframe')  # ensemble, _=y is omitted

    if feature_discretization:  # this is for LogReg and KNN. But not working well
        # col_pos = X.columns.get_loc('daily_f_range')  # get col position of daily_f_range, which is the 1st numerical var
        est = KBinsDiscretizer(n_bins=5, encode='onehot', strategy='quantile')
        est.fit(X)
        Xt = est.transform(X)  # transformed X

    # import the module to oversample the minority class, if OverSampling=True
    if OverSampling:
        stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')  # remove Using TensorFlow backend
        from imblearn.pipeline import Pipeline, make_pipeline
        from imblearn.over_sampling import SMOTE

        sys.stderr = stderr
    else:
        from sklearn.pipeline import Pipeline, make_pipeline

    # Data partitioning
    ## For linear reg, no stratify
    X_tr_lr, X_test_lr, y_tr_lr, y_test_lr = train_test_split(X_lr, y_lr, stratify=(y_lr >= boundary), test_size=0.2,
                                                              random_state=seed)  # 80/20 training

    ## For tree based ensemble models, try without one-hot encoding
    X_tr_es, X_test_es, _, _ = train_test_split(X_es, y, stratify=y, test_size=0.2, random_state=seed)  # 80/20 training
    X_tr_es, X_test_es = rename_cols(X_tr_es), rename_cols(X_test_es)  # rename columns to avoid error in XGBoost

    ## for classfiers
    if feature_discretization:
        X_tr, X_test, y_tr, y_test = train_test_split(Xt, y, stratify=y, test_size=0.2, random_state=seed)
    else:
        X_tr, X_test, y_tr, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=seed)  # 80/20

    X_tr, X_test = rename_cols(X_tr), rename_cols(X_test)  # rename cols to avoid error in XGBoost (duplicate)
    pos_neg_rate = np.sum(y_test == 0)[0] / np.sum(y_test == 1)[0]  # no. of positive class / negative class
    null_accuracy = max(np.sum(y_test == 1)[0], np.sum(y_test == 0)[0]) / len(y_test)
    print(f'Null Accuracy= {null_accuracy:.2%}')
    y_tr, y_test = np.ravel(y_tr), np.ravel(y_test)  # flatten to 1-D 扁平化
    k_range = tuple(X_tr.shape[1] + np.asarray(k_features_search_range))  # convert to tuple for k_features_search_range

    # ---------------------------------------Model fitting and prediction-------------------------------------- #
    if plot_roc_combine:
        fig = plt.figure(figsize=(6, 5.5), dpi=200)  # create an empty figure, for combined ROC curves. dpi default=100

    if 'LinReg' in method:
        print(f'|Training of LinearRegression starts......|')
        t1 = time.perf_counter()
        sfs = SFS(LinearRegression(), k_features='best', forward=True, floating=True,
                  scoring='neg_mean_absolute_error', cv=5, n_jobs=n_jobs)
        sfs = sfs.fit(X_tr_lr, y_tr_lr)
        if show_sfs_var_selection:
            print(f'best combination (MAE by 5-fold CV: {-sfs.k_score_: .4}), {len(sfs.k_feature_idx_)} '
                  f'out of {X.shape[1]} features selected: {sfs.k_feature_idx_}')
        # predict using selected features
        X_test_sfs = sfs.transform(X_test_lr)
        cm, mae = linreg_classifier(X=X_test_sfs, y=y_test_lr, thres=boundary)  # print stats after selection
        acc, tpr, tnr = class_metrics(cm, null_accuracy)  # print metrics and return acc, tpr, tnr
        t2 = time.perf_counter()
        result_df = result_df.append({'station': k, 'method': 'LinReg', 'null_acc': null_accuracy, 'mae': mae,
                                      'acc': acc, 'tpr': tpr, 'fpr': 1 - tnr, 'tnr': tnr, 'auroc': float('nan'),
                                      'auprc': float('nan'), 'best_thres': float('nan'), 'tpr_best': float('nan'),
                                      'fpr_best': float('nan'), 'tnr_best': float('nan'),
                                      'exec_time': round((t2 - t1) / 60, 2)}, ignore_index=True)

    if 'LogReg' in method:
        print(f'|Training of LogisticRegression starts......|')
        t1 = time.perf_counter()
        if balanced_class:
            clf = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=seed)
        else:
            clf = LogisticRegression(random_state=seed, max_iter=5000, n_jobs=n_jobs)

        sfs = SFS(clf, k_features='best', forward=True, floating=True, scoring=metric, cv=5, n_jobs=n_jobs)

        if OverSampling:  # if oversampling, use SMOTE in Pipeline
            pipe = make_pipeline(StandardScaler(), SMOTE(random_state=seed, n_jobs=n_jobs), sfs)
        else:
            pipe = make_pipeline(StandardScaler(), sfs)

        pipe.fit(X_tr, y_tr)
        if show_sfs_var_selection:
            print(f'best combination (AUC by 5-fold CV: {sfs.k_score_:.3}), {len(sfs.k_feature_idx_)} '
                  f'out of {X_tr.shape[1]} features selected: {sfs.k_feature_idx_}')
        # predict using selected features
        X_tr_sfs = sfs.transform(X_tr)
        X_test_sfs = sfs.transform(X_test)
        pipe = make_pipeline(StandardScaler(), clf)  # make another pipeline for classifier
        model = pipe.fit(X_tr_sfs, y_tr)

        auc = print_store_results(method_name='LR', class_report=False)

        if plot_roc_combine and report_auroc:
            fpr, tpr, deci_thres = roc_curve(y_test, model.predict_proba(X_test_sfs)[:, 1])
            plt.plot(fpr, tpr, label=f'LR, AUC={auc:.3}', linewidth=curve_lw, marker='.', markersize=curve_ms)

            # locate and plot (if plot_optimal_thres=true) best decision threshold, on ROC curve
            locate_plot_optimal_thres(apply_label=False)


        elif plot_roc_combine and report_auroc == False:
            precision, recall, deci_thres = precision_recall_curve(y_test, model.predict_proba(X_test_sfs)[:, 1])
            plt.plot(recall, precision, linewidth=curve_lw, marker='.', markersize=curve_ms, label=f'LR'
                     #             f', AUC={auc:.3}',
                     )

    # NB will use the same variables as ensemble methods
    if 'NB' in method:
        print(f'|Training of Naive Bayes starts......|')
        t1 = time.perf_counter()

        clf = ComplementNB()
        sfs = SFS(clf, k_features='best', forward=True, floating=True, scoring=metric, cv=5, n_jobs=n_jobs)

        if OverSampling:  # if oversampling, use SMOTE in Pipeline
            pipe = make_pipeline(SMOTE(random_state=seed, n_jobs=n_jobs), sfs)
            pipe_model = make_pipeline(SMOTE(random_state=seed, n_jobs=n_jobs), clf)
        else:
            pipe = make_pipeline(sfs)
            pipe_model = make_pipeline(clf)

        pipe.fit(X_tr, y_tr)
        if show_sfs_var_selection:
            print(f'best combination (AUC 5-fold CV: {sfs.k_score_:.3}), {len(sfs.k_feature_idx_)} '
                  f'out of {X_tr.shape[1]} features selected: {sfs.k_feature_idx_}')
        # predict using selected features
        X_tr_sfs = sfs.transform(X_tr)
        X_test_sfs = sfs.transform(X_test)
        model = pipe_model.fit(X_tr_sfs, y_tr)

        auc = print_store_results(method_name='NB', class_report=False)

        if plot_roc_combine and report_auroc:
            fpr, tpr, deci_thres = roc_curve(y_test, model.predict_proba(X_test_sfs)[:, 1])
            plt.plot(fpr, tpr, label=f'NB, AUC={auc:.3}', linewidth=curve_lw, marker='.', markersize=curve_ms)

            # locate and plot (if plot_optimal_thres=true) best decision threshold, on ROC curve
            locate_plot_optimal_thres(apply_label=False)

        elif plot_roc_combine and report_auroc == False:
            precision, recall, deci_thres = precision_recall_curve(y_test, model.predict_proba(X_test_sfs)[:, 1])
            plt.plot(recall, precision, linewidth=curve_lw, marker='.', markersize=curve_ms, label=f'NB'
                     #             f', AUC={auc:.3}',
                     )

    if 'KNN' in method:
        print(f'|Training of KNN starts......|')
        t1 = time.perf_counter()
        # first, get rid of some features
        clf = KNeighborsClassifier(n_jobs=n_jobs)
        if filter_before_GridSearch:
            X_tr_new = pre_filter_features(k_features='parsimonious')
        else:
            X_tr_new = X_tr

        if OverSampling:  # if oversampling, use SMOTE in Pipeline
            pipe = Pipeline(
                [('scale', StandardScaler()), ('SMOTE', SMOTE(random_state=seed, n_jobs=n_jobs)), ('clf', clf)])
        else:
            pipe = Pipeline([('scale', StandardScaler()), ('clf', clf)])

        # then do grid search on hyperparam, k and weights
        param_grid = {'clf__weights': ['uniform', 'distance'],
                      'clf__n_neighbors': np.arange(1, 7 + 1, 2)}

        gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring=metric, verbose=0)
        gs = gs.fit(X_tr_new, y_tr)  # use subset of features
        if print_GSearch_details:
            for i in range(len(gs.cv_results_['params'])):
                print(gs.cv_results_['params'][i], 'test acc.:', gs.cv_results_['mean_test_score'][i])
            print(f'Best parameters via GridSearch {gs.best_params_}, best score: {gs.best_score_:.3}')

        # use selected hyperparam to fit (sfs) and predict
        clf = gs.best_estimator_['clf']

        sfs = SFS(clf, k_features='best', forward=True, floating=True, scoring=metric, cv=5, n_jobs=n_jobs)

        if OverSampling:  # if oversampling, use SMOTE in Pipeline
            pipe = make_pipeline(StandardScaler(), SMOTE(random_state=seed, n_jobs=n_jobs), sfs)
            pipe_model = make_pipeline(StandardScaler(), SMOTE(random_state=seed, n_jobs=n_jobs), clf)
        else:
            pipe = make_pipeline(StandardScaler(), sfs)
            pipe_model = make_pipeline(StandardScaler(), clf)

        pipe.fit(X_tr, y_tr)
        if show_sfs_var_selection:
            print(f'best combination (AUC by 5-fold CV: {sfs.k_score_:.3}), {len(sfs.k_feature_idx_)} '
                  f'out of {X_tr.shape[1]} features selected: {sfs.k_feature_idx_}')
        # predict using selected features
        X_tr_sfs = sfs.transform(X_tr)
        X_test_sfs = sfs.transform(X_test)
        model = pipe_model.fit(X_tr_sfs, y_tr)

        auc = print_store_results(method_name='KNN', class_report=False)

        if plot_roc_combine and report_auroc:
            fpr, tpr, deci_thres = roc_curve(y_test, model.predict_proba(X_test_sfs)[:, 1])
            plt.plot(fpr, tpr, label=f'KNN, AUC={auc:.3}', linewidth=curve_lw, marker='.', markersize=curve_ms)

            # locate and plot (if plot_optimal_thres=true) best decision threshold, on ROC curve
            locate_plot_optimal_thres(apply_label=False)

        elif plot_roc_combine and report_auroc == False:
            precision, recall, deci_thres = precision_recall_curve(y_test, model.predict_proba(X_test_sfs)[:, 1])
            plt.plot(recall, precision, linewidth=curve_lw, marker='.', markersize=curve_ms, label=f'KNN'
                     #             f', AUC={auc:.3}',
                     )

    if 'SVM' in method:
        print(f'|Training of SVM starts......|')
        t1 = time.perf_counter()
        # first, get rid of some features, pre_filter_features, shorten GridSearchCV time
        clf = SVC(random_state=seed, cache_size=7000, class_weight='balanced')
        if filter_before_GridSearch:
            X_tr_new = pre_filter_features(k_features=k_range, forward=False, floating=False)
        else:
            X_tr_new = X_tr

        if OverSampling:  # if oversampling, use SMOTE in Pipeline
            pipe = Pipeline(
                [('scale', StandardScaler()), ('SMOTE', SMOTE(random_state=seed)), ('clf', clf)])
        else:
            pipe = Pipeline([('scale', StandardScaler()), ('clf', clf)])

        # then do grid search on hyperparam, C, kernel and gamma (for RBF)
        param_grid = [
            {'clf__C': np.logspace(-2, 2, 5, base=2),  # 正则化系数，正则化强度的倒数，正数，值越小，正则化强度越大，即防止过拟合的程度更大
             'clf__kernel': ['rbf'],
             'clf__gamma': np.logspace(-3, 2, 6, base=2),  # gamma默认值1/样本特征数
             },
            {'clf__C': np.logspace(-2, 2, 5, base=2),
             'clf__kernel': ['linear'],
             }
        ]
        gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring=metric, n_jobs=n_jobs, cv=5, verbose=0)
        gs = gs.fit(X_tr_new, y_tr)  # use subset of features
        if print_GSearch_details:
            for i in range(len(gs.cv_results_['params'])):
                print(gs.cv_results_['params'][i], 'test acc.:', gs.cv_results_['mean_test_score'][i])
            print(f'Best parameters via GridSearch {gs.best_params_}, best score: {gs.best_score_:.3}')

        clf = gs.best_estimator_['clf']

        try:
            clf_prob = SVC(C=gs.best_params_['clf__C'], kernel=gs.best_params_['clf__kernel'],
                           gamma=gs.best_params_['clf__gamma'], class_weight='balanced', random_state=seed,
                           probability=True, cache_size=7000)
        except KeyError:  # maybe the linear kernel is selected and gamma does not exist
            clf_prob = SVC(C=gs.best_params_['clf__C'], kernel=gs.best_params_['clf__kernel'],
                           class_weight='balanced', random_state=seed,
                           probability=True, cache_size=7000)

        sfs = SFS(clf, k_features=k_range, forward=False, floating=True, scoring=metric, cv=5,
                  n_jobs=n_jobs)  # backward

        if OverSampling:  # if oversampling, use SMOTE in Pipeline
            pipe = make_pipeline(StandardScaler(), SMOTE(random_state=seed, n_jobs=n_jobs), sfs)
            pipe_model = make_pipeline(StandardScaler(), SMOTE(random_state=seed, n_jobs=n_jobs), clf_prob)
        else:
            pipe = make_pipeline(StandardScaler(), sfs)
            pipe_model = make_pipeline(StandardScaler(), clf_prob)

        pipe.fit(X_tr, y_tr)
        if show_sfs_var_selection:
            print(f'best combination (AUC by 5-fold CV: {sfs.k_score_:.3}), {len(sfs.k_feature_idx_)} '
                  f'out of {X_tr.shape[1]} features selected: {sfs.k_feature_idx_}')
        # predict using selected features
        X_tr_sfs = sfs.transform(X_tr)
        X_test_sfs = sfs.transform(X_test)
        model = pipe_model.fit(X_tr_sfs, y_tr)

        auc = print_store_results(method_name='SVM', class_report=False)

        if plot_roc_combine and report_auroc:
            fpr, tpr, deci_thres = roc_curve(y_test, model.predict_proba(X_test_sfs)[:, 1])
            plt.plot(fpr, tpr, label=f'SVM, AUC={auc:.3}', linewidth=curve_lw, marker='.', markersize=curve_ms)

            # locate and plot (if plot_optimal_thres=true) best decision threshold, on ROC curve
            locate_plot_optimal_thres(apply_label=False)

        elif plot_roc_combine and report_auroc == False:
            precision, recall, deci_thres = precision_recall_curve(y_test, model.predict_proba(X_test_sfs)[:, 1])
            plt.plot(recall, precision, linewidth=curve_lw, marker='.', markersize=curve_ms, label=f'SVM'
                     #             f', AUC={auc:.3}',
                     )

    ## Ensemble methods begin
    if not use_1hot_trees:  # not using 1 hot for trees method, then assign X_tr_es and X_test_es to X_tr and X_test
        X_tr = X_tr_es
        X_test = X_test_es
        # Convert to tuple for k_features_search_range. 再次执行update k_range
        k_range = tuple(X_tr.shape[1] + np.asarray(k_features_search_range))

    if 'RF' in method:
        print(f'|Training of RandomForestClassifier starts......|')
        t1 = time.perf_counter()
        clf = RandomForestClassifier(random_state=seed, class_weight='balanced', n_jobs=n_jobs, oob_score=True)
        # first, get rid of some features, pre_filter_features, shorten GridSearchCV time
        if filter_before_GridSearch:
            X_tr_new = pre_filter_features(k_features=k_range, forward=False, floating=False)
        else:
            X_tr_new = X_tr

        if OverSampling:  # if oversampling, use SMOTE in Pipeline
            pipe = Pipeline([('SMOTE', SMOTE(random_state=seed, n_jobs=n_jobs)), ('clf', clf)])
        else:
            pipe = Pipeline([('clf', clf)])

        # then do grid search on hyperparam
        param1 = {'clf__n_estimators': range(80, 410, 20)}  # 1. 先定No. of trees
        param2 = {'clf__max_depth': range(8, 31 + 1, 2),
                  'clf__min_samples_split': range(2, 51, 5)}  # 2. 对决策树最大深度和内部节点再划分所需最小样本数进行搜索
        param3 = {'clf__min_samples_split': range(2, 51, 5),
                  'clf__min_samples_leaf': range(1, 50, 5)}  # 3. 再对内部节点再划分所需最小样本数和叶子节点最少样本数min_samples_leaf一起调参
        max_features_deft_pct = np.sqrt(X_tr_new.shape[1]) / X_tr_new.shape[1]  # max_features_default = sqrt(p)
        # 4. 最后调参分裂时参与判断的最大特征数, 默认百分比为sqrt(p). 浮点数，代表考虑特征百分比，即考虑（百分比xp）取整后的特征数, p为总特征数
        param4 = {'clf__max_features': np.arange(max_features_deft_pct, 1.0, .08)}  # pct increase by 8%

        pipe = Step_GSearch(param_grid=param1, X=X_tr_new, pipe=pipe)  # update the n_estimators
        pipe = Step_GSearch(param_grid=param2, X=X_tr_new, pipe=pipe)  # update the max_depth and min_samples_split
        pipe = Step_GSearch(param_grid=param3, X=X_tr_new, pipe=pipe)
        pipe = Step_GSearch(param_grid=param4, X=X_tr_new, pipe=pipe)
        # ----------------------------------------------------------------#
        sfs = SFS(pipe, k_features=k_range, forward=False, floating=False, scoring=metric, cv=5,
                  n_jobs=n_jobs)  # use backward. 如果用Forward, 之前GSearch选定的max_features会报错
        sfs.fit(X_tr, y_tr)
        if show_sfs_var_selection:
            print(f'best combination (AUC by 5-fold CV: {sfs.k_score_:.3}), {len(sfs.k_feature_idx_)} '
                  f'out of {X_tr.shape[1]} features selected: {sfs.k_feature_idx_}')
        # predict using selected features
        X_tr_sfs = sfs.transform(X_tr)
        X_test_sfs = sfs.transform(X_test)
        model = pipe.fit(X_tr_sfs, y_tr)

        auc = print_store_results(method_name='RF', class_report=False)

        if plot_roc_combine and report_auroc:
            fpr, tpr, deci_thres = roc_curve(y_test, model.predict_proba(X_test_sfs)[:, 1])
            plt.plot(fpr, tpr, label=f'RF, AUC={auc:.3}', linewidth=curve_lw, marker='.', markersize=curve_ms)

            # locate and plot (if plot_optimal_thres=true) best decision threshold, on ROC curve
            locate_plot_optimal_thres(apply_label=False)

        elif plot_roc_combine and report_auroc == False:
            precision, recall, deci_thres = precision_recall_curve(y_test, model.predict_proba(X_test_sfs)[:, 1])
            plt.plot(recall, precision, linewidth=curve_lw, marker='.', markersize=curve_ms, label=f'RF'
                     #             f', AUC={auc:.3}',
                     )

    if 'GB' in method:
        print(f'|Training of GradientBoosting starts......|')
        t1 = time.perf_counter()
        # 1. 先定No. of trees, 固定步长为0.1
        param1 = {'clf__n_estimators': range(50, 201, 10)}
        # 2. 对决策树最大深度和内部节点再划分所需最小样本数进行搜索
        param2 = {'clf__max_depth': range(3, 15 + 1, 2), 'clf__min_samples_split': range(2, 42, 5)}
        # 3. 再对内部节点再划分所需最小样本数和叶子节点最少样本数一起调参
        param3 = {'clf__min_samples_split': range(2, 42, 5), 'clf__min_samples_leaf': range(1, 41, 5)}
        # 4. 再用上面调好的参数对no. of trees和步长调参
        param4 = {'clf__n_estimators': range(20, 181, 10),  # tune # of trees again.
                  # 'clf__learning_rate': np.logspace(-2, -1, 2, base=10) # give up tuning training rate since too slow
                  }

        clf = GradientBoostingClassifier(random_state=seed, learning_rate=.1, subsample=0.8)
        pipe = Pipeline([('clf', clf)])

        pipe = Step_GSearch(param_grid=param1, X=X_tr, pipe=pipe)  # update the n_estimators
        pipe = Step_GSearch(param_grid=param2, X=X_tr, pipe=pipe)  # update the max_depth and min_samples_split
        pipe = Step_GSearch(param_grid=param3, X=X_tr, pipe=pipe)
        pipe = Step_GSearch(param_grid=param4, X=X_tr, pipe=pipe)

        # ----------------------------------------------------------------#
        sfs = SFS(pipe, k_features=k_range, forward=False, floating=False, scoring=metric, cv=5,
                  n_jobs=n_jobs)  # Use backward and cv=3 to save time.
        sfs.fit(X_tr, y_tr)
        if show_sfs_var_selection:
            print(f'best combination (AUC by 5-fold CV: {sfs.k_score_:.3}), {len(sfs.k_feature_idx_)} '
                  f'out of {X_tr.shape[1]} features selected: {sfs.k_feature_idx_}')
        # predict using selected features
        X_tr_sfs = sfs.transform(X_tr)
        X_test_sfs = sfs.transform(X_test)
        model = pipe.fit(X_tr_sfs, y_tr)

        auc = print_store_results(method_name='GBDT', class_report=False)

        if plot_roc_combine and report_auroc:
            fpr, tpr, deci_thres = roc_curve(y_test, model.predict_proba(X_test_sfs)[:, 1])
            plt.plot(fpr, tpr, label=f'GBDT, AUC={auc:.3}', linewidth=curve_lw, marker='.',
                     markersize=curve_ms)

            # locate and plot (if plot_optimal_thres=true) best decision threshold, on ROC curve
            locate_plot_optimal_thres(apply_label=False)

        elif plot_roc_combine and report_auroc == False:
            precision, recall, deci_thres = precision_recall_curve(y_test, model.predict_proba(X_test_sfs)[:, 1])
            plt.plot(recall, precision, linewidth=curve_lw, marker='.', markersize=curve_ms, label=f'GBDT'
                     #             f', AUC={auc:.3}',
                     )

    if 'XGB' in method:
        print(f'|Training of XGBoost starts......|')
        t1 = time.perf_counter()
        # 1. 先定No. of trees, 固定learning rate为0.1
        param1 = {'clf__n_estimators': range(50, 301, 10)}
        # 2. 对决策树最大深度和内部节点再划分所需最小样本数进行搜索
        param2 = {'clf__max_depth': range(3, 10 + 1, 2),
                  'clf__min_child_weight': range(1, 6, 2)}
        # 3. 对gamma调优
        param3 = {'clf__gamma': [i / 10.0 for i in range(0, 5)], }
        # 4. subsample 和 colsample_bytree
        param4 = {'clf__subsample': [i / 10.0 for i in range(6, 10)],
                  'clf__colsample_bytree': [i / 10.0 for i in range(6, 10)], }
        # 5. 正则化参数调优
        param5 = {'clf__reg_alpha': [0, 0.001, 0.005, 0.01], }
        # 4. 最后降低学习率，确定理想参数. 这个要跟n_estimators一起调
        param6 = {'clf__learning_rate': [.05, .1, .2],
                  'clf__n_estimators': range(50, 301, 10), }

        clf = XGBClassifier(max_depth=5, learning_rate=0.1, verbosity=1, objective='binary:logistic', random_state=seed,
                            min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=pos_neg_rate)  # scale_pos_weight用于调整balance, 0值的样本数量/1值的样本数量
        pipe = Pipeline([('clf', clf)])

        pipe = Step_GSearch(param_grid=param1, X=X_tr, pipe=pipe)
        pipe = Step_GSearch(param_grid=param2, X=X_tr, pipe=pipe)
        pipe = Step_GSearch(param_grid=param3, X=X_tr, pipe=pipe)
        pipe = Step_GSearch(param_grid=param4, X=X_tr, pipe=pipe)
        pipe = Step_GSearch(param_grid=param5, X=X_tr, pipe=pipe)
        pipe = Step_GSearch(param_grid=param6, X=X_tr, pipe=pipe)

        # ----------------------------------------------------------------#
        sfs = SFS(pipe, k_features=k_range, forward=False, floating=False, scoring=metric, cv=5,
                  n_jobs=n_jobs)  # Use backward and cv=3 to save time.
        sfs.fit(X_tr, y_tr)
        if show_sfs_var_selection:
            print(f'best combination (AUC by 5-fold CV: {sfs.k_score_:.3}), {len(sfs.k_feature_idx_)} '
                  f'out of {X_tr.shape[1]} features selected: {sfs.k_feature_idx_}')
        # predict using selected features
        X_tr_sfs = sfs.transform(X_tr)
        X_test_sfs = sfs.transform(X_test)
        model = pipe.fit(X_tr_sfs, y_tr)

        auc = print_store_results(method_name='XGBoost', class_report=False)

        if plot_roc_combine and report_auroc:
            fpr, tpr, deci_thres = roc_curve(y_test, model.predict_proba(X_test_sfs)[:, 1])
            plt.plot(fpr, tpr, label=f'XGBoost, AUC={auc:.3}', linewidth=curve_lw, marker='.', markersize=curve_ms,
                     color='C6')

            # locate and plot (if plot_optimal_thres=true) best decision threshold, on ROC curve
            locate_plot_optimal_thres(apply_label=True)  # apply best thres label on XGBoost

            if show_thres_XGBoost:  # whether to show thres indicator (dashline, text, etc.) for XGBoost
                # -0.02 and +0.02 is for minor adjustments
                plt.axvline(x=fpr[idx], ymax=tpr[idx] - 0.02, color='black', linestyle=':', linewidth=2)
                plt.axhline(y=tpr[idx], xmax=fpr[idx] + 0.02, color='black', linestyle=':', linewidth=2)
                plt.text(0, tpr[idx] + .04, f'TPR={tpr[idx]:.1%}', horizontalalignment='left',
                         verticalalignment='center', fontsize=12, fontweight='bold')  # .04 is to adjust position
                plt.text(fpr[idx] + .025, 0, f'FPR={fpr[idx]:.1%}', horizontalalignment='left',
                         verticalalignment='center', fontsize=12, fontweight='bold')  # .025 is to adjust position

        elif plot_roc_combine and report_auroc == False:
            fpr, tpr, deci_thres = roc_curve(y_test, model.predict_proba(X_test_sfs)[:, 1])
            precision, recall, deci_thres = precision_recall_curve(y_test, model.predict_proba(X_test_sfs)[:, 1])
            plt.plot(recall, precision, linewidth=curve_lw, marker='.', markersize=curve_ms, label=f'XGBoost',
                     color='C6',
                     #             f', AUC={auc:.3}',
                     )

            if show_thres_XGBoost:  # whether to show thres indicator (dashline, text, etc.) for XGBoost on PR curve
                # return idx of best decision threshold on ROC curve
                idx = argmax(tpr - fpr)

                # Retrieve index of recall where recall = fpr[idx], Manually apply best thres label on XGBoost, for PR curve
                recall_idx = np.where(recall == tpr[idx])
                best_recall = recall[recall_idx[0][0]]
                best_precision = precision[recall_idx[0][0]]
                plt.scatter(best_recall, best_precision, marker='o', color='black', s=thres_ms,
                            label='Optimal Threshold')

                # need minor adjustments since length of y axis is not 1
                plt.axvline(x=best_recall, ymax=best_precision - 0.1, color='black', linestyle=':', linewidth=2)
                plt.axhline(y=best_precision, xmax=best_recall - 0.018, color='black', linestyle=':', linewidth=2)
                plt.text(0, best_precision + .03, f'Precision={best_precision:.1%}', horizontalalignment='left',
                         verticalalignment='center', fontsize=12, fontweight='bold')  # .04 is to adjust position
                plt.text(best_recall - .02, 0.19, f'TPR={best_recall:.1%}', horizontalalignment='right',
                         verticalalignment='center', fontsize=12, fontweight='bold')  # .025 is to adjust position

    # ---------------------------------------Plot-------------------------------------- #
    if plot_roc:
        if plot_roc_combine and report_auroc:  # if plot ROC
            plt.legend(loc='lower right', prop={'size': 11.5})
            plt.xlabel("False Positive Rate", fontsize=12)
            plt.ylabel("True Positive Rate", fontsize=12)
            plt.plot([0, 1], [0, 1], color='lightgrey', linestyle='--')  # add dashed line for dumb classifier
            if plot_title:
                plt.title(f'Year={year} {k} ROC curve, Boundary=\u00B1{boundary}°F'
                          # f', OverSampling={OverSampling}'
                          )
            if plot_save:
                plt.savefig(
                    f'graphs/Year={year} max({metric}) use_DST={use_DST} {k} ROC curve, Boundary=\u00B1{boundary}°F, Deci_Thres={plot_optimal_thres}'
                    # f', OverSampling={OverSampling}'
                    f'.png',
                    bbox_inches='tight')
        elif plot_roc_combine and report_auroc == False:  # if plot PR curve
            plt.legend(loc='upper right', prop={'size': 11.5})
            plt.xlabel('True Positive Rate', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            no_skill = np.sum(y_test == 1) / len(y_test)
            plt.plot([0, 1], [no_skill, no_skill], color='lightgrey', linestyle='--')  # dashed line for noskill
            if plot_title:
                plt.title(f'Year={year} {k} PR curve, Boundary=\u00B1{boundary}°F'
                          # f', OverSampling={OverSampling}'
                          )
            if plot_save:
                plt.savefig(
                    f'graphs/Year={year} max({metric}) use_DST={use_DST} {k} PR curve, Boundary=\u00B1{boundary}°F, Deci_Thres={plot_optimal_thres}'
                    # f', OverSampling={OverSampling}'
                    f'.png',
                    bbox_inches='tight')
        else:
            plot_roc_curve(clf, X_test_sfs, y_test)  # plot ROC curve from single classifier

        plt.show()

    if plot_confusion:
        y_pred = (model.predict_proba(X_test_sfs)[:, 1] > alpha).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=200)

        sns.set_style('ticks')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(cmap='Blues', values_format='.0f', ax=ax)
        plt.show()

    if plot_feature_selection:
        fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
        plt.title('Sequential Forward Selection (w. StdErr)')
        plt.grid()
        plt.show()

    # plot correlation matrix
    if plot_corr_matrix:
        corr = X.iloc[:, 1:].corr()  # remove intercept from the analysis
        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True,
                         annot=True, fmt='.2f', annot_kws={"fontsize": 6})
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        fig.tight_layout()
        plt.savefig(f'graphs/Corr Matrix, Station={k}.png', bbox_inches='tight')
        plt.show()

    t_end = time.perf_counter()
    print(f'|-------------Execution finished in {round((t_end - t_start) / 60, 1)} minute(s)-------------|')

if result_df_save:
    result_df.to_excel(f'Year={year} sunshine_var={use_sunshine} max({metric}) use_DST={use_DST} Modeling Results.xlsx',
                       index=False)
