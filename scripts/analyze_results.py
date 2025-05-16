#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on %(date)s @author: %(username)s

#%% const

# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_01'
# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_10_4/ep1500_ts1000_128x3_generate10_sample4_9x9'
# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_10_4/ep1500_ts1000_128x3_generate10_sample4_9x9_ma'
# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_10_4/ep3000_ts1000_128x3_generate10_sample4_9x9_ma'
# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_10_4/ep1500_ts1000_256x3_generate10_sample4_9x9'

# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_sample/ep1500_ts1000_128x3_generate10_sample1_6x6'
# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_sample/ep1500_ts1000_128x3_generate10_sample2_6x6'
# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_sample/ep1500_ts1000_128x3_generate10_sample3_6x6'
# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_sample/ep1500_ts1000_128x3_generate10_sample4_9x9'
# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_sample/ep1500_ts1000_128x3_generate10_sample5_6x6'
# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_sample/ep1500_ts1000_128x3_generate10_sample6_6x6'
# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_sample/ep1500_ts1000_128x3_generate10_sample7_6x6'
# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_sample/ep1500_ts1000_128x3_generate10_sample8_6x6'
# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_sample/ep1500_ts1000_128x3_generate10_sample9_6x6'
# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_sample/ep1500_ts1000_128x3_generate10_sample10_6x6'


# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_generate/ep1500_ts1000_128x3_generate20_sample8_6x6'
wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_generate/ep1500_ts1000_128x3_generate30_sample12_6x6'
# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_generate/ep1500_ts1000_128x3_generate40_sample16_6x6'
# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_generate/ep1500_ts1000_128x3_generate50_sample20_6x6'

# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_sample_generate30/ep1500_ts1000_128x3_generate30_sample3_6x6'
# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_sample_generate30/ep1500_ts1000_128x3_generate30_sample6_6x6'
# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_sample_generate30/ep1500_ts1000_128x3_generate30_sample9_6x6'
# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_sample_generate30/ep1500_ts1000_128x3_generate30_sample16_6x6'

# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_10_4/ep1500_ts800_128x3_generate10_sample4_6x6/'
# wd = u'/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/exp_10_4/ep1500_ts1200_128x3_generate10_sample4_6x6/'

sd = 42

n_row = 6
n_col = 20


#%% lib & func

from IPython.display import display
# =============================================================================
# sys

import sys

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import time

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

# =============================================================================

import os
cwd = os.getcwd()
print("current:", cwd)
os.chdir(wd)
print('defined:',wd)

def my_makedir(customized_path, f=False):
    if (os.path.exists(customized_path)) and (f == False):
        print("\nFolder '{}' exists.".format(customized_path))
    else:
        os.makedirs(customized_path)
        print("\nFolder '{}' created.".format(customized_path))
    print('\n' + '-'*60)

def my_create_file(file_path):
    if os.path.isfile(file_path):
        os.system("rm -rf %s" % file_path)
    print('remove_create: %s' % file_path)

# import csv,gzip,h5py,lmdb,pickle,shutil
import csv,gzip,joblib,pickle,shutil

# =============================================================================
# numpy, pandas, etc

import numpy as np
np.random.seed(sd)

import pandas as pd
pd.set_option('display.max_columns', n_col)
pd.set_option('display.min_rows', n_row)

def cut_value_counts(data,bins,verbose=False):
    data_binned = pd.cut(data, bins=bins)
    counts = data_binned.value_counts()
    counts = counts.reset_index().sort_values(by=counts.reset_index().columns[0])
    if verbose:
        print(counts)
    return counts.sort_values

# import datetime
from datetime import datetime

# =============================================================================
# plot

import matplotlib
from matplotlib import colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt


import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

# from mpl_toolkits.mplot3d import Axes3D
# from yellowbrick.cluster import KElbowVisualizer


# from lifelines import KaplanMeierFitter
# from lifelines.statistics import logrank_test

# =============================================================================
# scipy

import scipy

from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, kendalltau, pearsonr, spearmanr

def my_t_test(data1, data2):
    t_statistic, p_value = stats.ttest_ind(data1, data2)
    return p_value

def my_u_test(data1, data2):
    t_statistic, p_value = stats.mannwhitneyu(data1, data2)
    return p_value

def my_k_test(data1, data2):
    correlation, p_value = stats.kendalltau(data1, data2)
    # print("Kendall 秩相关系数:", correlation)
    # print("p 值:", p_value)
    return p_value

# =============================================================================
# sklearn

from imblearn.over_sampling import RandomOverSampler, SMOTE, SVMSMOTE

from sklearn.cluster import AgglomerativeClustering,KMeans
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.metrics import accuracy_score,average_precision_score,confusion_matrix,f1_score,recall_score,roc_auc_score,precision_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder,StandardScaler


# =============================================================================
# torch

import torch

# =============================================================================
# networkx

# import networkx as nx

# ============================================
# rdkit
# from rdkit import Chem
# from rdkit.Chem import AllChem,Draw,PandasTools
# from rdkit.Chem.rdMolAlign import GetBestAlignmentTransform



#%% load data

test_trial_boundary = 6
test_sample_boundary = 6

df_all = pd.read_csv('metrics_all.csv')

df_all = df_all.loc[
    (df_all['trial'] < test_trial_boundary) & (df_all['sample'] < test_sample_boundary)
    ,:
]

df_all.loc[df_all['train_type']=='real_ros','train_type'] = 'real_oversample'
df_all.loc[df_all['train_type']=='random','train_type'] = 'diffusion'

metric_columns = ['test_accuracy', 'test_precision', 'test_recall', 'test_micro_f1', 'test_macro_f1', 'test_roc_auc', 'test_pr_auc']

analyze_result_folder = f'analyze_results_trial_{test_trial_boundary}_sample_{test_sample_boundary}'
os.makedirs(analyze_result_folder,exist_ok=True)


df_all.sort_values(by=['train_type','trial','sample'], inplace=True)
df_all

#%% average matrix

def split_matrix(df, cm_col):
    df = df.copy()
    matrix_columns = ['test_cm_0', 'test_cm_1', 'test_cm_2', 'test_cm_3']
    df[matrix_columns[0]] = df[cm_col].apply(lambda x: int(x[1:-1].split(', ')[0]))
    df[matrix_columns[1]] = df[cm_col].apply(lambda x: int(x[1:-1].split(', ')[1]))
    df[matrix_columns[2]] = df[cm_col].apply(lambda x: int(x[1:-1].split(', ')[2]))
    df[matrix_columns[3]] = df[cm_col].apply(lambda x: int(x[1:-1].split(', ')[3]))
    return df.copy(), matrix_columns

df_all_matrix, matrix_columns = split_matrix(df_all, 'test_cm')

df_all_mean = df_all_matrix.groupby('train_type')[matrix_columns].mean()
df_all_mean.to_csv(f'{analyze_result_folder}/confusion_matrix_mean.csv')

print(df_all_mean)

#%% metric

df_metric = df_all[['trial', 'sample', 'train_type']+metric_columns]

df_metric_mean = df_metric.groupby('train_type')[metric_columns].mean()
df_metric_std = df_metric.groupby('train_type')[metric_columns].std()


df_metric_summary = pd.DataFrame(columns=df_metric_mean.columns,index=df_metric_mean.index)

for row_ in df_metric_summary.index:
    for col_ in df_metric_summary.columns:
        
        mean_ = str( "{:.3f}".format(np.round(float(df_metric_mean.loc[row_,col_]),3)))
        std_ = str( "{:.3f}".format(np.round(float(df_metric_std.loc[row_,col_]),3)))
        
        df_metric_summary.loc[row_,col_] = mean_ + '±' + std_

df_metric_summary.to_csv(f'{analyze_result_folder}/metric_summary.csv')

print(df_metric_summary)

#%% t-test


def test_metric(df_metric, metric_, group_0, group_1):
    data_0 = df_metric.loc[df_metric['train_type']==group_0,metric_].copy()
    data_1 = df_metric.loc[df_metric['train_type']==group_1,metric_].copy()
    p_value = my_t_test(data_0, data_1)
    return p_value

print('recall')
print(test_metric(df_metric, 'test_recall', 'diffusion', 'real_smote'))
# 7.624904409241656e-26
print('roc_auc')
print(test_metric(df_metric, 'test_roc_auc', 'diffusion', 'real_oversample'))
# 2.1011550739585273e-06
print('pr_auc')
print(test_metric(df_metric, 'test_pr_auc', 'diffusion', 'real_oversample'))
# 0.6637351114371679

#%% plot test_recall

y_col = 'test_recall'

plt.figure(figsize=(12, 8))
sns.boxplot(data=df_metric, x='train_type', y=y_col)
plt.title(f'Boxplot of {y_col} by Train Type')
plt.savefig(f'{analyze_result_folder}/{y_col}.png', dpi=300)
# plt.show()
plt.close()

#%% plot test_roc_auc

y_col = 'test_roc_auc'

plt.figure(figsize=(12, 8))
sns.boxplot(data=df_metric, x='train_type', y=y_col)
plt.title(f'Boxplot of {y_col} by Train Type')
plt.savefig(f'{analyze_result_folder}/{y_col}.png', dpi=300)
# plt.show()
plt.close()

#%% plot test_pr_auc

y_col = 'test_pr_auc'

plt.figure(figsize=(12, 8))
sns.boxplot(data=df_metric, x='train_type', y=y_col)
plt.title(f'Boxplot of {y_col} by Train Type')
plt.savefig(f'{analyze_result_folder}/{y_col}.png', dpi=300)
# plt.show()
plt.close()

#%% plot all

# plot_metric_columns = ['test_accuracy', 'test_precision', 'test_recall', 'test_micro_f1', 'test_macro_f1', 'test_roc_auc', 'test_pr_auc']

# plt.figure(figsize=(20, 20))

# for i, column in enumerate(plot_metric_columns, 1):
#     plt.subplot(4, 2, i)  # 4行2列的子图布局
#     sns.boxplot(data=df_metric, x='train_type', y=column)
#     plt.title(f'Boxplot of {column} by Train Type')

# # 调整布局
# plt.tight_layout()
# plt.savefig(f'{analyze_result_folder}/metric_boxplot.png', dpi=300)
# plt.show()

#%% end 
