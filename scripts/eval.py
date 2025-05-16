import argparse

# import subprocess
# import optuna
# import time
# import tqdm

import os
from copy import deepcopy
from glob import glob
# from pathlib import Path
# import pickle
import shutil

import lib
# from utils_train import *
from utils_eval import *
import ast

# from datetime import datetime

import numpy as np
import pandas as pd
pd.set_option('display.min_rows', 10)
pd.set_option('display.max_columns', 20)

# import seaborn as sns
import matplotlib.pyplot as plt

import math
import zero
import torch

# ------------------------------------------------------------------------
# imblearn

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE

# ------------------------------------------------------------------------
# sklearn

from sklearn.utils import shuffle

# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from sklearn.ensemble import RandomForestClassifier


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------


def eval_rf(n_sample_, n_epoch_, raw_config):

    seed = int(n_sample_.split('_')[-1])
    zero.improve_reproducibility(seed) 
    print('seed:',seed)
    raw_config['eval']['main']['seed'] = seed

    numerical_feature_columns = raw_config['data']['x']['numerical_feature_columns']
    categorical_feature_columns = raw_config['data']['x']['categorical_feature_columns']    
    feature_columns = numerical_feature_columns + categorical_feature_columns
    label_column = raw_config['data']['y']['label_column']  

    # ------ real data ------

    real_data_dir = raw_config['data']['real_data_dir']
    X_real_train, y_real_train = create_real_dataset('train',real_data_dir,feature_columns,label_column)
    X_real_val, y_real_val = create_real_dataset('val',real_data_dir,feature_columns,label_column)
    X_real_test, y_real_test = create_real_dataset('test',real_data_dir,feature_columns,label_column)

    X_real_train = pd.concat([X_real_train, X_real_val], axis=0)
    y_real_train = pd.concat([y_real_train, y_real_val], axis=0)

    X_real_train_ros, y_real_train_ros, X_real_train_smote, y_real_train_smote, X_real_train_svnsm, y_real_train_svnsm = balance_dataset(X_real_train, y_real_train)
    
    # ------ sampled data ------

    generated_dir = raw_config['eval']['main']['epoch_dir']
    X_random_train, y_random_train = read_generated_data(generated_dir, feature_columns, label_column)

    # ------ run random forest ------

    all_sets = [
        [X_real_train_ros,      y_real_train_ros,   X_real_test, y_real_test],
        [X_real_train_smote,    y_real_train_smote, X_real_test, y_real_test],
        [X_real_train_svnsm,    y_real_train_svnsm, X_real_test, y_real_test],
        [X_random_train,        y_random_train,     X_real_test, y_real_test],
    ]

    all_set_names = [
        'real_ros',
        'real_smote',
        'real_svnsm',
        'random',
    ]

    eval_dir = raw_config['eval']['main']['eval_dir']
    
    df_metrics = pd.DataFrame(columns=[
        'train_type',
        'test_cm',
        'test_accuracy',
        'test_precision',
        'test_recall',
        'test_micro_f1',
        'test_macro_f1',
        'test_roc_auc',
        'test_pr_auc'
        ])
    
    # for i in range(len(all_sets)):
    for i_, ((X_train_rf_, y_train_rf_, X_test_rf_, y_test_rf_), set_name_) in enumerate(zip(all_sets, all_set_names)):
        
        # i = 0
        print('-'*19,f'{set_name_}','-'*19)
        set_dir_ = f"{eval_dir}/{str(i_)}_{set_name_}"
        os.makedirs(set_dir_,exist_ok=True)
        print('created',set_dir_)
        
        clf_rf = run_rondom_forest(X_train_rf_, y_train_rf_, raw_config)
        # joblib.dump(clf_rf, f'{set_dir_}/random_forest_model.pkl')

        df_cm_test, accuracy_score_test, precision_score_test, recall_score_test, micro_f1_score_test, macro_f1_score_test, roc_auc_score_test, average_precision_score_test = eval_rondom_forest(clf_rf, X_test_rf_, y_test_rf_, 'test')
        
        df_cm_test.to_csv(f'{set_dir_}/confusion_matrix_test.csv')
        print(df_cm_test)
    
        df_metrics.loc[len(df_metrics),:] =[
            set_name_,
            str(df_cm_test.values.flatten().tolist()),
            str(accuracy_score_test),
            str(precision_score_test),
            str(recall_score_test),
            str(micro_f1_score_test),
            str(macro_f1_score_test),
            str(roc_auc_score_test),
            str(average_precision_score_test)
            ]

    df_metrics.to_csv(f'{eval_dir}/metrics.csv',index=False)
    print(df_metrics[['train_type','test_cm','test_recall','test_roc_auc','test_pr_auc']].sort_values(by='test_roc_auc', ascending=False))

    # raw_config_converted = lib.convert_numpy_to_native(raw_config)
    # lib.dump_config(raw_config_converted, f"{raw_config['main']['raw_config_path']}") 
    # lib.dump_config(raw_config_converted, f"{raw_config['train']['main']['trained_model_dir']}/config.toml") 
    # lib.dump_config(raw_config_converted, f"{raw_config['sample']['main']['sample_dir']}/config.toml") 
    # lib.dump_config(raw_config_converted, f"{eval_dir}/config.toml") 

    return df_metrics

