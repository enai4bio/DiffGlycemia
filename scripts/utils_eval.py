import argparse

import os
from copy import deepcopy
from glob import glob
import shutil

import lib
from utils_train import *
import ast


import numpy as np
import pandas as pd
pd.set_option('display.min_rows', 10)
pd.set_option('display.max_columns', 20)

import matplotlib.pyplot as plt

import math
import zero
import torch

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE

from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from sklearn.ensemble import RandomForestClassifier


def create_real_dataset(split_set,real_data_dir,feature_columns,label_column):
    num_path_ = f'{real_data_dir}/X_num_{split_set}.csv'
    cat_path_ = f'{real_data_dir}/X_cat_{split_set}.csv'
    y_path_ = f'{real_data_dir}/y_{split_set}.csv'
    X_num_ = pd.read_csv(num_path_,index_col=0)
    X_cat_ = pd.read_csv(cat_path_,index_col=0)
    y_ = pd.read_csv(y_path_,index_col=0)
    mapping = {'A': 0, 'B': 1}
    X_cat_01_ = X_cat_.apply(lambda x: x.map(mapping))
    X_ = pd.concat([X_num_,X_cat_01_],axis=1)
    X_columned_ = X_[feature_columns].copy()
    return X_columned_, y_[label_column]



def create_normalized_dataset(D, split_set,feature_columns,categorical_feature_columns,label_column):
    X_normalized_ = pd.DataFrame(
        np.concatenate([D.X_num[split_set], D.X_cat[split_set], ],axis=1),
        columns=feature_columns
    )
    X_normalized_[categorical_feature_columns] = X_normalized_[categorical_feature_columns].astype(int)
    X_normalized_ = X_normalized_[feature_columns].copy()
    y_normalized_ = pd.DataFrame(
        D.y[split_set],
        columns=[label_column]
    )
    return X_normalized_.copy(), y_normalized_[label_column].copy()



def read_generated_data(generated_dir_, feature_columns, label_column, verbose=False):
    X_generated_train = pd.read_csv(f"{generated_dir_}/X_.csv")
    y_generated_train = pd.read_csv(f"{generated_dir_}/y_.csv")[label_column]
    mapping = {'A': 0, 'B': 1}
    X_generated_train = X_generated_train.apply(lambda x: x.map(mapping) if x.dtype == 'object' else x)
    X_generated_train = X_generated_train[feature_columns].copy()
    if verbose:
        print('X_generated_train.shape:',X_generated_train.shape)
    return X_generated_train, y_generated_train



def balance_dataset(X_, y_, verbose=False):
    y_ = y_.astype(int)
    if verbose:
        print('unbalance rate: 1/(0+1) = %f' % (sum(y_)/len(y_)))
    ros = RandomOverSampler()
    X_train_ros, y_train_ros = ros.fit_resample(X_, y_)
    sm = SMOTE()
    X_train_smote, y_train_smote = sm.fit_resample(X_, y_)
    svnsm = SVMSMOTE()
    X_train_svnsm, y_train_svnsm = svnsm.fit_resample(X_, y_)
    return X_train_ros, y_train_ros, X_train_smote, y_train_smote, X_train_svnsm, y_train_svnsm



def run_rondom_forest(X_,y_,raw_config):
    shuffle_switch = ast.literal_eval(raw_config['eval']['main']['shuffle_switch'])
    seed = raw_config['eval']['main']['seed']
    if shuffle_switch:
        X_, y_ = shuffle(X_, y_, random_state=seed)
    clf_rf = RandomForestClassifier(
                n_estimators=300,
                random_state=seed,
                class_weight='balanced',
                n_jobs=-1,
                verbose=0)
    hist_ = clf_rf.fit(X_, y_)
    return clf_rf



def eval_rondom_forest(clf_rf, X_eval_rf_, y_eval_rf_, eval_type):

    y_pred = clf_rf.predict(X_eval_rf_)
    y_pred_p_real = clf_rf.predict_proba(X_eval_rf_)[:,1]
    
    df_cm = pd.DataFrame(
        confusion_matrix(y_eval_rf_, y_pred), 
        index=['Actual 0', 'Actual 1'], 
        columns=['Predicted 0', 'Predicted 1']
        )
    
    recall_score_value = recall_score(np.array(y_eval_rf_), np.array(y_pred))
    roc_auc_score_value = roc_auc_score(np.array(y_eval_rf_), np.array(y_pred_p_real))
    average_precision_score_value = average_precision_score(np.array(y_eval_rf_), np.array(y_pred_p_real))

    return df_cm, recall_score_value, roc_auc_score_value, average_precision_score_value




