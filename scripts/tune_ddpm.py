# ------------------------------------------------------------------------

import argparse
import os
from glob import glob
import shutil
import lib
from utils_train import *

import numpy as np
import pandas as pd

import torch

from train import train
from sample import sample
from eval import eval_rf as eval

parser = argparse.ArgumentParser()
parser.add_argument('--wd', type=str, default='')
parser.add_argument('--job', type=str, default='')
args = parser.parse_args()

work_dir = args.wd
os.chdir(work_dir)
print('='*70, '\nwork_dir:',work_dir,'\n','='*70)

raw_config_path = f'{work_dir}/exp/config.toml' 
raw_config = lib.load_config(raw_config_path)
print('loaded:', raw_config_path)

raw_config['main']['work_dir'] = work_dir 
raw_config['main']['exp_dir'] = f'{work_dir}/exp' 
raw_config['main']['raw_config_path'] = raw_config_path 
raw_config_converted = lib.convert_numpy_to_native(raw_config)
lib.dump_config(raw_config_converted, f"{raw_config['main']['raw_config_path']}") 

job = args.job

n_trials = raw_config['main']['n_trials']

exp_dir = raw_config['main']['exp_dir']
metrics_all_path = f"{exp_dir}/metrics_all.csv"

for trial in range(n_trials):
    
    trial_dir = f"{exp_dir}/trial_%02d" % trial 
    trained_model_dir = f"{trial_dir}/trained_model" 
    raw_config['train']['main']['trial_dir'] = trial_dir
    raw_config['train']['main']['trained_model_dir'] = trained_model_dir
    
    if 'train' in job:
        print('='*30, 'train-%d/%d'%((trial+1),n_trials),'='*30)
        if os.path.isdir(trial_dir):
            shutil.rmtree(trial_dir)
            print('deleted', trial_dir)
        os.makedirs(trained_model_dir, exist_ok=True)
        train(trial, raw_config)
        torch.cuda.empty_cache()

    if 'sample' in job:
        n_sample_batches = raw_config['sample']['main']['n_sample_batches']
        for n_sample_batch_ in range(n_sample_batches):
            print('='*20, 'train-%d/%d-sample-%d/%d'%(trial,n_trials,n_sample_batch_,n_sample_batches),'='*20)
            sample_dir = f"{trial_dir}/sample_%02d" % n_sample_batch_
            raw_config['sample']['main']['sample_dir'] = sample_dir
            os.makedirs(sample_dir, exist_ok=True)
            print('-'*19,'\ncreated', sample_dir)
            all_best_models = np.sort(glob(f'{trained_model_dir}/*/'))
            for best_i_, selected_model in enumerate(all_best_models):
                print(f'use {selected_model}')
                sample(n_sample_batch_, selected_model, raw_config)
                torch.cuda.empty_cache()
    
    if 'eval' in job:
        all_sample_dirs = np.sort(glob(f'{trial_dir}/sample_*/'))
        for eval_sample_i_, sample_dir_ in enumerate(all_sample_dirs):
            n_sample_ = sample_dir_.split('/')[-2]
            all_epoch_dirs = np.sort(glob(f'{sample_dir_}/epochs_*/'))
            for val_sample_epoch_i_, epoch_dir_ in enumerate(all_epoch_dirs):
                n_epoch_ = ('_').join(epoch_dir_.split('/')[-2].split('_')[:2])         
                print('='*20, 'train-%d/%d-%s-%s-eval'%(trial,n_trials,n_sample_,n_epoch_),'='*20)  
                eval_dir = f"{trial_dir}/eval/{n_epoch_}/{n_sample_}"
                os.makedirs(eval_dir, exist_ok=True)
                print('-'*19,'\ncreated', eval_dir)                
                raw_config['eval']['main']['epoch_dir'] = epoch_dir_
                raw_config['eval']['main']['eval_dir'] = eval_dir
                raw_config_converted = lib.convert_numpy_to_native(raw_config)
                lib.dump_config(raw_config_converted, f"{raw_config['main']['raw_config_path']}") 
                lib.dump_config(raw_config_converted, f"{raw_config['train']['main']['trained_model_dir']}/config.toml") 
                lib.dump_config(raw_config_converted, f"{raw_config['sample']['main']['sample_dir']}/config.toml") 
                lib.dump_config(raw_config_converted, f"{eval_dir}/config.toml") 
                df_metrics_ = eval(n_sample_, n_epoch_, raw_config)
                torch.cuda.empty_cache()
                df_metrics_['trial'] = trial
                df_metrics_['epoch'] = int(n_epoch_.split('_')[-1])
                df_metrics_['sample'] = int(n_sample_.split('_')[-1])
                if trial==0 and eval_sample_i_ == 0 and val_sample_epoch_i_==0:
                    df_metrics_all = df_metrics_.copy()
                else:
                    df_metrics_all = pd.concat([df_metrics_,df_metrics_all], axis=0)
                    df_metrics_all.index = range(len(df_metrics_all))
                df_metrics_all.to_csv(metrics_all_path,index=False)

print('-'*30,'end','-'*30)
                
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------