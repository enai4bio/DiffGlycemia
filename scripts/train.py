import argparse

# import subprocess
# import optuna
# import time
# import tqdm

import os
from copy import deepcopy
# from glob import glob
# from pathlib import Path
# import pickle
import shutil

import ast
import lib
from utils_train import *

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

from tab_ddpm import GaussianMultinomialDiffusion
from tab_ddpm.modules import MLPDiffusion
# from tab_ddpm.modules import ResNetDiffusion
# from tab_ddpm.modules import TransformerDiffusion


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# def repeat_1d_tensor_to_size(original_tensor, target_size):
#     original_size = original_tensor.size(0)
#     repeat_count = target_size // original_siz
#     remaining_elements = target_size % original_size
#     expanded_tensor = original_tensor.repeat(repeat_count)
#     if remaining_elements > 0:
#         expanded_tensor = torch.cat([expanded_tensor, original_tensor[:remaining_elements]])
#     return expanded_tensor

def repeat_2d_tensor_to_rows(original_tensor, target_rows):
    original_rows = original_tensor.size(0)
    repeat_count = target_rows // original_rows  # 计算需要完整重复的次数
    remaining_rows = target_rows % original_rows  # 剩余行数
    expanded_tensor = original_tensor.repeat(repeat_count, 1)
    if remaining_rows > 0:
        expanded_tensor = torch.cat([expanded_tensor, original_tensor[:remaining_rows]], dim=0)
    return expanded_tensor
    
def output_forwarded(x_in, t, model_out, raw_config, output_type):

    numerical_feature_columns = raw_config['data']['x']['numerical_feature_columns']
    categorical_feature_columns = raw_config['data']['x']['categorical_feature_columns']
    categorical_feature_columns_ab = [f"{feature}_{suffix}" for feature in categorical_feature_columns for suffix in ['_a', '_b']]
    
    df_x_in = pd.DataFrame(x_in.cpu().numpy(),columns=numerical_feature_columns+categorical_feature_columns_ab)
    df_t =  pd.Series(t.cpu().numpy())
    df_model_out = pd.DataFrame(model_out.detach().cpu().numpy(),columns=numerical_feature_columns+categorical_feature_columns_ab)
    
    best_model_dir = raw_config['sample']['main']['best_model_dir'] 
    
    df_x_in.to_csv(f'{best_model_dir}/{output_type}_in.csv', index=False)
    df_t.to_csv(f'{best_model_dir}/{output_type}_t.csv', index=False)
    df_model_out.to_csv(f'{best_model_dir}/{output_type}_out.csv', index=False)
    
    # print('outputted', (f'{best_model_dir}/{output_type}_in.csv'))
    # print('outputted', (f'{best_model_dir}/{output_type}_t.csv'))
    # print('outputted', (f'{best_model_dir}/{output_type}_out.csv'))

class Trainer: # 训练模型
    
    def __init__(self, dataset, diffusion, train_iter, raw_config):

        self.dataset = dataset
        self.diffusion = diffusion
        self.raw_config = raw_config

        self.train_iter = train_iter # train_loader
        self.steps = steps = raw_config['train']['main']['steps']
        self.init_lr = lr = raw_config['train']['main']['init_lr']
        self.weight_decay = weight_decay = raw_config['train']['main']['weight_decay']
        self.optimizer = torch.optim.AdamW(
            self.diffusion.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            )
        self.log_every = raw_config['train']['main']['log_every']
        self.device = raw_config['main']['device']
        self.loss_history = pd.DataFrame(columns=[
            'step', 
            'batch_mloss','batch_gloss', 
            'train_mloss','train_gloss', 
            'val_mloss','val_gloss', 
            'test_mloss','test_gloss', 
            'batch_loss',
            'train_loss',
            'val_loss',
            'test_loss',
            'val_best'
            ])

        # self.ema_model = deepcopy(self.diffusion._denoise_fn)
        # self.ema_model = deepcopy(self.diffusion)
        # for param in self.ema_model.parameters():
        #     param.detach_()
        
        self.x_train = torch.from_numpy(np.hstack((
            self.dataset.X_num['train'], self.dataset.X_cat['train']
            )))
        self.x_val = torch.from_numpy(np.hstack((
            self.dataset.X_num['val'], self.dataset.X_cat['val']
            )))
        self.x_test = torch.from_numpy(np.hstack((
            self.dataset.X_num['test'], self.dataset.X_cat['test']
            )))

        self.out_dict_train = {'y':torch.from_numpy(self.dataset.y['train'])}
        self.out_dict_val = {'y': torch.from_numpy(self.dataset.y['val'])}
        self.out_dict_test = {'y': torch.from_numpy(self.dataset.y['test'])}

        # //////////// __init__ ////////////

    def _anneal_lr_linear(self, step): # decrease learning rate; # from Line 57
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _anneal_lr_exp(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * math.exp(-self.decay_rate * frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict): # from Line 55 # back-p
        x = x.to(self.device)
        out_dict['y'] = out_dict['y'].long().to(self.device)

        # ------------
        self.diffusion.train() 
        self.optimizer.zero_grad()
        loss_multi, loss_gauss, _, _, _ = self.diffusion.mixed_loss(x, out_dict, self.raw_config)  # 分别返回 categorical 和 numverical 的 loss
        loss = loss_multi + loss_gauss # categorical 和 numverical 的 loss 的 加和
        loss.backward()
        self.optimizer.step() 
        # ------------
        # with torch.enable_grad():  
        #     self.optimizer.zero_grad()
        #     loss_multi, loss_gauss, _, _, _ = self.diffusion.mixed_loss(x, out_dict, self.raw_config)  # 分别返回 categorical 和 numverical 的 loss
        #     loss = loss_multi + loss_gauss # categorical 和 numverical 的 loss 的 加和
        #     loss.backward()
        #     self.optimizer.step() 
        # ------------
        return loss_multi.item(), loss_gauss.item() # 反向传递后，继续上传loss，做记录
    

    # def _run_eval(self, x, out_dict,n_head):
    def _run_eval(self, x, out_dict, ma_model=False):
        x = x.to(self.device)
        out_dict['y'] = out_dict['y'].to(self.device)
        # ------------
        self.diffusion.eval()
        loss_multi, loss_gauss, x_in, t, model_out = self.diffusion.mixed_loss(x, out_dict, self.raw_config)
        # ------------
        # with torch.no_grad():
        #     if ma_model:
        #         # loss_multi, loss_gauss, x_in, t, model_out = self.ema_model.mixed_loss(x, out_dict, self.raw_config)
        #         pass
        #     else:
        #         loss_multi, loss_gauss, x_in, t, model_out = self.diffusion.mixed_loss(x, out_dict, self.raw_config)
        # ------------
        return loss_multi.item(), loss_gauss.item(), x_in, t, model_out

    def run_loop(self): # 相当于 train，最重要的步骤

        raw_config = self.raw_config
        trained_model_dir = f"{self.raw_config['train']['main']['trained_model_dir']}"

        float_digit = 3
        log_start = raw_config['train']['main']['log_start'] # log_start = 1000
        print('-'*19,'\nprint_start:', log_start)

        step = 0
        steps = raw_config['train']['main']['steps']
        print('steps:', steps)

        print('lr:', self.init_lr)
        if ast.literal_eval(raw_config['train']['main']['anneal']):
            anneal_method = raw_config['train']['main']['anneal_method']
            print(anneal_method)
            if anneal_method == 'exp':
                print(raw_config['train']['main']['decay_rate'])

        print('device:',self.device)

        diffusion_moving_average = ast.literal_eval(raw_config['train']['main']['diffusion_moving_average'])
        print('diffusion_moving_average:', diffusion_moving_average)

        best_model_dir_list = []
        while step < steps: # self.steps = 30000 跑3万个step

            x, out_dict = next(self.train_iter) # x = 4096 x 11; y = 4096
            out_dict = {'y': out_dict}

            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict) # 分别返回 categorical 和 numerical 的 loss
            # update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters()) # targ=targ×rate+src×(1−rate)

            if ast.literal_eval(raw_config['train']['main']['anneal']):
                if anneal_method == 'liner':
                    self._anneal_lr_linear(step) # 修改 lr， 熄火
                elif anneal_method == 'exp':
                    self._anneal_lr_exp(step) # 修改 lr， 熄火
                else:
                    pass

            train_mloss, train_gloss, train_in, train_t, train_out = self._run_eval(self.x_train, self.out_dict_train,diffusion_moving_average)
            val_mloss, val_gloss, val_in, val_t, val_out = self._run_eval(self.x_val, self.out_dict_val,diffusion_moving_average)
            test_mloss, test_gloss, test_in, test_t, test_out = self._run_eval(self.x_test, self.out_dict_test,diffusion_moving_average)

            self.loss_history.loc[len(self.loss_history)] =[
                step, 
                batch_loss_multi, batch_loss_gauss,
                train_mloss, train_gloss, 
                val_mloss, val_gloss, 
                test_mloss, test_gloss, 
                batch_loss_multi + batch_loss_gauss,
                train_mloss + train_gloss,
                val_mloss + val_gloss,
                test_mloss + test_gloss,
                ''            
                ]
            min_loss_val_idx = self.loss_history['val_loss'].idxmin()

            if min_loss_val_idx == step:
                self.loss_history.loc[len(self.loss_history)-1,'val_best'] = 'O'
            
            # ------------

            if (step < log_start):
                step += 1
                continue

            # ------------
            if (min_loss_val_idx!=step) and ((step+1)%self.log_every!=0) and ((step+1)!=steps):
                step += 1
                continue
            
            loss_tag = '***' if (min_loss_val_idx==step) else ''
            print(f'step: {(step+1)}/{steps} --- '
                f'batch: {np.around(batch_loss_multi, float_digit)}+{np.around(batch_loss_gauss, float_digit)}={np.around((batch_loss_multi + batch_loss_gauss), float_digit)}; '
                f'train: {np.around(train_mloss, float_digit)}+{np.around(train_gloss, float_digit)}={np.around((train_mloss + train_gloss), float_digit)}; '
                f'valid: {np.around(val_mloss, float_digit)}+{np.around(val_gloss, float_digit)}={np.around((val_mloss + val_gloss), float_digit)}{loss_tag}; '
                f'test: {np.around(test_mloss, float_digit)}+{np.around(test_gloss, float_digit)}={np.around((test_mloss + test_gloss), float_digit)}; '
                )

            best_model_dir = f"{trained_model_dir}/step_{'%06d' % (step+1)}"
            raw_config['sample']['main']['best_model_dir'] = best_model_dir
            raw_config['sample']['main']['best_model_path'] = f'{best_model_dir}/diffusion.pt'
            # raw_config['sample']['main']['average_model_path'] = f'{best_model_dir}/move_ave.pt'

            raw_config_converted = lib.convert_numpy_to_native(raw_config)
            lib.dump_config(raw_config_converted, f"{raw_config['main']['raw_config_path']}") 
            lib.dump_config(raw_config_converted, f"{raw_config['train']['main']['trained_model_dir']}/config.toml")

            os.makedirs(best_model_dir,exist_ok=True)
            torch.save(self.diffusion._denoise_fn, f"{best_model_dir}/diffusion.pt") 
            # torch.save(self.ema_model, f"{best_model_dir}/ema.pt") 

            best_model_dir_list.append(best_model_dir)
            if len(best_model_dir_list) > raw_config['train']['main']['n_trained_model_dir_list']:
                remove_model_dir = best_model_dir_list.pop(0)
                shutil.rmtree(remove_model_dir)
                # print('removed', remove_model_dir)

            output_forwarded(train_in, train_t, train_out, raw_config, 'train')
            output_forwarded(val_in, val_t, val_out, raw_config, 'val')
            output_forwarded(test_in, test_t, test_out, raw_config, 'test')

            step += 1

            # ------------

        self.loss_history.to_csv(f"{trained_model_dir}/loss.csv" , index=False)

        # ------------
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history.loc[:,'step'], self.loss_history.loc[:,'train_loss'], label='train_loss', alpha=0.6)
        plt.plot(self.loss_history.loc[:,'step'], self.loss_history.loc[:,'val_loss'], label='val_loss', alpha=0.6)
        plt.plot(self.loss_history.loc[:,'step'], self.loss_history.loc[:,'test_loss'], label='test_loss', alpha=0.6)
        val_best_indices = self.loss_history[self.loss_history['val_best'] == 'O'].index
        plt.scatter(self.loss_history.loc[val_best_indices, 'step'], self.loss_history.loc[val_best_indices, 'val_loss'], marker='*', color='red', label='Val Best')
        plt.legend()
        plt.title('Training, Validation, and Test Loss Over Steps')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{trained_model_dir}/loss_plot.png" , bbox_inches='tight')  # 保存图片到当前目录
        plt.close()  # 关闭图形，不显示
        # //////////// while ////////////



def train(trial, raw_config): 
    
    seed = trial
    zero.improve_reproducibility(seed) 
    raw_config['train']['main']['seed'] = seed
    print('train seed:',seed)

    # ------------

    real_data_dir = raw_config['data']['real_data_dir'] # '/media/jie/toshiba_4t/7exp_t/tang/cvds_AUC_correct/data'
    raw_config['train']['T']['seed'] = seed
    T = lib.Transformations(**raw_config['train']['T']) 
    # change_val = ast.literal_eval(raw_config['train']['main']['change_val'])
    # print('change_val:',change_val)
    dataset = make_dataset( # 准备数据： 补na；归一化
        real_data_dir,
        T,
        # change_val
    )
    raw_config['data']['y']['num_classes'] = len(np.unique(dataset.y['train']))
    trained_model_dir = raw_config['train']['main']['trained_model_dir']
    dataset_path = f"{trained_model_dir}/dataset_path.pkl"
    # Path(dataset_path).write_bytes(pickle.dumps(dataset))
    lib.dump_pickle(dataset, dataset_path)
    raw_config['train']['main']['dataset_path'] = dataset_path

    # ------------
    # batch_size = np.shape(dataset.y['train'])[0]
    batch_size = raw_config['train']['main']['batch_size']
    print('batch_size:',batch_size)
    # ------------
    raw_config['train']['main']['batch_size'] = batch_size
    epoch = raw_config['train']['main']['epoch']
    steps = int(np.ceil(len(dataset.y['train']) * epoch / batch_size))
    # steps = int(np.floor(len(dataset.y['train']) * epoch / batch_size))
    # steps = int(np.round(len(dataset.y['train']) * epoch / batch_size))
    print('steps:',steps)
    raw_config['train']['main']['steps'] = steps

    # ------------

    K = np.array(dataset.get_category_sizes('train')) # 各个category feature 的 nunique; array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    n_numerical_features = dataset.X_num['train'].shape[1] # numerical feature 的个数
    n_categorical_features = dataset.X_cat['train'].shape[1] # categorical feature 的个数

    raw_config['data']['x']['num_classes'] = K
    raw_config['data']['x']['n_numerical_features'] = n_numerical_features
    raw_config['data']['x']['n_categorical_features'] = n_categorical_features
    
    dim_in = np.sum(K) + n_numerical_features # 模型初始输入的维度
    raw_config['train_mlp_params']['dim_in'] = dim_in # model的 超参数
    
    device = raw_config['main']['device']
    denoise_mlp = MLPDiffusion(raw_config) # denoise model
    denoise_mlp.to(device)
    # print('-'*69)
    # print('denoise model:', denoise_mlp)

    # ------------
    raw_config_converted = lib.convert_numpy_to_native(raw_config)
    lib.dump_config(raw_config_converted, f"{raw_config['main']['raw_config_path']}") 
    lib.dump_config(raw_config_converted, f"{raw_config['train']['main']['trained_model_dir']}/config.toml")

    diffusion = GaussianMultinomialDiffusion(denoise_mlp, raw_config) 
    print('-'*19,f'\n{diffusion}')
    diffusion.to(device)
    diffusion.train()

    # ------------

    # train_loader = lib.prepare_beton_loader(dataset, split='train', batch_size=batch_size)
    train_loader = lib.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)
    trainer = Trainer(dataset, diffusion, train_loader, raw_config)
    
    print('-'*69, '\ntrain the denoise model')

    trainer.run_loop() # jump to Line 46 # TRAIN !!!

# ////////////
