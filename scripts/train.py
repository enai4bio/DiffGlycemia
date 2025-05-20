import argparse

import os
from copy import deepcopy
import shutil

import ast
import lib
from utils_train import *

import numpy as np
import pandas as pd
pd.set_option('display.min_rows', 10)
pd.set_option('display.max_columns', 20)

import matplotlib.pyplot as plt

import math
import zero
import torch

from tab_ddpm import GaussianMultinomialDiffusion
from tab_ddpm.modules import MLPDiffusion


def repeat_2d_tensor_to_rows(original_tensor, target_rows):
    original_rows = original_tensor.size(0)
    repeat_count = target_rows // original_rows 
    remaining_rows = target_rows % original_rows 
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

class Trainer: 
    
    def __init__(self, dataset, diffusion, train_iter, raw_config):

        self.dataset = dataset
        self.diffusion = diffusion
        self.raw_config = raw_config

        self.train_iter = train_iter 
        self.steps = steps = raw_config['train']['main']['steps']
        self.init_lr = lr = raw_config['train']['main']['init_lr']
        self.weight_decay = weight_decay = raw_config['train']['main']['weight_decay']
        self.optimizer = torch.optim.AdamW(
            self.diffusion.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            )
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

    def _anneal_lr_linear(self, step): 
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _anneal_lr_exp(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * math.exp(-self.decay_rate * frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict): 
        x = x.to(self.device)
        out_dict['y'] = out_dict['y'].long().to(self.device)
        self.diffusion.train() 
        self.optimizer.zero_grad()
        loss_multi, loss_gauss, _, _, _ = self.diffusion.mixed_loss(x, out_dict, self.raw_config)  
        loss = loss_multi + loss_gauss 
        loss.backward()
        self.optimizer.step() 
        return loss_multi.item(), loss_gauss.item() 
    

    def _run_eval(self, x, out_dict):
        x = x.to(self.device)
        out_dict['y'] = out_dict['y'].to(self.device)
        self.diffusion.eval()
        loss_multi, loss_gauss, x_in, t, model_out = self.diffusion.mixed_loss(x, out_dict, self.raw_config)
        return loss_multi.item(), loss_gauss.item(), x_in, t, model_out

    def run_loop(self): 

        raw_config = self.raw_config
        trained_model_dir = f"{self.raw_config['train']['main']['trained_model_dir']}"
        step = 0
        steps = raw_config['train']['main']['steps']

        if ast.literal_eval(raw_config['train']['main']['anneal']):
            anneal_method = raw_config['train']['main']['anneal_method']
            print(anneal_method)
            if anneal_method == 'exp':
                print(raw_config['train']['main']['decay_rate'])

        print('device:',self.device)

        best_model_dir_list = []
        while step < steps: 

            x, out_dict = next(self.train_iter) 
            out_dict = {'y': out_dict}

            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict) 

            if ast.literal_eval(raw_config['train']['main']['anneal']):
                if anneal_method == 'liner':
                    self._anneal_lr_linear(step) 
                elif anneal_method == 'exp':
                    self._anneal_lr_exp(step) 
                else:
                    pass

            train_mloss, train_gloss, train_in, train_t, train_out = self._run_eval(self.x_train, self.out_dict_train)
            val_mloss, val_gloss, val_in, val_t, val_out = self._run_eval(self.x_val, self.out_dict_val)
            test_mloss, test_gloss, test_in, test_t, test_out = self._run_eval(self.x_test, self.out_dict_test)

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
            
            if raw_config['data']['y']['label_column'] == 'CVDs':
                if (min_loss_val_idx!=step):
                    step += 1
                    continue
            elif raw_config['data']['y']['label_column'] == 'Adverse Pregnancy Outcome':
                if step < steps-1:
                    step += 1
                    continue

            
            best_model_dir = f"{trained_model_dir}/step_{'%06d' % (step+1)}"
            raw_config['sample']['main']['best_model_dir'] = best_model_dir
            raw_config['sample']['main']['best_model_path'] = f'{best_model_dir}/diffusion.pt'

            raw_config_converted = lib.convert_numpy_to_native(raw_config)
            lib.dump_config(raw_config_converted, f"{raw_config['main']['raw_config_path']}") 
            lib.dump_config(raw_config_converted, f"{raw_config['train']['main']['trained_model_dir']}/config.toml")

            os.makedirs(best_model_dir,exist_ok=True)
            torch.save(self.diffusion._denoise_fn, f"{best_model_dir}/diffusion.pt") 

            best_model_dir_list.append(best_model_dir)
            if len(best_model_dir_list) > raw_config['train']['main']['n_trained_model_dir_list']:
                remove_model_dir = best_model_dir_list.pop(0)
                shutil.rmtree(remove_model_dir)

            step += 1

        self.loss_history.to_csv(f"{trained_model_dir}/loss.csv" , index=False)

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
        plt.savefig(f"{trained_model_dir}/loss_plot.png" , bbox_inches='tight')  
        plt.close() 


def train(trial, raw_config): 
    
    seed = trial
    zero.improve_reproducibility(seed) 
    raw_config['train']['main']['seed'] = seed
    print('train seed:',seed)


    real_data_dir = raw_config['data']['real_data_dir'] 
    raw_config['train']['T']['seed'] = seed
    T = lib.Transformations(**raw_config['train']['T']) 
    dataset = make_dataset( 
        real_data_dir,
        T,
    )
    raw_config['data']['y']['num_classes'] = len(np.unique(dataset.y['train']))
    trained_model_dir = raw_config['train']['main']['trained_model_dir']
    dataset_path = f"{trained_model_dir}/dataset_path.pkl"
    # Path(dataset_path).write_bytes(pickle.dumps(dataset))
    lib.dump_pickle(dataset, dataset_path)
    raw_config['train']['main']['dataset_path'] = dataset_path

    batch_size = raw_config['train']['main']['batch_size']
    print('batch_size:',batch_size)
    raw_config['train']['main']['batch_size'] = batch_size
    epoch = raw_config['train']['main']['epoch']
    steps = int(np.ceil(len(dataset.y['train']) * epoch / batch_size))
    raw_config['train']['main']['steps'] = steps


    K = np.array(dataset.get_category_sizes('train')) 
    n_numerical_features = dataset.X_num['train'].shape[1] 
    n_categorical_features = dataset.X_cat['train'].shape[1] 

    raw_config['data']['x']['num_classes'] = K
    raw_config['data']['x']['n_numerical_features'] = n_numerical_features
    raw_config['data']['x']['n_categorical_features'] = n_categorical_features
    
    dim_in = np.sum(K) + n_numerical_features 
    raw_config['train_mlp_params']['dim_in'] = dim_in
    
    device = raw_config['main']['device']
    denoise_mlp = MLPDiffusion(raw_config)
    denoise_mlp.to(device)

    raw_config_converted = lib.convert_numpy_to_native(raw_config)
    lib.dump_config(raw_config_converted, f"{raw_config['main']['raw_config_path']}") 
    lib.dump_config(raw_config_converted, f"{raw_config['train']['main']['trained_model_dir']}/config.toml")

    diffusion = GaussianMultinomialDiffusion(denoise_mlp, raw_config) 
    diffusion.to(device)
    diffusion.train()

    train_loader = lib.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)
    trainer = Trainer(dataset, diffusion, train_loader, raw_config)
    
    print('-'*69, '\ntrain the denoise model')

    trainer.run_loop() 
