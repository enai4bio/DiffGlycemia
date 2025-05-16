import argparse

# import subprocess
# import optuna
# import time
# import tqdm

import os
# from copy import deepcopy
# from glob import glob
# from pathlib import Path
# import pickle
# import shutil

import ast
import lib
from utils_train import *

# from datetime import datetime

import numpy as np
import pandas as pd
pd.set_option('display.min_rows', 10)
pd.set_option('display.max_columns', 20)

# import seaborn as sns
# import matplotlib.pyplot as plt

# import math
import zero
import torch

# ------------------------------------------------------------------------


from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------


def sample(n_sample_batch_, best_model_dir, raw_config):

    seed = n_sample_batch_
    zero.improve_reproducibility(seed) 
    print('sample seed:',seed)
    raw_config['sample']['main']['seed'] = seed

    raw_config_converted = lib.convert_numpy_to_native(raw_config)
    lib.dump_config(raw_config_converted, f"{raw_config['main']['raw_config_path']}") 
    lib.dump_config(raw_config_converted, f"{raw_config['train']['main']['trained_model_dir']}/config.toml")
    lib.dump_config(raw_config_converted, f"{raw_config['sample']['main']['sample_dir']}/config.toml") 

    D = lib.load_pickle(raw_config['train']['main']['dataset_path'])

    # best_model_path = raw_config['sample']['main']['best_model_path']
    best_model_path = f"{best_model_dir}diffusion.pt"
    denoise_mlp = torch.load(best_model_path)
    print('-'*19,'\nloaded:', best_model_path)

    diffusion = GaussianMultinomialDiffusion(denoise_mlp, raw_config) 
    device = raw_config['main']['device']
    diffusion.to(device)
    diffusion.eval()

    diffusion.random_sample(D, best_model_path, raw_config)
    # diffusion.forward_sample(D, raw_config)

    print()



    # # if balance_sample_method == 'fix':
    #     # empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], empirical_class_dist[0]
    #     # x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)
    #     # pass

    # # elif balance_sample_method == 'fill':
    #     # ix_major = empirical_class_dist.argmax().item()
    #     # val_major = empirical_class_dist[ix_major].item()
    #     # x_gen, y_gen = [], []
    #     # for i in range(empirical_class_dist.shape[0]):
    #     #     if i == ix_major:
    #     #         continue
    #     #     distrib = torch.zeros_like(empirical_class_dist)
    #     #     distrib[i] = 1
    #     #     num_samples = val_major - empirical_class_dist[i].item()
    #     #     x_temp, y_temp = diffusion.sample_all(num_samples, batch_size, distrib.float(), ddim=False)
    #     #     x_gen.append(x_temp)
    #     #     y_gen.append(y_temp)
    #     # x_gen = torch.cat(x_gen, dim=0)
    #     # y_gen = torch.cat(y_gen, dim=0)
    #     # pass

    # ------------

    