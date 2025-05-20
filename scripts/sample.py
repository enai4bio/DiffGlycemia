import argparse

import os

import ast
import lib
from utils_train import *

import numpy as np
import pandas as pd

import zero
import torch

from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion


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

    best_model_path = f"{best_model_dir}diffusion.pt"
    denoise_mlp = torch.load(best_model_path)
    print('-'*19,'\nloaded:', best_model_path)

    diffusion = GaussianMultinomialDiffusion(denoise_mlp, raw_config) 
    device = raw_config['main']['device']
    diffusion.to(device)
    diffusion.eval()

    diffusion.random_sample(D, best_model_path, raw_config)


