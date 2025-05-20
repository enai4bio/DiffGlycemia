"""
Based on https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
and https://github.com/ehoogeboom/multinomial_diffusion
"""

from .utils import *

import os
import lib
from copy import deepcopy

import ast
from tqdm import tqdm

from utils_train import *

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE

import math
import random

import torch
import torch.nn.functional as F

import joblib

"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8



# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.preprocessing import StandardScaler

def cox_box_scale_transform(df):
    """
    对输入的DataFrame的每一列进行Cox-Box变换，然后进行标准化。

    参数:
    df (pd.DataFrame): 输入的DataFrame

    返回:
    transformed_df (pd.DataFrame): 包含Cox-Box变换和标准化后的DataFrame
    lambdas (dict): 每一列对应的λ值
    scalers (dict): 每一列对应的StandardScaler对象
    """
    transformed_df = df.copy()  # 创建一个副本以避免修改原始DataFrame
    lambdas = {}  # 存储每一列的λ值
    scalers = {}  # 存储每一列的StandardScaler对象

    for col in transformed_df.columns:
        # 对每一列进行Cox-Box变换，并存储λ值
        transformed_df[col], lambdas[col] = boxcox(transformed_df[col] + 1)  # 加1以避免零值问题

        # 对每一列进行标准化
        scaler = StandardScaler()
        transformed_df[col] = scaler.fit_transform(transformed_df[col].values.reshape(-1, 1)).flatten()
        scalers[col] = scaler  # 存储scaler对象

    return transformed_df, lambdas, scalers

def inverse_cox_box_scale_transform(transformed_df, lambdas, scalers):
    """
    对输入的DataFrame的每一列进行逆标准化和逆Cox-Box变换。

    参数:
    transformed_df (pd.DataFrame): 经过Cox-Box变换和标准化后的DataFrame
    lambdas (dict): 每一列对应的λ值
    scalers (dict): 每一列对应的StandardScaler对象

    返回:
    pd.DataFrame: 包含逆变换后数据的DataFrame
    """
    original_df = transformed_df.copy()  # 创建一个副本以避免修改原始DataFrame

    for col in original_df.columns:
        # 对每一列进行逆标准化
        original_df[col] = scalers[col].inverse_transform(original_df[col].values.reshape(-1, 1)).flatten()

        # 对每一列进行Cox-Box逆变换
        original_df[col] = inv_boxcox(original_df[col], lambdas[col]) - 1  # 减去1以恢复原始数据

    return original_df

# generated_samples, out_dict, label, n_selects = generated_samples, out_dict, 0, 100
def random_select(generated_samples, out_dict, label, n_selects):
    indices = torch.where(out_dict['y'] == label)[0]
    # n_selects = min(len(indices), n_selects) ## only for when generate_times == sample_times
    if indices.shape[0] > n_selects:
        selected_indices = indices[np.random.choice(indices.shape[0], n_selects, replace=False)].to('cpu')
    else:
        print('full sample')
        selected_indices = indices.to('cpu')
    selected_samples = generated_samples[selected_indices]
    selected_out_dict_y = out_dict['y'][selected_indices]
    return selected_samples, selected_out_dict_y, selected_indices

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    Focal Loss implementation in PyTorch.
    :param inputs: Model predictions, shape: [batch_size, num_classes]
    :param targets: True labels, shape: [batch_size]
    :param alpha: Balancing parameter for positive class.
    :param gamma: Focusing parameter.
    :param reduction: Reduction type, 'mean' or 'sum'.
    :return: Focal loss value.
    """
    # Calculate probabilities of the positive class
    probs = torch.sigmoid(inputs)
    
    # Calculate the focal loss components
    term1 = (1 - probs) ** gamma * torch.log(probs)
    term2 = probs ** gamma * torch.log(1 - probs)
    
    # Calculate the weighted loss for positive and negative samples
    loss = torch.where(targets == 1, alpha * term1, (1 - alpha) * term2)
    
    # Apply reduction
    if reduction == 'mean':
        return -loss.mean()
    elif reduction == 'sum':
        return -loss.sum()
    else:
        return -loss


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
  
# df = df01.copy()
def normalize_samples(df):
    # 对DataFrame中的每个样本进行归一化
    norms = df.apply(lambda x: (x**2).sum(), axis=1)**0.5
    return df.div(norms, axis=0).copy()


# df1, df2 = df_train_in.copy(), df_test_in.copy()
def cosine_similarity(df1, df2):
    df1_normal = normalize_samples(df1.copy())
    df2_normal = normalize_samples(df2.copy())
    df_similarity = pd.DataFrame(
        index=df1_normal.index, 
        columns=df2_normal.index)
    # for idx1 in tqdm(df1_normal.index):
    for idx1 in tqdm(df1_normal.index):
        # print(idx1,'/',len(df1_normal.index))
        for idx2 in df2_normal.index:
            series1 = df1_normal.loc[idx1]
            series2 = df2_normal.loc[idx2]
            df_similarity.loc[idx1, idx2] = np.dot(
                np.array(series1),
                np.array(series2)
            )
    return df_similarity


def read_forworded(D, raw_config, tag):
    
    best_model_dir = f"{raw_config['sample']['main']['best_model_dir']}"
    
    df_train_ = pd.read_csv(f"{best_model_dir}/train_{tag}.csv")
    df_val_ = pd.read_csv(f"{best_model_dir}/val_{tag}.csv")
    df_test_ = pd.read_csv(f"{best_model_dir}/test_{tag}.csv")

    df_train_.index = D.split_index['train']
    df_val_.index = D.split_index['val']
    df_test_.index = D.split_index['test']

    df_ = pd.concat([df_train_, df_val_], axis=0).sort_index()
    
    return df_, df_test_

# df_, y_ = df_train_in_idxmax.copy(), y_train_idxmax.copy()
def balance_forwarded(df_, y_):

    print('unbalance rate: 1/(0+1) = %f' % (sum(y_)/len(y_)))
    ros = RandomOverSampler()
    X_train_ros, y_train_ros = ros.fit_resample(df_, y_)
    sm = SMOTE()
    X_train_smote, y_train_smote = sm.fit_resample(df_, y_)
    svnsm = SVMSMOTE()
    X_train_svnsm, y_train_svnsm = svnsm.fit_resample(df_, y_)

    return X_train_ros, y_train_ros, X_train_smote, y_train_smote, X_train_svnsm, y_train_svnsm

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------



def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps): # num_diffusion_timesteps = 100
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)



# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------



class GaussianMultinomialDiffusion(torch.nn.Module): # diffusaion 过程
    # def __init__(
    #         self,
    #         num_classes: np.array,
    #         num_numerical_features: int,
    #         denoise_fn, # 
    #         num_timesteps=1000,
    #         gaussian_loss_type='mse',
    #         gaussian_parametrization='eps',
    #         multinomial_loss_type='vb_stochastic',
    #         parametrization='x0',
    #         scheduler='cosine',
    #         device=torch.device('cpu')
    #     ):
    def __init__(self, denoise_fn, raw_config):
        super(GaussianMultinomialDiffusion, self).__init__()

        multinomial_loss_type = raw_config['train_diffusion_params']['multinomial_loss_type'] 
        parametrization = raw_config['train_diffusion_params']['parametrization'] 

        if multinomial_loss_type == 'vb_all':
            print('Computing the loss using the bound on _all_ timesteps.'
                  ' This is expensive both in terms of memory and computation.')

        self.device = raw_config['main']['device']
        self.num_classes = raw_config['data']['x']['num_classes'] 
        self.num_numerical_features = raw_config['data']['x']['n_numerical_features'] #
        self.num_classes = np.array(self.num_classes)
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([self.num_classes[i].repeat(self.num_classes[i]) for i in range(len(self.num_classes))])).to(self.device) 

        self.slices_for_classes = [np.arange(self.num_classes[0])] 
        offsets = np.cumsum(self.num_classes) 
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i])) 
        self.offsets = torch.from_numpy(np.append([0], offsets)).to(self.device) 

        self._denoise_fn = denoise_fn
        self.gaussian_loss_type = raw_config['train_diffusion_params']['gaussian_loss_type']
        self.gaussian_parametrization = raw_config['train_diffusion_params']['gaussian_parametrization'] 
        self.multinomial_loss_type = raw_config['train_diffusion_params']['multinomial_loss_type'] 
        self.num_timesteps = raw_config['train_diffusion_params']['n_timesteps'] 
        self.parametrization = raw_config['train_diffusion_params']['parametrization'] 
        self.scheduler = raw_config['train_diffusion_params']['scheduler'] 

        alphas = 1. - get_named_beta_schedule(self.scheduler, self.num_timesteps)
        alphas = torch.tensor(alphas.astype('float64')) 
        betas = 1. - alphas 

        log_alpha = np.log(alphas) 
        log_cumprod_alpha = np.cumsum(log_alpha) 

        log_1_min_alpha = log_1_min_a(log_alpha) 
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha) 

        alphas_cumprod = np.cumprod(alphas, axis=0) 
        alphas_cumprod_prev = torch.tensor(np.append(1.0, alphas_cumprod[:-1])) 
        alphas_cumprod_next = torch.tensor(np.append(alphas_cumprod[1:], 0.0)) 
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        # Gaussian diffusion

        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.from_numpy(
            np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        ).float().to(self.device)
        self.posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).float().to(self.device)
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * np.sqrt(alphas.numpy())
            / (1.0 - alphas_cumprod)
        ).float().to(self.device)

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # Convert to float32 and register buffers.
        self.register_buffer('alphas', alphas.float().to(self.device))
        self.register_buffer('log_alpha', log_alpha.float().to(self.device))
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float().to(self.device))
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float().to(self.device))
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float().to(self.device))
        self.register_buffer('alphas_cumprod', alphas_cumprod.float().to(self.device))
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float().to(self.device))
        self.register_buffer('alphas_cumprod_next', alphas_cumprod_next.float().to(self.device))
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod.float().to(self.device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod.float().to(self.device))
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod.float().to(self.device))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod.float().to(self.device))

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))


    # Gaussian part
    def gaussian_q_mean_variance(self, x_start, t):
        mean = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_1_min_cumprod_alpha, t, x_start.shape
        )
        return mean, variance, log_variance
    
    def gaussian_q_sample(self, x_start, t, noise=None): 
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape 
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    
    def gaussian_q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def gaussian_p_mean_variance(
        self, model_output, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None
    ):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_variance = torch.cat([self.posterior_variance[1].unsqueeze(0).to(self.device), (1. - self.alphas)[1:]], dim=0)  
        model_log_variance = torch.log(model_variance)

        model_variance = extract(model_variance, t, x.shape)
        model_log_variance = extract(model_log_variance, t, x.shape)


        if self.gaussian_parametrization == 'eps':
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        elif self.gaussian_parametrization == 'x0':
            pred_xstart = model_output
        else:
            raise NotImplementedError
            
        model_mean, _, _ = self.gaussian_q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        ), f'{model_mean.shape}, {model_log_variance.shape}, {pred_xstart.shape}, {x.shape}'

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    
    def _vb_terms_bpd(
        self, model_output, x_start, x_t, t, clip_denoised=False, model_kwargs=None
    ):
        true_mean, _, true_log_variance_clipped = self.gaussian_q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.gaussian_p_mean_variance(
            model_output, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"], "out_mean": out["mean"], "true_mean": true_mean}
    
    def _prior_gaussian(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        # t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=self.device)
        qt_mean, _, qt_log_variance = self.gaussian_q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)
    
    def _gaussian_loss(self, model_out, x_start, x_t, t, noise, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        terms = {}
        if self.gaussian_loss_type == 'mse':
            terms["loss"] = mean_flat((noise - model_out) ** 2)
        elif self.gaussian_loss_type == 'kl':
            terms["loss"] = self._vb_terms_bpd(
                model_output=model_out,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]


        return terms['loss']
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def gaussian_p_sample(
        self,
        model_out,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
    ):
        out = self.gaussian_p_mean_variance(
            model_out,
            x,
            t,
            clip_denoised=clip_denoised, 
            denoised_fn=denoised_fn, 
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        ) 

        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    # Multinomial part

    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - torch.log(self.num_classes_expanded)
        )

        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - torch.log(self.num_classes_expanded)
        )

        return log_probs

    def predict_start(self, model_out, log_x_t, t, out_dict):

        # model_out = self._denoise_fn(x_t, t.to(x_t.device), **out_dict)

        assert model_out.size(0) == log_x_t.size(0)
        assert model_out.size(1) == np.sum(self.num_classes), f'{model_out.size()}'

        log_pred = torch.empty_like(model_out)
        for ix in self.slices_for_classes:
            log_pred[:, ix] = F.log_softmax(model_out[:, ix], dim=1)
        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        # EV_log_qxt_x0 = self.q_pred(log_x_start, t)

        # print('sum exp', EV_log_qxt_x0.exp().sum(1).mean())
        # assert False

        # log_qxt_x0 = (log_x_t.exp() * EV_log_qxt_x0).sum(dim=1)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        # t_broadcast = t.to(log_x_start.device).view(-1, *num_axes) * torch.ones_like(log_x_start)
        t_broadcast = t.to(self.device).view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0.to(torch.float32))

        # unnormed_logprobs = log_EV_qxtmin_x0 +
        #                     log q_pred_one_timestep(x_t, t)
        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs \
            - sliced_logsumexp(unnormed_logprobs, self.offsets)

        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, model_out, log_x, t, out_dict):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(model_out, log_x, t=t, out_dict=out_dict)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(model_out, log_x, t=t, out_dict=out_dict)
        else:
            raise ValueError
        return log_model_pred

    @torch.no_grad()
    def p_sample(self, model_out, log_x, t, out_dict):
        model_log_prob = self.p_pred(model_out, log_x=log_x, t=t, out_dict=out_dict)
        out = self.log_sample_categorical(model_log_prob)
        return out

    @torch.no_grad()
    def p_sample_loop(self, shape, out_dict):
        # device = self.log_alpha.device

        b = shape[0]
        # start with random normal image.
        img = torch.randn(shape, device=device)

        for i in reversed(range(1, self.num_timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=self.device , dtype=torch.long), out_dict)
        return img

    @torch.no_grad()
    def _sample(self, image_size, out_dict, batch_size = 16):
        return self.p_sample_loop((batch_size, 3, image_size, image_size), out_dict)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, self.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=self.device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(img, torch.full((b,), i, device=self.device, dtype=torch.long))

        return img

    def log_sample_categorical(self, logits):
        full_sample = []
        for i in range(len(self.num_classes)):
            one_class_logits = logits[:, self.slices_for_classes[i]]
            uniform = torch.rand_like(one_class_logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
            sample = (gumbel_noise + one_class_logits).argmax(dim=1)
            full_sample.append(sample.unsqueeze(1))
        full_sample = torch.cat(full_sample, dim=1)
        log_sample = index_to_log_onehot(full_sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t) 

        log_sample = self.log_sample_categorical(log_EV_qxt_x0) 

        return log_sample

    def nll(self, log_x_start, out_dict):
        b = log_x_start.size(0)
        # device = log_x_start.device
        loss = 0
        for t in range(0, self.num_timesteps):
            t_array = (torch.ones(b, device=self.device) * t).long()

            kl = self.compute_Lt(
                log_x_start=log_x_start,
                log_x_t=self.q_sample(log_x_start=log_x_start, t=t_array),
                t=t_array,
                out_dict=out_dict)

            loss += kl

        loss += self.kl_prior(log_x_start)

        return loss

    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        # device = log_x_start.device
        ones = torch.ones(b, device=self.device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes_expanded * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def compute_Lt(self, model_out, log_x_start, log_x_t, t, out_dict, detach_mean=False):
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_x_t, t=t)
        log_model_prob = self.p_pred(model_out, log_x=log_x_t, t=t, out_dict=out_dict)

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, self.device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = (Lt_sqrt / Lt_sqrt.sum()).to(self.device)

            t = torch.multinomial(pt_all, num_samples=b, replacement=True).to(self.device)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long() 

            pt = torch.ones_like(t).float() / self.num_timesteps  
            return t, pt
        else:
            raise ValueError

    def _multinomial_loss(self, model_out, log_x_start, log_x_t, t, pt, out_dict):

        if self.multinomial_loss_type == 'vb_stochastic':
            kl = self.compute_Lt(
                model_out, log_x_start, log_x_t, t, out_dict
            )
            kl_prior = self.kl_prior(log_x_start)
            # Upweigh loss term of the kl
            vb_loss = kl / pt + kl_prior

            return vb_loss

        elif self.multinomial_loss_type == 'vb_all':
            # Expensive, dont do it ;).
            # DEPRECATED
            return -self.nll(log_x_start)
        else:
            raise ValueError()

    def log_prob(self, x, out_dict):
        # b, device = x.size(0), x.device
        b, device = x.size(0), self.device
        if self.training:
            return self._multinomial_loss(x, out_dict)

        else:
            log_x_start = index_to_log_onehot(x, self.num_classes)

            t, pt = self.sample_time(b, self.device, 'importance')

            kl = self.compute_Lt(
                log_x_start, self.q_sample(log_x_start=log_x_start, t=t), t, out_dict)

            kl_prior = self.kl_prior(log_x_start)

            # Upweigh loss term of the kl
            loss = kl / pt + kl_prior

            return -loss

    # inputs, targets, alpha, gamma, reduction = loss_multi_all, out_dict['y'], 0.25, 2.0, 'mean'

    def mixed_loss(self, x, out_dict, raw_config, mid_out=''):

        b = x.shape[0] # batch_size
        t, pt = self.sample_time(b, self.device, 'uniform') 

        x_num = x[:, :self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features:]

        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num) 
            x_num_t = self.gaussian_q_sample(x_num, t, noise=noise)
        if x_cat.shape[1] > 0:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes) 
            log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t) 
        x_in = torch.cat([x_num_t, log_x_cat_t], dim=1) 
        model_out = self._denoise_fn.forward( 
            x_in,
            t,
            **out_dict
        )
        model_out_num = model_out[:, :self.num_numerical_features]
        model_out_cat = model_out[:, self.num_numerical_features:]

        if x_cat.shape[1] > 0:
            loss_multi_all = self._multinomial_loss(model_out_cat, log_x_cat, log_x_cat_t, t, pt, out_dict) / len(self.num_classes) 
        if x_num.shape[1] > 0:
            loss_gauss_all = self._gaussian_loss(model_out_num, x_num, x_num_t, t, noise) 

        balance_loss_method = raw_config['train']['main']['balance_loss_method']
        if balance_loss_method == 'inverse_frequency':
            num_pos = (out_dict['y'] == 1).sum().item()
            num_neg = (out_dict['y'] == 0).sum().item()
            weight_pos = 1.0 / num_pos if num_pos > 0 else 1.0
            weight_neg = 1.0 / num_neg if num_neg > 0 else 1.0
            loss_multi = torch.where(out_dict['y'] == 1, 
                                     loss_multi_all * weight_pos, 
                                     loss_multi_all * weight_neg).mean()
            loss_gauss = torch.where(out_dict['y'] == 1, 
                                     loss_gauss_all * weight_pos, 
                                     loss_gauss_all * weight_neg).mean()

        elif balance_loss_method == 'focal_loss':
            loss_multi = focal_loss(loss_multi_all, out_dict['y'], alpha=0.25, gamma=2.0, reduction='mean')
            loss_gauss = focal_loss(loss_gauss_all, out_dict['y'], alpha=0.25, gamma=2.0, reduction='mean')

        else:
            loss_multi, loss_gauss = loss_multi_all.mean(), loss_gauss_all.mean()

        return loss_multi, loss_gauss, x_in, t, model_out 
    
    @torch.no_grad()
    def mixed_elbo(self, x0, out_dict):
        b = x0.size(0)
        # device = x0.device
        device = self.device

        x_num = x0[:, :self.num_numerical_features]
        x_cat = x0[:, self.num_numerical_features:]
        has_cat = x_cat.shape[1] > 0
        if has_cat:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes).to(self.device)

        gaussian_loss = []
        xstart_mse = []
        mse = []
        mu_mse = []
        out_mean = []
        true_mean = []
        multinomial_loss = []
        for t in range(self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()
            noise = torch.randn_like(x_num)

            x_num_t = self.gaussian_q_sample(x_start=x_num, t=t_array, noise=noise)
            if has_cat:
                log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t_array)
            else:
                log_x_cat_t = x_cat

            model_out = self._denoise_fn(
                torch.cat([x_num_t, log_x_cat_t], dim=1),
                t_array,
                **out_dict
            )
            
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]

            kl = torch.tensor([0.0])
            if has_cat:
                kl = self.compute_Lt(
                    model_out=model_out_cat,
                    log_x_start=log_x_cat,
                    log_x_t=log_x_cat_t,
                    t=t_array,
                    out_dict=out_dict
                )

            out = self._vb_terms_bpd(
                model_out_num,
                x_start=x_num,
                x_t=x_num_t,
                t=t_array,
                clip_denoised=False
            )

            multinomial_loss.append(kl)
            gaussian_loss.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_num) ** 2))
            # mu_mse.append(mean_flat(out["mean_mse"]))
            out_mean.append(mean_flat(out["out_mean"]))
            true_mean.append(mean_flat(out["true_mean"]))

            eps = self._predict_eps_from_xstart(x_num_t, t_array, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        gaussian_loss = torch.stack(gaussian_loss, dim=1)
        multinomial_loss = torch.stack(multinomial_loss, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)
        # mu_mse = torch.stack(mu_mse, dim=1)
        out_mean = torch.stack(out_mean, dim=1)
        true_mean = torch.stack(true_mean, dim=1)



        prior_gauss = self._prior_gaussian(x_num)

        prior_multin = torch.tensor([0.0])
        if has_cat:
            prior_multin = self.kl_prior(log_x_cat)

        total_gauss = gaussian_loss.sum(dim=1) + prior_gauss
        total_multin = multinomial_loss.sum(dim=1) + prior_multin
        return {
            "total_gaussian": total_gauss,
            "total_multinomial": total_multin,
            "losses_gaussian": gaussian_loss,
            "losses_multinimial": multinomial_loss,
            "xstart_mse": xstart_mse,
            "mse": mse,
            # "mu_mse": mu_mse
            "out_mean": out_mean,
            "true_mean": true_mean
        }

    @torch.no_grad()
    def gaussian_ddim_step(
        self,
        model_out_num,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        eta=0.0
    ):
        out = self.gaussian_p_mean_variance(
            model_out_num,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=None,
        )

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise

        return sample
    
    @torch.no_grad()
    def gaussian_ddim_sample(
        self,
        noise,
        T,
        out_dict,
        eta=0.0
    ):
        x = noise
        b = x.shape[0]
        # device = x.device
        device = self.device
        for t in reversed(range(T)):
            print('-'*15)
            print(f'Sample timestep {t:4d}', end='\r')
            t_array = (torch.ones(b, device=self.device) * t).long()
            out_num = self._denoise_fn(x, t_array, **out_dict)
            x = self.gaussian_ddim_step(
                out_num,
                x,
                t_array
            )
        print()
        return x

    @torch.no_grad()
    def gaussian_ddim_reverse_step(
        self,
        model_out_num,
        x,
        t,
        clip_denoised=False,
        eta=0.0
    ):
        assert eta == 0.0, "Eta must be zero."
        out = self.gaussian_p_mean_variance(
            model_out_num,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=None,
            model_kwargs=None,
        )

        eps = (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = extract(self.alphas_cumprod_next, t, x.shape)

        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_next)
            + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return mean_pred

    @torch.no_grad()
    def gaussian_ddim_reverse_sample(
        self,
        x,
        T,
        out_dict,
    ):
        b = x.shape[0]
        # device = x.device
        device = self.device
        for t in range(T):
            print(f'Reverse timestep {t:4d}', end='\r')
            t_array = (torch.ones(b, device=self.device) * t).long()
            out_num = self._denoise_fn(x, t_array, **out_dict)
            x = self.gaussian_ddim_reverse_step(
                out_num,
                x,
                t_array,
                eta=0.0
            )
        print()

        return x

    @torch.no_grad()
    def multinomial_ddim_step(
        self,
        model_out_cat,
        log_x_t,
        t,
        out_dict,
        eta=0.0
    ):
        # not ddim, essentially
        log_x0 = self.predict_start(model_out_cat, log_x_t=log_x_t, t=t, out_dict=out_dict)
        alpha_bar = extract(self.alphas_cumprod, t, log_x_t.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, log_x_t.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        coef1 = sigma
        coef2 = alpha_bar_prev - sigma * alpha_bar
        coef3 = 1 - coef1 - coef2
        log_ps = torch.stack([
            torch.log(coef1) + log_x_t,
            torch.log(coef2) + log_x0,
            torch.log(coef3) - torch.log(self.num_classes_expanded)
        ], dim=2)
        log_prob = torch.logsumexp(log_ps, dim=2)
        out = self.log_sample_categorical(log_prob)
        return out

    @torch.no_grad()
    def sample_ddim(self, num_samples, y_dist):
        b = num_samples
        # device = self.log_alpha.device
        device = self.device
        z_norm = torch.randn((b, self.num_numerical_features), device=self.device)

        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=self.device).float()
        if has_cat:
            uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=self.device)
            log_z = self.log_sample_categorical(uniform_logits)

        y = torch.multinomial(
            y_dist,
            num_samples=b,
            replacement=True
        )
        out_dict = {'y': y.long().to(self.device)}
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Sample diffusion timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            model_out = self._denoise_fn(
                torch.cat([z_norm, log_z], dim=1).float(),
                t,
                **out_dict
            )
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]
            z_norm = self.gaussian_ddim_step(model_out_num, z_norm, t, clip_denoised=False)
            if has_cat:
                log_z = self.multinomial_ddim_step(model_out_cat, log_z, t, out_dict)
        print()
        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample, out_dict

    @torch.no_grad()
    def sample(self, num_samples, y_dist): 
        b = num_samples
        # device = self.log_alpha.device
        device = self.device
        z_norm = torch.randn((b, self.num_numerical_features), device=self.device) 
        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=self.device).float()
        if has_cat:
            uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=self.device)
            log_z = self.log_sample_categorical(uniform_logits)
        y = torch.multinomial(
            y_dist,
            num_samples=b,
            replacement=True
        )
        out_dict = {'y': y.long().to(self.device)}
        for i in reversed(range(0, self.num_timesteps)): 
            print(f'Sample timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            model_out = self._denoise_fn(
                torch.cat([z_norm, log_z], dim=1).float(),
                t,
                **out_dict
            )
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]
            z_norm = self.gaussian_p_sample(model_out_num, z_norm, t, clip_denoised=False)['sample']
            if has_cat:
                log_z = self.p_sample(model_out_cat, log_z, t, out_dict)
        print()
        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample, out_dict
    
    def sample_all(self, num_samples, batch_size, y_dist, ddim=False):
        if ddim:
            print('Sample using DDIM.')
            sample_fn = self.sample_ddim
        else:
            sample_fn = self.sample
        b = batch_size
        all_y = []
        all_samples = []
        num_generated = 0
        while num_generated < num_samples:
            sample, out_dict = sample_fn(b, y_dist)
            mask_nan = torch.any(sample.isnan(), dim=1)
            sample = sample[~mask_nan]
            out_dict['y'] = out_dict['y'][~mask_nan]

            all_samples.append(sample)
            all_y.append(out_dict['y'].cpu())
            if sample.shape[0] != b:
                raise FoundNANsError
            num_generated += sample.shape[0]
        x_gen = torch.cat(all_samples, dim=0)[:num_samples]
        y_gen = torch.cat(all_y, dim=0)[:num_samples]
        return x_gen, y_gen
    

    @torch.no_grad()
    def sample2(self, D, n_samples, y_dist): 
        b = int(n_samples.item())
        device = self.device
        z_norm = torch.randn((b, self.num_numerical_features), device=self.device) 
        uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=self.device)
        log_z = self.log_sample_categorical(uniform_logits) 
        y = torch.multinomial(
            # y_dist,
            y_dist.float(),
            num_samples=b,
            replacement=True
        ) 
        out_dict = {'y': y.long().to(self.device )} 
        print('-'*19)
        for i in reversed(range(0, self.num_timesteps)):
            if i % 200 == 0:
                print(f'Sample timestep {i:4d}\n', end='\r')
            t = torch.full((b,), i, device=self.device , dtype=torch.long) 
            # ------------
            model_out = self._denoise_fn( 
                torch.cat([z_norm, log_z], dim=1).float(), 
                t, 
                **out_dict  
            ) 
            # ------------
            model_out_num = model_out[:, :self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features:]
            z_norm = self.gaussian_p_sample(model_out_num, z_norm, t, clip_denoised=True)['sample'] # 
            
            log_z = self.p_sample(model_out_cat, log_z, t, out_dict)
        
        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        z_cat = ohe_to_categories(z_ohe, self.num_classes)
        generated_samples = torch.cat([z_norm, z_cat], dim=1).cpu()
        return generated_samples, out_dict
    

    def random_sample(self, D, best_model_path, raw_config):

        _, y_dist = torch.unique(torch.from_numpy(D.y['train']), return_counts=True)
        print('-'*19)
        
        n_generate_times = raw_config['sample']['main']['n_generate_times'] 
        print('n_generate_times:',n_generate_times)
        n_sample_times = raw_config['sample']['main']['n_sample_times'] 
        print('n_sample_times:',n_sample_times)
        assert(n_generate_times>=n_sample_times)

        n_samples = np.ceil((y_dist.max() ** 2) / y_dist.min() + y_dist.max()) * n_generate_times + 100

        generated_samples, out_dict = self.sample2(D, n_samples, y_dist)
        generated_samples_0, selected_out_dict_y_0, indices_0 = random_select(generated_samples, out_dict, 0, y_dist.sum().item()*n_sample_times)
        generated_samples_1, selected_out_dict_y_1,indices_1 = random_select(generated_samples, out_dict, 1, y_dist.sum().item()*n_sample_times)

        x_gen = torch.cat((generated_samples_0, generated_samples_1), dim=0)
        y_gen = torch.cat((selected_out_dict_y_0, selected_out_dict_y_1), dim=0)
        print('-'*19,'\nfinal 1 / (0+1) rate: %.0f%%' % (y_gen.sum()/len(y_gen)*100) )

        X_gen, y_gen = x_gen.cpu().numpy(), y_gen.cpu().numpy()

        n_numerical_features = raw_config['data']['x']['n_numerical_features']
        X_num_generated = D.num_transformer.inverse_transform(X_gen[:, :n_numerical_features])
        X_cat_generated = D.cat_transformer.inverse_transform(X_gen[:, n_numerical_features:])
        
        numerical_feature_columns = raw_config['data']['x']['numerical_feature_columns']
        categorical_feature_columns = raw_config['data']['x']['categorical_feature_columns']
        

        epoch = int(best_model_path.split('/')[-2].split('_')[-1])
        sample_dir = f"{raw_config['sample']['main']['sample_dir']}"
        epoch_dir = f'{sample_dir}/epochs_%06d_random_generated' % epoch
        os.makedirs(epoch_dir, exist_ok=True)


        X_num_train_boxcox = pd.read_csv(f"{raw_config['data']['real_data_dir']}/X_num_train.csv",index_col=0)
        X_num_generated_boxcox = pd.DataFrame(X_num_generated,columns=numerical_feature_columns)
        X_num_generated_boxcox.to_csv(f'{epoch_dir}/X_num_generated_boxcox.csv',index=False)
        
        lambdas_scalers = joblib.load(f"{raw_config['data']['lambdas_scalers_path']}")
        lambdas = lambdas_scalers['lambdas']
        scalers = lambdas_scalers['scalers']

        X_num_train_reverse = inverse_cox_box_scale_transform(X_num_train_boxcox, lambdas, scalers)
        X_num_generated_reverse = inverse_cox_box_scale_transform(X_num_generated_boxcox, lambdas, scalers)

        X_num_train_boxcox.to_csv(f'{epoch_dir}/X_num_train.csv',index=False)
        X_num_train_reverse.to_csv(f'{epoch_dir}/X_num_train_reverse.csv',index=False)
        X_num_generated_reverse.to_csv(f'{epoch_dir}/X_num_generated_reverse.csv',index=False)

        label_column = raw_config['data']['y']['label_column']
        X_generated = pd.concat([
            pd.DataFrame(X_num_generated, columns = numerical_feature_columns),
            pd.DataFrame(X_cat_generated, columns = categorical_feature_columns)
        ],axis=1)
        y_generated = pd.DataFrame(y_gen, columns=[label_column])

        X_generated.to_csv(f'{epoch_dir}/X_.csv',index=False)
        y_generated.to_csv(f'{epoch_dir}/y_.csv',index=False)















