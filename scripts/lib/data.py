import hashlib
from collections import Counter
from copy import deepcopy
from dataclasses import astuple, dataclass, replace
from importlib.resources import path
from pathlib import Path
from typing import Any, Literal, Optional, Union, cast, Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import sklearn.preprocessing
import torch
import os
from category_encoders import LeaveOneOutEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from . import env, util
from .metrics import calculate_metrics as calculate_metrics_
from .util import TaskType, load_json

ArrayDict = Dict[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]


CAT_MISSING_VALUE = '__nan__'
CAT_RARE_VALUE = '__rare__'
Normalization = Literal['standard', 'quantile', 'minmax']
NumNanPolicy = Literal['drop-rows', 'mean']
CatNanPolicy = Literal['most_frequent']
CatEncoding = Literal['one-hot', 'counter']
YPolicy = Literal['default']


class StandardScaler1d(StandardScaler):
    def partial_fit(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().partial_fit(X[:, None], *args, **kwargs)

    def transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().transform(X[:, None], *args, **kwargs).squeeze(1)

    def inverse_transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().inverse_transform(X[:, None], *args, **kwargs).squeeze(1)


def get_category_sizes(X: Union[torch.Tensor, np.ndarray]) -> List[int]:
    XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
    return [len(set(x)) for x in XT]


@dataclass(frozen=False)
class Dataset:
    X_num: Optional[ArrayDict]
    X_cat: Optional[ArrayDict]
    y: ArrayDict
    y_info: Dict[str, Any]
    task_type: TaskType
    n_classes: Optional[int]

    @classmethod
    def from_dir(cls, dir_: Union[Path, str]) -> 'Dataset':
        dir_ = Path(dir_)
        splits = [k for k in ['train', 'val', 'test'] if dir_.joinpath(f'y_{k}.npy').exists()]

        def load(item) -> ArrayDict:
            return {
                x: cast(np.ndarray, np.load(dir_ / f'{item}_{x}.npy', allow_pickle=True))  # type: ignore[code]
                for x in splits
            }

        if Path(dir_ / 'info.json').exists():
            info = util.load_json(dir_ / 'info.json')
        else:
            info = None
        return Dataset(
            load('X_num') if dir_.joinpath('X_num_train.npy').exists() else None,
            load('X_cat') if dir_.joinpath('X_cat_train.npy').exists() else None,
            load('y'),
            {},
            TaskType(info['task_type']),
            info.get('n_classes'),
        )

    @property
    def is_binclass(self) -> bool:
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num['train'].shape[1]

    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat['train'].shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, part: Optional[str]) -> int:
        return sum(map(len, self.y.values())) if part is None else len(self.y[part])

    @property
    def nn_output_dim(self) -> int:
        if self.is_multiclass:
            assert self.n_classes is not None
            return self.n_classes
        else:
            return 1

    def get_category_sizes(self, part: str) -> List[int]:
        return [] if self.X_cat is None else get_category_sizes(self.X_cat[part])

    def calculate_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        prediction_type: Optional[str],
    ) -> Dict[str, Any]:
        metrics = {
            x: calculate_metrics_(
                self.y[x], predictions[x], self.task_type, prediction_type, self.y_info
            )
            for x in predictions
        }
        if self.task_type == TaskType.REGRESSION:
            score_key = 'rmse'
            score_sign = -1
        else:
            score_key = 'accuracy'
            score_sign = 1
        for part_metrics in metrics.values():
            part_metrics['score'] = score_sign * part_metrics[score_key]
        return metrics

    def calculate_metrics_recall(
        self,
        X_train_fake_shape, y_fake_sum,
        X_train_val_real_shape, y_real_sum,
        X_test_real_shape, y_test_sum,
        df_metrics_both,
    ) -> Dict[str, Any]:
        
        report_metrics = {k1: {} for k1 in ['train','val','test']}
        for k1 in report_metrics.keys():
            report_metrics[k1] = {k2: {} for k2 in ['0','1','accuracy','macro avg','weighted avg','roc_auc','score']}
            for k2 in ['0','1','macro avg','weighted avg']:
                report_metrics[k1][k2] = {k3: {} for k3 in ['precision','recall','f1-score','support']}

        score_key = 'recall'

        # train 空白
        report_metrics['train']['0']['precision'] = np.nan
        report_metrics['train']['0']['recall'] = np.nan
        report_metrics['train']['0']['f1-score'] = np.nan
        report_metrics['train']['0']['support'] = np.nan
        report_metrics['train']['1']['precision'] = np.nan
        report_metrics['train']['1']['recall'] = np.nan
        report_metrics['train']['1']['f1-score'] = np.nan
        report_metrics['train']['1']['support'] = np.nan
        report_metrics['train']['accuracy'] = np.nan
        report_metrics['train']['macro avg']['precision'] = np.nan
        report_metrics['train']['macro avg']['recall'] = np.nan
        report_metrics['train']['macro avg']['f1-score'] = np.nan
        report_metrics['train']['macro avg']['support'] = np.nan
        report_metrics['train']['weighted avg']['precision'] = np.nan
        report_metrics['train']['weighted avg']['recall'] = np.nan
        report_metrics['train']['weighted avg']['f1-score'] = np.nan
        report_metrics['train']['weighted avg']['support'] = np.nan
        report_metrics['train']['roc_auc'] = np.nan
        report_metrics['train']['score'] = np.nan
        # real model 结果
        report_metrics['val']['0']['precision'] = np.nan
        report_metrics['val']['0']['recall'] = np.nan
        report_metrics['val']['0']['f1-score'] = np.nan
        report_metrics['val']['0']['support'] = np.nan
        report_metrics['val']['1']['precision'] = df_metrics_both.loc[df_metrics_both['model'].apply(lambda x: 'real' in x), "precision"][0]
        report_metrics['val']['1']['recall'] = df_metrics_both.loc[df_metrics_both['model'].apply(lambda x: 'real' in x), "recall"][0]
        report_metrics['val']['1']['f1-score'] =  df_metrics_both.loc[df_metrics_both['model'].apply(lambda x: 'real' in x), "micro_f1"][0]
        report_metrics['val']['1']['support'] = X_train_val_real_shape[0]
        report_metrics['val']['accuracy'] = np.nan
        report_metrics['val']['macro avg']['precision'] = np.nan
        report_metrics['val']['macro avg']['recall'] = np.nan
        report_metrics['val']['macro avg']['f1-score'] = np.nan
        report_metrics['val']['macro avg']['support'] = np.nan
        report_metrics['val']['weighted avg']['precision'] = np.nan
        report_metrics['val']['weighted avg']['recall'] = np.nan
        report_metrics['val']['weighted avg']['f1-score'] = np.nan
        report_metrics['val']['weighted avg']['support'] = np.nan
        report_metrics['val']['roc_auc'] = np.nan
        report_metrics['val']['score'] = report_metrics['val']['1'][score_key] 
        # fake model 结果
        report_metrics['test']['0']['precision'] = np.nan
        report_metrics['test']['0']['recall'] = np.nan
        report_metrics['test']['0']['f1-score'] = np.nan
        report_metrics['test']['0']['support'] = np.nan
        report_metrics['test']['1']['precision'] = df_metrics_both.loc[df_metrics_both['model'].apply(lambda x: 'fake' in x), "precision"].iloc[0]
        report_metrics['test']['1']['recall'] = df_metrics_both.loc[df_metrics_both['model'].apply(lambda x: 'fake' in x), "recall"].iloc[0]
        report_metrics['test']['1']['f1-score'] =  df_metrics_both.loc[df_metrics_both['model'].apply(lambda x: 'fake' in x), "micro_f1"].iloc[0]
        report_metrics['test']['1']['support'] = X_train_fake_shape[0]
        report_metrics['test']['accuracy'] = np.nan
        report_metrics['test']['macro avg']['precision'] = np.nan
        report_metrics['test']['macro avg']['recall'] = np.nan
        report_metrics['test']['macro avg']['f1-score'] = np.nan
        report_metrics['test']['macro avg']['support'] = np.nan
        report_metrics['test']['weighted avg']['precision'] = np.nan
        report_metrics['test']['weighted avg']['recall'] = np.nan
        report_metrics['test']['weighted avg']['f1-score'] = np.nan
        report_metrics['test']['weighted avg']['support'] = np.nan
        report_metrics['test']['roc_auc'] = np.nan
        # report_metrics['test']['score'] = report_metrics['test']['1'][score_key] 
        report_metrics['test']['score'] = report_metrics['test']['1']['recall'] + report_metrics['test']['1']['precision']

        return report_metrics
    
def num_process_nans(dataset: Dataset, policy: Optional[NumNanPolicy]) -> Dataset: 
    assert dataset.X_num is not None
    nan_masks = {k: np.isnan(v) for k, v in dataset.X_num.items()} 
    if not any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        assert policy is None
        return dataset

    assert policy is not None
    if policy == 'drop-rows':
        valid_masks = {k: ~v.any(1) for k, v in nan_masks.items()}
        assert valid_masks[
            'test'
        ].all(), 'Cannot drop test rows, since this will affect the final metrics.'
        new_data = {}
        for data_name in ['X_num', 'X_cat', 'y']:
            data_dict = getattr(dataset, data_name)
            if data_dict is not None:
                new_data[data_name] = {
                    k: v[valid_masks[k]] for k, v in data_dict.items()
                }
        dataset = replace(dataset, **new_data)
    elif policy == 'mean':
        new_values = np.nanmean(dataset.X_num['train'], axis=0)
        X_num = deepcopy(dataset.X_num)
        for k, v in X_num.items():
            num_nan_indices = np.where(nan_masks[k])
            v[num_nan_indices] = np.take(new_values, num_nan_indices[1])
        dataset = replace(dataset, X_num=X_num)
    else:
        assert util.raise_unknown('policy', policy)
    return dataset


def normalize(
    X: ArrayDict, normalization: Normalization, seed: Optional[int], return_normalizer : bool = False
) -> ArrayDict:
    X_train = X['train']
    if normalization == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == 'minmax':
        normalizer = sklearn.preprocessing.MinMaxScaler()
    elif normalization == 'quantile':
        n = 1e99
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=n,
            subsample=n,
            random_state=seed,
        )
    else:
        util.raise_unknown('normalization', normalization)

    normalizer.fit(X_train)

    if return_normalizer:
        return {k: normalizer.transform(v) for k, v in X.items()}, normalizer 
    
    return {k: normalizer.transform(v) for k, v in X.items()}


def cat_process_nans(X: ArrayDict, policy: Optional[CatNanPolicy]) -> ArrayDict:
    assert X is not None
    nan_masks = {k: v == CAT_MISSING_VALUE for k, v in X.items()}
    if any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        if policy is None:
            X_new = X
        elif policy == 'most_frequent':
            imputer = SimpleImputer(missing_values=CAT_MISSING_VALUE, strategy=policy) 
            imputer.fit(X['train'])
            X_new = {k: cast(np.ndarray, imputer.transform(v)) for k, v in X.items()}
        else:
            util.raise_unknown('categorical NaN policy', policy)
    else:
        assert policy is None
        X_new = X
    return X_new


def cat_drop_rare(X: ArrayDict, min_frequency: float) -> ArrayDict:
    assert 0.0 < min_frequency < 1.0
    min_count = round(len(X['train']) * min_frequency)
    X_new = {x: [] for x in X}
    for column_idx in range(X['train'].shape[1]):
        counter = Counter(X['train'][:, column_idx].tolist())
        popular_categories = {k for k, v in counter.items() if v >= min_count}
        for part in X_new:
            X_new[part].append(
                [
                    (x if x in popular_categories else CAT_RARE_VALUE)
                    for x in X[part][:, column_idx].tolist()
                ]
            )
    return {k: np.array(v).T for k, v in X_new.items()}


def cat_encode(
    X: ArrayDict,
    encoding: Optional[CatEncoding],
    y_train: Optional[np.ndarray],
    seed: Optional[int],
    return_encoder : bool = False
) -> Tuple[ArrayDict, bool, Optional[Any]]:  # (X, is_converted_to_numerical)
    if encoding != 'counter':
        y_train = None

    # Step 1. Map strings to 0-based ranges

    if encoding is None:
        unknown_value = np.iinfo('int64').max - 3
        oe = sklearn.preprocessing.OrdinalEncoder( 
            handle_unknown='use_encoded_value',  # type: ignore[code]
            unknown_value=unknown_value,  # type: ignore[code]
            dtype='int64',  # type: ignore[code]
        ).fit(X['train'])
        encoder = make_pipeline(oe)
        encoder.fit(X['train'])
        X = {k: encoder.transform(v) for k, v in X.items()} 
        max_values = X['train'].max(axis=0)
        for part in X.keys():
            if part == 'train': continue
            for column_idx in range(X[part].shape[1]): 
                X[part][X[part][:, column_idx] == unknown_value, column_idx] = (
                    max_values[column_idx] + 1
                )
        if return_encoder:
            return (X, False, encoder)
        return (X, False)

    # Step 2. Encode.

    elif encoding == 'one-hot':
        ohe = sklearn.preprocessing.OneHotEncoder(
            handle_unknown='ignore', sparse=False, dtype=np.float32 # type: ignore[code]
        )
        encoder = make_pipeline(ohe)

        # encoder.steps.append(('ohe', ohe))
        encoder.fit(X['train'])
        X = {k: encoder.transform(v) for k, v in X.items()}
    elif encoding == 'counter':
        assert y_train is not None
        assert seed is not None
        loe = LeaveOneOutEncoder(sigma=0.1, random_state=seed, return_df=False)
        encoder.steps.append(('loe', loe))
        encoder.fit(X['train'], y_train)
        X = {k: encoder.transform(v).astype('float32') for k, v in X.items()}  # type: ignore[code]
        if not isinstance(X['train'], pd.DataFrame):
            X = {k: v.values for k, v in X.items()}  # type: ignore[code]
    else:
        util.raise_unknown('encoding', encoding)
    
    if return_encoder:
        return X, True, encoder # type: ignore[code]
    return (X, True)


def build_target(
    y: ArrayDict, policy: Optional[YPolicy], task_type: TaskType
) -> Tuple[ArrayDict, Dict[str, Any]]:
    info: Dict[str, Any] = {'policy': policy}
    if policy is None:
        pass
    elif policy == 'default':
        if task_type == TaskType.REGRESSION:
            mean, std = float(y['train'].mean()), float(y['train'].std())
            y = {k: (v - mean) / std for k, v in y.items()}
            info['mean'] = mean
            info['std'] = std
    else:
        util.raise_unknown('policy', policy)
    return y, info


@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Optional[Normalization] = None
    num_nan_policy: Optional[NumNanPolicy] = None
    cat_nan_policy: Optional[CatNanPolicy] = None
    cat_min_frequency: Optional[float] = None
    cat_encoding: Optional[CatEncoding] = None
    y_policy: Optional[YPolicy] = 'default'


def transform_dataset(
    dataset: Dataset, # D
    transformations: Transformations, # T
    split_index,
    cache_dir: Optional[Path],
    return_transforms: bool = False
) -> Dataset:
    X_num_transformed, num_transformer = normalize(
        dataset.X_num,
        transformations.normalization,
        transformations.seed,
        return_normalizer=True
    )
    
    X_cat_transformed, is_num, cat_transformer = cat_encode( # encoding category data
        dataset.X_cat,
        transformations.cat_encoding,
        dataset.y['train'],
        transformations.seed,
        return_encoder=True
    )

    y = dataset.y
    y_info = {'policy': 'default'}

    dataset = replace(dataset, X_num=X_num_transformed, X_cat=X_cat_transformed, y=y, y_info=y_info)
    dataset.num_transformer = num_transformer
    dataset.cat_transformer = cat_transformer
    dataset.split_index = split_index

    return dataset



def build_dataset(
    path: Union[str, Path],
    transformations: Transformations,
    cache: bool
) -> Dataset:
    path = Path(path)
    dataset = Dataset.from_dir(path)
    return transform_dataset(dataset, transformations, path if cache else None)


def prepare_tensors(
    dataset: Dataset, device: Union[str, torch.device]
) -> Tuple[Optional[TensorDict], Optional[TensorDict], TensorDict]:
    X_num, X_cat, Y = (
        None if x is None else {k: torch.as_tensor(v) for k, v in x.items()}
        for x in [dataset.X_num, dataset.X_cat, dataset.y]
    )
    if device.type != 'cpu':
        X_num, X_cat, Y = (
            None if x is None else {k: v.to(device) for k, v in x.items()}
            for x in [X_num, X_cat, Y]
        )
    assert X_num is not None
    assert Y is not None
    if not dataset.is_multiclass:
        Y = {k: v.float() for k, v in Y.items()}
    return X_num, X_cat, Y

class TabDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset : Dataset, split : Literal['train', 'val', 'test']
    ):
        super().__init__()
        
        self.X_num = torch.from_numpy(dataset.X_num[split]) if dataset.X_num is not None else None
        self.X_cat = torch.from_numpy(dataset.X_cat[split]) if dataset.X_cat is not None else None
        self.y = torch.from_numpy(dataset.y[split])

        assert self.y is not None
        assert self.X_num is not None or self.X_cat is not None 

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        out_dict = {
            'y': self.y[idx].long() if self.y is not None else None,
        }

        x = np.empty((0,))
        if self.X_num is not None:
            x = self.X_num[idx]
        if self.X_cat is not None:
            x = torch.cat([x, self.X_cat[idx]], dim=0)
        return x.float(), out_dict

def prepare_dataloader(
    dataset : Dataset,
    split : str,
    batch_size: int,
):

    torch_dataset = TabDataset(dataset, split)
    loader = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=1,
    )
    while True:
        yield from loader

def prepare_torch_dataloader(
    dataset : Dataset,
    split : str,
    shuffle : bool,
    batch_size: int,
) -> torch.utils.data.DataLoader:

    torch_dataset = TabDataset(dataset, split)
    loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)

    return loader

def dataset_from_csv(paths : Dict[str, str], cat_features, target, T):
    assert 'train' in paths
    y = {}
    X_num = {}
    X_cat = {} if len(cat_features) else None
    for split in paths.keys():
        df = pd.read_csv(paths[split])
        y[split] = df[target].to_numpy().astype(float)
        if X_cat is not None:
            X_cat[split] = df[cat_features].to_numpy().astype(str)
        X_num[split] = df.drop(cat_features + [target], axis=1).to_numpy().astype(float)

    dataset = Dataset(X_num, X_cat, y, {}, None, len(np.unique(y['train'])))
    return transform_dataset(dataset, T, None)

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

def prepare_fast_dataloader(
    D : Dataset,
    split : str,
    batch_size: int
):
    if D.X_cat is not None:
        if D.X_num is not None:
            X = torch.from_numpy(np.concatenate([D.X_num[split], D.X_cat[split]], axis=1)).float()
        else:
            X = torch.from_numpy(D.X_cat[split]).float()
    else:
        X = torch.from_numpy(D.X_num[split]).float()
    y = torch.from_numpy(D.y[split])
    dataloader = FastTensorDataLoader(X, y, batch_size=batch_size, shuffle=(split=='train'))
    while True:
        yield from dataloader

def prepare_fast_torch_dataloader(
    D : Dataset,
    split : str,
    batch_size: int
):
    if D.X_cat is not None:
        X = torch.from_numpy(np.concatenate([D.X_num[split], D.X_cat[split]], axis=1)).float()
    else:
        X = torch.from_numpy(D.X_num[split]).float()
    y = torch.from_numpy(D.y[split])
    dataloader = FastTensorDataLoader(X, y, batch_size=batch_size, shuffle=(split=='train'))
    return dataloader

def round_columns(X_real, X_synth, columns):
    for col in columns:
        uniq = np.unique(X_real[:,col])
        dist = cdist(X_synth[:, col][:, np.newaxis].astype(float), uniq[:, np.newaxis].astype(float))
        X_synth[:, col] = uniq[dist.argmin(axis=1)]
    return X_synth

def concat_features(D : Dataset):
    if D.X_num is None:
        assert D.X_cat is not None
        X = {k: pd.DataFrame(v, columns=range(D.n_features)) for k, v in D.X_cat.items()}
    elif D.X_cat is None:
        assert D.X_num is not None
        X = {k: pd.DataFrame(v, columns=range(D.n_features)) for k, v in D.X_num.items()}
    else:
        X = {
            part: pd.concat(
                [
                    pd.DataFrame(D.X_num[part], columns=range(D.n_num_features)),
                    pd.DataFrame(
                        D.X_cat[part],
                        columns=range(D.n_num_features, D.n_features),
                    ),
                ],
                axis=1,
            )
            for part in D.y.keys()
        }

    return X

def concat_to_pd(X_num, X_cat, y):
    if X_num is None:
        return pd.concat([
            pd.DataFrame(X_cat, columns=list(range(X_cat.shape[1]))),
            pd.DataFrame(y, columns=['y'])
        ], axis=1)
    if X_cat is not None:
        return pd.concat([
            pd.DataFrame(X_num, columns=list(range(X_num.shape[1]))),
            pd.DataFrame(X_cat, columns=list(range(X_num.shape[1], X_num.shape[1] + X_cat.shape[1]))),
            pd.DataFrame(y, columns=['y'])
        ], axis=1)
    return pd.concat([
            pd.DataFrame(X_num, columns=list(range(X_num.shape[1]))),
            pd.DataFrame(y, columns=['y'])
        ], axis=1)

def read_pure_data(path, n_eval, split='train',verbose=False):
    if n_eval == -1:
        X_num = pd.read_csv(f'{path}/X_num_{split}.csv',index_col=0)
        X_cat = pd.read_csv(f'{path}/X_cat_{split}.csv',index_col=0)
        y = pd.read_csv(f'{path}/y_{split}.csv',index_col=0)
        split_index = y.index.tolist()
        X_num = np.array(X_num)
        X_cat = np.array(X_cat)
        y = np.array(y)
        # y = np.array(y).flatten()
    return X_num, X_cat, y, split_index

def read_changed_val(path, val_size=0.2):
    path = Path(path)
    X_num_train, X_cat_train, y_train = read_pure_data(path, 'train')
    X_num_val, X_cat_val, y_val = read_pure_data(path, 'val')
    is_regression = load_json(path / 'info.json')['task_type'] == 'regression'

    y = np.concatenate([y_train, y_val], axis=0)

    ixs = np.arange(y.shape[0])
    if is_regression:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777)
    else:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777, stratify=y)
    y_train = y[train_ixs]
    y_val = y[val_ixs]

    if X_num_train is not None:
        X_num = np.concatenate([X_num_train, X_num_val], axis=0)
        X_num_train = X_num[train_ixs]
        X_num_val = X_num[val_ixs]

    if X_cat_train is not None:
        X_cat = np.concatenate([X_cat_train, X_cat_val], axis=0)
        X_cat_train = X_cat[train_ixs]
        X_cat_val = X_cat[val_ixs]
    
    return X_num_train, X_cat_train, y_train, X_num_val, X_cat_val, y_val


def load_dataset_info(dataset_dir_name: str) -> Dict[str, Any]:
    path = Path("data/" + dataset_dir_name)
    info = util.load_json(path / 'info.json')
    info['size'] = info['train_size'] + info['val_size'] + info['test_size']
    info['n_features'] = info['n_num_features'] + info['n_cat_features']
    info['path'] = path
    return info

def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)

def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)