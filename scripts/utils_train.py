import lib

# ------------------------------------------------------------------------

# from tab_ddpm.modules import MLPDiffusion, ResNetDiffusion

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# def to_good_ohe(ohe, X):
#     indices = np.cumsum([0] + ohe._n_features_outs)
#     Xres = []
#     for i in range(1, len(indices)):
#         x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
#         t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
#         Xres.append(np.where(t >= 0, 1, 0))
#     return np.hstack(Xres)

def make_dataset(
    real_data_dir: str,
    T: lib.Transformations, # Transformation 参数
    # change_val: bool
):
    X_cat = {} 
    X_num = {} 
    y = {} 
    split_index = {}
    for split in ['train', 'val', 'test']:
        X_num_t, X_cat_t, y_t, split_index_t = lib.read_pure_data(real_data_dir, -1, split, True) #
        X_num[split] = X_num_t # 集合 X_num
        X_cat[split] = X_cat_t  # 集合 X_cat
        y[split] = y_t  # 集合 y
        split_index[split] = split_index_t  # 集合 y

    D = lib.Dataset( # 自定义特殊的 data class，用于这套算法的数据整合
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=lib.TaskType('binclass'), 
        n_classes=None # 无
    )

    # if change_val: # 重新分配 train 和 valid
    #     D = lib.change_val(D)
    #     print('changed the train and val sets')
    
    return lib.transform_dataset(D, T, split_index, None) # jump to data.py #460 # continue数据的归一化在这里面；category数据的encoding在这里面，同时，还保存了归一化和encoder的function，放入dataset(D) 中, 以便于将来 “逆” 归一化 / (de)encoding


def update_ema(target_params, source_params, rate=0.999): # from lIne 73;  targ=targ×rate+src×(1−rate)
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)

