import lib

def make_dataset(
    real_data_dir: str,
    T: lib.Transformations, 
):
    X_cat = {} 
    X_num = {} 
    y = {} 
    split_index = {}
    for split in ['train', 'val', 'test']:
        X_num_t, X_cat_t, y_t, split_index_t = lib.read_pure_data(real_data_dir, -1, split, True) #
        X_num[split] = X_num_t 
        X_cat[split] = X_cat_t  
        y[split] = y_t  
        split_index[split] = split_index_t 

    D = lib.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=lib.TaskType('binclass'), 
        n_classes=None 
    )
   
    return lib.transform_dataset(D, T, split_index, None) 

