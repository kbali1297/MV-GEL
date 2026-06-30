import os
import random
import numpy as np


def prepare_split(dataset_log_path):

    #cad_folder_paths = sorted(os.listdir(dataset_path), key= lambda x: int(x)) 
    with open(dataset_log_path, 'r') as fread:
        cad_folder_paths = fread.readlines()
    split_ratio = {'train': 0.9, 'val': 0.1}
    random_idxs = np.random.choice(np.arange(0, len(cad_folder_paths)), 
                                   len(cad_folder_paths),
                                   replace=False)
    
    train_idxs = random_idxs[:int(split_ratio['train'] * len(random_idxs))]
    val_idxs = random_idxs[int(split_ratio['train'] * len(random_idxs)):]
    cad_folder_paths = np.array(cad_folder_paths)
    train_folders, val_folders = cad_folder_paths[train_idxs], cad_folder_paths[val_idxs] 
    
    for split, split_folders in zip(['train', 'val'], [train_folders, val_folders]):
        fname = f"{split}_dataset.log" 
        with open(fname, 'w') as fwrite:
            for folder in split_folders:
                fwrite.write(f'{folder}')

if __name__ == '__main__':
    DATA_ROOT = os.environ.get(
        "MVGEL_ROOT", os.path.dirname(os.path.abspath(__file__)))
    dataset_log_path = os.path.join(DATA_ROOT, 'dataset.log')
    np.random.seed(42)
    prepare_split(dataset_log_path)


        