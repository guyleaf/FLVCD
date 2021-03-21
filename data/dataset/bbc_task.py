from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
from pl_bolts.datamodules.async_dataloader import AsynchronousLoader
import torch
import h5py
import tqdm
import numpy as np
import os
from typing import Optional

# helper function
def get_bbc_file_paths(base_folder):
    return [base_folder + ("/" if base_folder[-1] != "/" else "") + i for i in os.listdir(base_folder) if i not in [".ipynb_checkpoints"]]
def get_bbc_max_seq_len(base_folder):
    file_paths = get_bbc_file_paths(base_folder)
    lengths = []
    for i in file_paths:
        with h5py.File(i, 'r') as f:
            lengths.append(f["features"].shape[0])
    return max(lengths)

class BBCDataset(Dataset):
    '''
    Input is expected to be a list of sequence of frame features
    '''
    def __init__(self, base_folder, file_paths):
        self.base_folder = base_folder
        self.file_paths = file_paths
        self.data_cache = {}
    
    def _load_data(self, file_path):
        with h5py.File(self.base_folder + '/' + file_path, 'r') as f:
            if f.attrs["dataset_name"] == "BBC":
                self.data_cache[file_path] = [f.attrs["dataset_name"], np.array(f['/features']), np.array(f['/labels'].items())]
            else:
                self.data_cache[file_path] = [f.attrs["dataset_name"], np.array(f['/features']), np.array(f['/labels'])]
    
    def get_name(self, index):
        return self.data_cache[self.file_paths[index]][0]
    
    def __len__(self):
        return len(self.file_paths)

    def clear_cache(self):
        self.data_cache = {}

    def load_data(self, incides):
        for i in tqdm(incides, total=len(incides), desc="Loading data"):
            self._load_data(self.file_paths[i])

    def get_labels(self, index):
        return self.data_cache[self.file_paths[index]][2]

    def __getitem__(self, index):
        data = self.data_cache[self.file_paths[index]]
        return data[0], torch.from_numpy(data[1]), torch.from_numpy(data[2])

class BBCDataModule(LightningDataModule):
    def __init__(self, base_folder, train_paths, val_paths, test_paths=None):
        super().__init__()
        self.base_folder = base_folder
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths
    
    def setup(self, stage: Optional[str] = None):
        self.bbc_test = None
        if self.test_paths is not None:
            self.bbc_test = BBCDataset(self.base_folder, self.test_paths)
    
        self.bbc_train = BBCDataset(self.base_folder, self.train_paths)
        self.bbc_val = BBCDataset(self.base_folder, self.val_paths)
    
    def train_dataloader(self):
        self.bbc_train.load_data(range(len(self.train_paths)))
        return AsynchronousLoader(self.bbc_train)
    
    def val_dataloader(self):
        self.bbc_val.load_data(range(len(self.val_paths)))
        return AsynchronousLoader(self.bbc_val)
    
    def test_dataloader(self):
        if self.bbc_test is not None:
            self.bbc_test.load_data(range(len(self.test_paths)))
            return AsynchronousLoader(self.bbc_test)
