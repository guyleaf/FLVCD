from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning import LightningDataModule
from pl_bolts.datamodules.async_dataloader import AsynchronousLoader
import torch
import h5py
from tqdm import tqdm
import numpy as np
import os
from typing import Optional

# helper functions
def get_bbc_file_paths(base_folder):
    return [base_folder + ("/" if base_folder[-1] != "/" else "") + i for i in os.listdir(base_folder) if i not in [".ipynb_checkpoints"]]

def get_bbc_max_seq_len(base_folder):
    file_paths = get_bbc_file_paths(base_folder)
    lengths = []
    for i in file_paths:
        with h5py.File(i, 'r') as f:
            lengths.append(f["features"].shape[0])
    return max(lengths)

def get_bbc_max_summary_len(base_folder):
    file_paths = get_bbc_file_paths(base_folder)
    lengths = []
    for i in file_paths:
        with h5py.File(i, 'r') as f:
            lengths.append(f["/labels/annotator_0"].shape[0] * 2)
    return max(lengths)

class BBCDataset(Dataset):
    '''
    Data format:
        * h5.attr['dataset_name'] = "BBC"
        * h5['/features']: Shape[seq_len, 1, d_model]
        * h5['/labels/annotator_0~4']: Shape[summary_len, 1], [(starting_frame, end_frame)]
    Output:
        * encoder: Shape[1, max_seq_length, d_model]
        * decoder: Shape[1, max_summary_length, d_model]
        * target: Shape[1, max_summary_length]
    PROCESS:
        * PADDING: -1
        * START: 1
        * END: 0
    '''
    def __init__(self, base_folder, file_paths, max_seq_length, max_summary_length, d_model):
        self.base_folder = base_folder
        self.file_paths = file_paths
        self.data_cache = {}
        self.max_seq_length = max_seq_length + 2
        self.max_summary_length = max_summary_length + 2
        self.d_model = d_model

    def prepare_data(self):
        for key in self.data_cache.keys():
            features = self.data_cache[key][1].squeeze(1)
            labels = self.data_cache[key][2].flatten()

            encoder = np.zeros((self.max_seq_length, self.d_model), dtype=np.float32)
            decoder = np.zeros((self.max_summary_length, self.d_model), dtype=np.float32)
            target = np.zeros((self.max_summary_length, ), dtype=np.int)
            weight = np.zeros((self.max_summary_length, ), dtype=np.float32)

            feature_length = features.shape[0]
            label_length = labels.shape[0]

            encoder[:feature_length] = features
            # END: label_length
            encoder[feature_length+1:] = float(-1)
            
            # START: 0
            decoder[0] = float(1)
            decoder[1:label_length+1] = features[labels]
            # END: label_length+1
            decoder[label_length+2:] = float(-1)

            target[:label_length] = labels
            # END: label_length+1
            weight[:label_length+1] = float(1)
            target[label_length:] = feature_length

            self.data_cache[key] = {
                "encoder": encoder[np.newaxis, :],
                "decoder": decoder[np.newaxis, :],
                "weight": weight[np.newaxis, :],
                "target": target[np.newaxis, :],
            }

    def _load_data(self, file_path):
        with h5py.File(file_path, 'r') as f:
            if f.attrs["dataset_name"] == "BBC":
                self.data_cache[file_path] = [f.attrs["dataset_name"], np.array(f['/features'][()]), np.array(f['/labels/annotator_0'][()])]
            else:
                self.data_cache[file_path] = [f.attrs["dataset_name"], np.array(f['/features'][()]), np.array(f['/labels'][()])]
    
    def get_name(self, index):
        return self.data_cache[self.file_paths[index]][0]
    
    def __len__(self):
        return len(self.file_paths)

    def clear_cache(self):
        self.data_cache = {}

    def load_data(self, incides):
        for i in tqdm(incides, total=len(incides), desc="Loading data"):
            self._load_data(self.file_paths[i])
        self.prepare_data()

    def get_labels(self, index):
        return self.data_cache[self.file_paths[index]][2]

    def __getitem__(self, index):
        data = self.data_cache[self.file_paths[index]]
        return data

class BBCDataModule(LightningDataModule):
    def __init__(self, max_seq_length, max_summary_length, d_model, base_folder, train_paths, val_paths, test_paths=None):
        super().__init__()
        self.base_folder = base_folder
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths
        self.max_seq_length = max_seq_length
        self.max_summary_length = max_summary_length
        self.d_model = d_model
    
    def setup(self, stage: Optional[str] = None):
        self.bbc_test = None
        if self.test_paths is not None:
            self.bbc_test = BBCDataset(self.base_folder, self.test_paths, self.max_seq_length, self.max_summary_length, self.d_model)
    
        self.bbc_train = BBCDataset(self.base_folder, self.train_paths, self.max_seq_length, self.max_summary_length, self.d_model)
        self.bbc_val = BBCDataset(self.base_folder, self.val_paths, self.max_seq_length, self.max_summary_length, self.d_model)
    
    def train_dataloader(self):
        self.bbc_train.load_data(range(len(self.train_paths)))
        return AsynchronousLoader(DataLoader(self.bbc_train, num_workers=0, pin_memory=True))
    
    def val_dataloader(self):
        self.bbc_val.load_data(range(len(self.val_paths)))
        return AsynchronousLoader(DataLoader(self.bbc_val, num_workers=0, pin_memory=True))
    
    def test_dataloader(self):
        if self.bbc_test is not None:
            self.bbc_test.load_data(range(len(self.test_paths)))
            return AsynchronousLoader(DataLoader(self.bbc_test, num_workers=0, pin_memory=True))
