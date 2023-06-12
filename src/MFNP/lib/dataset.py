import numpy as np
import torch
from torch.utils.data import Dataset

class MultiDataset(Dataset):
    def __init__(self, l2_x_paths, l2_y_paths, l1_x_paths, l1_y_paths, l2_x_scaler=None, l2_y_scaler=None, l1_x_scaler=None, l1_y_scaler=None, variables=None):
        self.l2_x_paths = l2_x_paths
        self.l2_y_paths = l2_y_paths
        self.l1_x_paths = l1_x_paths
        self.l1_y_paths = l1_y_paths
        self.l2_x_scaler = l2_x_scaler
        self.l2_y_scaler = l2_y_scaler
        self.l1_x_scaler = l1_x_scaler
        self.l1_y_scaler = l1_y_scaler
        self.l2_variables, self.l1_variables = variables

    def __getitem__(self, idx):
        l2_x = np.load(self.l2_x_paths[idx], mmap_mode='r')
        l2_y = np.load(self.l2_y_paths[idx], mmap_mode='r')
        l1_x = np.load(self.l1_x_paths[idx], mmap_mode='r')
        l1_y = np.load(self.l1_y_paths[idx], mmap_mode='r')

        if self.l2_variables is not None:
            l2_y = l2_y[:, :self.l2_variables]
            l1_y = l1_y[:, :self.l1_variables]
        if self.l2_x_scaler is not None:
            l2_x = self.l2_x_scaler.transform(l2_x)
        if self.l2_y_scaler is not None:
            l2_y = self.l2_y_scaler.transform(l2_y)
        if self.l1_x_scaler is not None:
            l1_x = self.l1_x_scaler.transform(l1_x)
        if self.l1_y_scaler is not None:
            l1_y = self.l1_y_scaler.transform(l1_y)
        l2_x = torch.from_numpy(l2_x).float()
        l2_y = torch.from_numpy(l2_y).float()
        l1_x = torch.from_numpy(l1_x).float()
        l1_y = torch.from_numpy(l1_y).float()
        return l2_x, l2_y, l1_x, l1_y
    
    def __len__(self):
        return len(self.l2_x_paths)
