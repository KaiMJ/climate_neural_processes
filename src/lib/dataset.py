import numpy as np
import warnings
import torch
from torch.utils.data import Dataset

class l2Dataset(Dataset):
    def __init__(self, x_paths, y_paths, x_scaler=None, y_scaler=None, variables=None):
        self.x_paths = x_paths
        self.y_paths = y_paths
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.variables = variables

    def __getitem__(self, idx):
        x = np.load(self.x_paths[idx], mmap_mode='r')
        y = np.load(self.y_paths[idx], mmap_mode='r')

        if self.variables is not None:
            y = y[:, :self.variables]
        if self.x_scaler is not None:
            x = self.x_scaler.transform(x)
        if self.y_scaler is not None:
            y = self.y_scaler.transform(y)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x, y
    
    def __len__(self):
        return len(self.x_paths)

class MutliDataset(Dataset):
    def __init__(self, l1_x_paths, l1_y_paths, l2_x_paths, l2_y_paths, \
                l1_x_scaler=None, l1_y_scaler=None, l2_x_scaler=None, l2_y_scaler=None, \
                nested=True, variables=None):
        self.l1_x_paths = l1_x_paths
        self.l1_y_paths = l1_y_paths
        self.l2_x_paths = l2_x_paths
        self.l2_y_paths = l2_y_paths
        self.l1_x_scaler = l1_x_scaler
        self.l1_y_scaler = l1_y_scaler
        self.l2_x_scaler = l2_x_scaler
        self.l2_y_scaler = l2_y_scaler
        
        self.nested = nested
        self.variables = variables
        self.idxs = np.random.permutation(len(l1_x_paths))
        if not nested:
            self.l2_idx = np.random.permutation(len(l2_x_paths))

    def __getitem__(self, idx):
        l1_idx = self.idxs[idx]
        if self.nested:
            l2_idx = l1_idx % len(self.l2_x_paths)
        else:
            # pick random l2 idx
            if len(self.l2_idx) == 0:
                self.l2_idx = np.random.permutation(len(self.l2_x_paths))
            l2_idx = self.l2_idx[0]
            self.l2_idx = self.l2_idx[1:]

        l1_x = np.load(self.l1_x_paths[l1_idx])
        l1_y = np.load(self.l1_y_paths[l1_idx])
        l2_x = np.load(self.l2_x_paths[l2_idx])
        l2_y = np.load(self.l2_y_paths[l2_idx])

        if self.variables is not None:
            l1_y = l1_y[:, :self.variables[0]]
            l2_y = l2_y[:, :self.variables[1]]

        if self.l1_x_scaler is not None:
            l1_x = self.l1_x_scaler.transform(l1_x)
        if self.l1_y_scaler is not None:
            l1_y = self.l1_y_scaler.transform(l1_y)
        if self.l2_x_scaler is not None:
            l2_x = self.l2_x_scaler.transform(l2_x)
        if self.l2_y_scaler is not None:
            l2_y = self.l2_y_scaler.transform(l2_y)

        l1_x = torch.from_numpy(l1_x).float()
        l1_y = torch.from_numpy(l1_y).float()
        l2_x = torch.from_numpy(l2_x).float()
        l2_y = torch.from_numpy(l2_y).float()

        return l1_x, l1_y, l2_x, l2_y

    def __len__(self):
        return len(self.l1_x_paths)

class TransformerDataset(Dataset):
    def __init__(self, x_paths, y_paths, x_scaler=None, y_scaler=None):
        self.x_paths = x_paths
        self.y_paths = y_paths
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def __getitem__(self, idx):
        x = np.load(self.x_paths[idx], mmap_mode='r')
        y = np.load(self.y_paths[idx], mmap_mode='r')

        if self.x_scaler is not None:
            x = self.x_scaler.transform(x)
        if self.y_scaler is not None:
            y = self.y_scaler.transform(y)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x, y, idx

    def __len__(self):
        return len(self.x_paths)
