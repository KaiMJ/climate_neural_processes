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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            if self.x_scaler is not None:
                if type(self.x_scaler) == np.ndarray:
                    x = x / self.x_scaler
                else:
                    x = self.x_scaler.transform(x)
            if self.y_scaler is not None:
                if type(self.y_scaler) == np.ndarray:
                    y = y / self.y_scaler
                else:
                    y = self.y_scaler.transform(y)
            x[np.isnan(x)] = 0
            y[np.isnan(y)] = 0

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x, y
    
    def __len__(self):
        return len(self.x_paths)

class MultiDataset(Dataset):
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
        self.l1_idxs = np.random.permutation(len(l1_x_paths))
        self.l2_idxs = self.l1_idxs
        if not nested:
            self.l2_idxs = np.random.permutation(len(l2_x_paths))

    def __getitem__(self, idx):
        l1_idx = self.l1_idxs[idx]
        l2_idx = self.l2_idxs[idx]

        l1_x = np.load(self.l1_x_paths[l1_idx], mmap_mode='r')
        l1_y = np.load(self.l1_y_paths[l1_idx], mmap_mode='r')
        l2_x = np.load(self.l2_x_paths[l2_idx], mmap_mode='r')
        l2_y = np.load(self.l2_y_paths[l2_idx], mmap_mode='r')

        if self.variables is not None:
            l1_y = l1_y[:, :self.variables[0]]
            l2_y = l2_y[:, :self.variables[1]]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
                
            if self.l1_x_scaler is not None:
                if type(self.l1_x_scaler) == np.ndarray:
                    l1_x = l1_x / self.l1_x_scaler
                else:
                    l1_x = self.l1_x_scaler.transform(l1_x)
            if self.l1_y_scaler is not None:
                if type(self.l1_y_scaler) == np.ndarray:
                    l1_y = l1_y / self.l1_y_scaler
                else:
                    l1_y = self.l1_y_scaler.transform(l1_y)
            if self.l2_x_scaler is not None:
                if type(self.l2_x_scaler) == np.ndarray:
                    l2_x = l2_x / self.l2_x_scaler
                else:
                    l2_x = self.l2_x_scaler.transform(l2_x)
            if self.l2_y_scaler is not None:
                if type(self.l2_y_scaler) == np.ndarray:
                    l2_y = l2_y / self.l2_y_scaler
                else:
                    l2_y = self.l2_y_scaler.transform(l2_y)
            l1_x[np.isnan(l1_x)] = 0
            l1_y[np.isnan(l1_y)] = 0
            l2_x[np.isnan(l2_x)] = 0
            l2_y[np.isnan(l2_y)] = 0

        l1_x = torch.from_numpy(l1_x).float()
        l1_y = torch.from_numpy(l1_y).float()
        l2_x = torch.from_numpy(l2_x).float()
        l2_y = torch.from_numpy(l2_y).float()

        return l1_x, l1_y, l2_x, l2_y

    def __len__(self):
        return len(self.l1_x_paths)
