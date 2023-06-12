import numpy as np
import torch
from torch.utils.data import Dataset
import warnings

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
