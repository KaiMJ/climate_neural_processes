"""
    Make_dir function,
    MinMax and Standard scaler,
    Sorting functions,
    Logger,
    Save and load progress
"""

import torch
import datetime
import logging
import os
import warnings
import numpy as np
import sys

device = torch.device("cuda")

def make_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)


class MinMaxScaler:
    """
    Min max 2d feature.
    """

    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, x):
        if self.min is None:
            self.min = np.min(x, axis=0)
            self.max = np.max(x, axis=0)
        else:
            self.min = np.min(np.vstack([self.min, np.min(x, axis=0)]), axis=0)
            self.max = np.max(np.vstack([self.max, np.max(x, axis=0)]), axis=0)

    def transform(self, data):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            out = (data - self.min) / (self.max - self.min)
            out[np.isinf(out) | np.isnan(out)] = 0
            return out

    def inverse_transform(self, data):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            out = data * (self.max - self.min) + self.min
            out[np.isinf(out) | np.isnan(out)] = 0
            return out

class StandardScaler:
    """
    Standard Scaler
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.total = None

    def fit(self, x):
        if self.mean is None:
            self.mean = np.mean(x, axis=0)
            self.std = np.sqrt(np.sum(np.square(x - self.mean), axis=0) / len(x))
            self.total = len(x)
        else:
            # Welford's online algorithm
            var = np.square(self.std) * self.total
            assert len(x.shape) == 2, "2D shape needed"
            for i in range(len(x)):
                self.total += 1
                delta = x[i] - self.mean
                self.mean += delta / self.total
                delta2 = x[i] - self.mean
                var += delta * delta2
            self.std = np.sqrt(var / self.total)

    def transform(self, data):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            out = (data - self.mean) / self.std
            out[np.isinf(out) | np.isnan(out)] = 0
            return out

    def inverse_transform(self, data):
        return data * self.std + self.mean


def sort_fn(filename):
    date_string = filename[-14:-4]
    datetime_object = datetime.datetime.strptime(date_string, "%Y-%m-%d")
    return datetime_object


# Change level for "DEBUG"
def get_logger(log_dir=".", name="log", level="INFO", log_filename="info.log"):
    make_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Reset
    logger.handlers = []
    # Format message
    formatter = logging.Formatter('%(asctime)s - %(name)s: %(message)s')
    # Handler for file output
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Handler for terminal output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
