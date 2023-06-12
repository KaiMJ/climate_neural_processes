import numpy as np
import warnings
import datetime

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
            out[np.isinf(out) | np.isnan(out) | (out > 1e8)] = 0
            return out

    def inverse_transform(self, data):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            out = data * (self.max - self.min) + self.min
            out[np.isinf(out) | np.isnan(out) | (out > 1e8)] = 0
            return out


def sort_fn(filename):
    date_string = filename[-14:-4]
    datetime_object = datetime.datetime.strptime(date_string, "%Y-%m-%d")
    return datetime_object