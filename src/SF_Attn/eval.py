import sys
sys.path.append('../')
from lib.loss import NegRLoss, mae_loss, mse_loss, norm_rmse_loss, mae_metric, nll_loss, kl_div
from lib.dataset import l2Dataset
from lib.model import SF_ATTN_Model as Model
from lib.utils import sort_fn, split_context_target
import numpy as np
import os

import torch
from torch.utils.data import DataLoader
import yaml
import glob
from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator():
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.config = yaml.safe_load(open(f"{dirpath}/saved_config.yaml"))
        self.init_dataloader()
        self.init_model()

    def init_model(self):
        self.device = torch.device('cuda')
        model_dict = torch.load(f"{self.dirpath}/best.pt", map_location=torch.device('cuda'))
        model = Model(model_dict['config']['model']).to(self.device)

        model.load_state_dict(model_dict['model'])
        model.eval()
        self.model = model

    def init_dataloader(self):
        l2_x_data = sorted(
            glob.glob(f"{self.config['data_dir']}/SPCAM5/inputs_*"), key=sort_fn)
        l2_y_data = sorted(
            glob.glob(f"{self.config['data_dir']}/SPCAM5/outputs_*"), key=sort_fn)

        split_n = int(365*0.8)
        l2_x_train = l2_x_data[:split_n]
        l2_y_train = l2_y_data[:split_n]
        l2_x_valid = l2_x_data[split_n:365]
        l2_y_valid = l2_y_data[split_n:365]
        l2_x_test = l2_x_data[365:]
        l2_y_test = l2_y_data[365:]

        x_scaler_minmax = np.load(f"../../notebooks/scalers/dataset_2_x_max.npy")
        self.y_scaler_minmax = np.load(f"../../notebooks/scalers/dataset_2_y_max.npy")

        train_dataset = l2Dataset(
            l2_x_train, l2_y_train, x_scaler=x_scaler_minmax, y_scaler=self.y_scaler_minmax, variables=26)
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config['batch_size'], shuffle=True, drop_last=False, num_workers=2, pin_memory=True)
        valid_dataset = l2Dataset(
            l2_x_valid, l2_y_valid, x_scaler=x_scaler_minmax, y_scaler=self.y_scaler_minmax, variables=26)
        self.valid_loader = DataLoader(
            valid_dataset, batch_size=self.config['batch_size'], shuffle=False, drop_last=False, num_workers=2, pin_memory=True)
        test_dataset = l2Dataset(
            l2_x_test, l2_y_test, x_scaler=x_scaler_minmax, y_scaler=self.y_scaler_minmax, variables=26)
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.config['batch_size'], shuffle=False, drop_last=False, num_workers=2, pin_memory=True)

    def get_metrics(self, loader):
        self.get_R_stats(loader)
        self.r = self.ssxym / np.sqrt(self.ssxm * self.ssym)
        return self.non_mae, self.nmae, self.r

    def forward_pass(self, data):
        with torch.no_grad():            
            non_y_all = None
            non_y_pred_all = None

            x_, y_ = data
            n_mb = 9
            # For Tuning
            rand_idxs = np.random.permutation(np.arange(x_.shape[1] / 144 * n_mb))
            x_ = x_[:, rand_idxs]
            y_ = y_[:, rand_idxs]

            # TODO: make mini batches random
            for bg_idx in range(n_mb):
                x = x_[:, bg_idx::n_mb]
                y = y_[:, bg_idx::n_mb]
                if eval is False:  # Random dropout during training
                    dropout = self.config['batch_dropout']
                    n = int(x.shape[1] * (1 - dropout))
                    idxs = np.random.permutation(np.arange(x.shape[1]))[:n]
                    x = x[:, idxs, :]
                    y = y[:, idxs, :]

                context_idxs, target_idxs = split_context_target(
                    x, self.config['context_percentage_low'], self.config['context_percentage_high'], axis=1)
                x_context = x[:, context_idxs, :].to(device)
                y_context = y[:, context_idxs].to(device)
                x_target = x[:, target_idxs, :].to(device)
                y_target = y[:, target_idxs].to(device)

                with torch.cuda.amp.autocast():
                    l2_output_mu, l2_output_cov, l2_z_mu_c, l2_z_cov_c, l2_z_mu_all, l2_z_cov_all = self.model(
                        x_context, y_context, x_target, y_target)

                    if type(self.y_scaler_minmax) == np.ndarray:
                        non_y_pred = self.y_scaler_minmax * l2_output_mu.squeeze().cpu().numpy()
                        non_y = self.y_scaler_minmax * y_target.squeeze().cpu().numpy()
                    else:
                        non_y_pred = self.y_scaler_minmax.inverse_transform(l2_output_mu.squeeze().cpu().numpy())
                        non_y = self.y_scaler_minmax.inverse_transform(y_target.squeeze().cpu().numpy())

                    non_y_all = np.concatenate((non_y_all, non_y), axis=0) if non_y_all is not None else non_y
                    non_y_pred_all = np.concatenate((non_y_pred_all, non_y_pred), axis=0) if non_y_pred_all is not None else non_y_pred

            return non_y, non_y_pred, context_idxs, target_idxs

    def get_loss(self):
        step_size = len(self.train_loader)
        train_event_file = sorted(glob.glob(os.path.join(
            self.dirpath, "runs/train/events.out.tfevents*")), key=os.path.getctime)[-1]
        valid_event_file = sorted(glob.glob(os.path.join(
            self.dirpath, "runs/valid/events.out.tfevents*")), key=os.path.getctime)[-1]
        train_acc = EventAccumulator(train_event_file)
        valid_acc = EventAccumulator(valid_event_file)
        train_acc.Reload()
        valid_acc.Reload()

        valid_values = [s.value for s in valid_acc.Scalars("non_mae")]
        n_epochs = len(valid_values)

        # Change each iteration to epochs
        train_values = np.array(
            [s.value for s in train_acc.Scalars("non_mae")])
        max_n = min(len(train_values) // step_size, n_epochs)
        train_values = train_values[:max_n * step_size]
        valid_values = valid_values[:max_n]
        train_values = train_values.reshape(-1, step_size).mean(axis=1)

        return train_values, valid_values

    def get_R_stats(self, loader):
        self._get_stats(loader)
        self.ssxm = 0
        self.ssxym = 0
        self.ssym = 0

        with torch.no_grad():
            for i, data in enumerate(tqdm(loader, total=len(loader))):
                non_y, non_y_pred, _, _ = self.forward_pass(data)
                self.ssxm += ((non_y - self.y_mean)**2).sum(0)
                self.ssym += ((non_y_pred - self.y_pred_mean)**2).sum(0)
                self.ssxym += ((non_y - self.y_mean) *
                               (non_y_pred - self.y_pred_mean)).sum(0)
            # Get average
            self.ssxm /= self.n_total
            self.ssym /= self.n_total
            self.ssxym /= self.n_total

    def _get_stats(self, loader):
        self.n_total = 0
        self.x_total = 0
        self.y_total = 0
        self.xy_total = 0
        self.x2_total = 0
        self.y2_total = 0
        self.y_mean = 0
        self.y_pred_mean = 0
        self.nmae = 0
        self.non_mae = 0
        self.y_max = 0


        with torch.no_grad():
            for i, data in enumerate(tqdm(loader, total=len(loader))):
                non_y, non_y_pred, _, _ = self.forward_pass(data)

                non_mae = mae_metric(non_y_pred, non_y, mean=False)

                self.y_mean += non_y.sum(axis=0)
                self.y_pred_mean += non_y_pred.sum(axis=0)
                self.non_mae += non_mae.sum(axis=0)
                self.n_total += non_y.shape[0]
                self.x_total += non_y.sum(axis=0)
                self.y_total += non_y_pred.sum(axis=0)
                self.x2_total += (non_y ** 2).sum(axis=0)
                self.y2_total += (non_y_pred ** 2).sum(axis=0)
                self.xy_total += (non_y_pred * non_y).sum(axis=0)

                self.y_max = np.maximum(self.y_max, np.abs(non_y).max(axis=0))

        self.y_mean /= self.n_total
        self.y_pred_mean /= self.n_total
        self.nmae /= self.n_total
        self.non_mae /= self.n_total
        self.nmae = self.non_mae / self.y_max
