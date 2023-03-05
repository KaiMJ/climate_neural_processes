from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../src/')
import sys
from lib.dataset import *
from lib.loss import *
from lib.utils import *

import torch
from torch.utils.data import DataLoader
from SFNP.lib.model import Model as SFNP_Model
from transformer.lib.model import Model as Transformer_Model
from MFNP.model import Model as MFNP_Model
from SF_Attn.model import Model as SF_Attn_Model
import os, yaml, glob, dill
from tqdm import tqdm


def split_context_target(x, y, context_percentage_low, context_percentage_high):
    """Helper function to split randomly into context and target"""
    context_percentage = np.random.uniform(
        context_percentage_low, context_percentage_high)
    n_context = int(x.shape[1]*context_percentage)
    ind = np.arange(x.shape[1])
    mask = np.random.choice(ind, size=n_context, replace=False)
    others = np.delete(ind, mask)

    return mask, others

def split_context_target(x, y, context_percentage_low, context_percentage_high):
    """Helper function to split randomly into context and target"""
    context_percentage = np.random.uniform(
        context_percentage_low, context_percentage_high)
    n_context = int(x.shape[1]*context_percentage)
    ind = np.arange(x.shape[1])
    mask = np.random.choice(ind, size=n_context, replace=False)
    others = np.delete(ind, mask)

    return mask, others

class Evaluator():
    def __init__(self, dirpath, Model):
        self.dirpath = dirpath

        if Model == Transformer_Model:
            self.model_type = "forward"
        elif Model == MFNP_Model:
            self.model_type = "multi"
        elif Model == SF_Attn_Model:
            self.model_type = "attentive"
        else:
            self.model_type = "sfnp"

        self.config = yaml.safe_load(open(f"{dirpath}/saved_config.yaml"))
        self.init_dataloader()
        self.Model = Model
        self.init_model()

    def init_model(self):
        self.device = torch.device('cuda')
        model_dict = torch.load(f"{self.dirpath}/best.pt", map_location=torch.device('cuda'))
        model = self.Model(model_dict['config']['model']).to(self.device)

        model.load_state_dict(model_dict['model'])
        model.eval()
        self.model = model

    def init_dataloader(self):
        l2_x_data = sorted(glob.glob(f"{self.config['data_dir']}/SPCAM5/inputs_*"), key=sort_fn)
        l2_y_data = sorted(glob.glob(f"{self.config['data_dir']}/SPCAM5/outputs_*"), key=sort_fn)
        l1_x_data = sorted(glob.glob(f"{self.config['data_dir']}/CAM5/inputs_*"), key=sort_fn)
        l1_y_data = sorted(glob.glob(f"{self.config['data_dir']}/CAM5/outputs_*"), key=sort_fn)

        n = int(365*0.8)
        self.l2_x_train = l2_x_data[:n]
        self.l2_y_train = l2_y_data[:n]
        self.l2_x_valid = l2_x_data[n:365]
        self.l2_y_valid = l2_y_data[n:365]
        self.l2_x_test = l2_x_data[365:]
        self.l2_y_test = l2_y_data[365:]
        self.l1_x_train = l1_x_data[:n]
        self.l1_y_train = l1_y_data[:n]
        self.l1_x_valid = l1_x_data[n:365]
        self.l1_y_valid = l1_y_data[n:365]
        self.l1_x_test = l1_x_data[365:]
        self.l1_y_test = l1_y_data[365:]
        l1_x_scaler_minmax = dill.load(open(f"../../scalers/x_CAM5_minmax_scaler.dill", 'rb'))
        l1_y_scaler_minmax = dill.load(open(f"../../scalers/y_CAM5_minmax_scaler.dill", 'rb'))

        l2_x_scaler_minmax = dill.load(open(f"../../scalers/x_SPCAM5_minmax_scaler.dill", 'rb'))
        l2_y_scaler_minmax = dill.load(open(f"../../scalers/y_SPCAM5_minmax_scaler.dill", 'rb'))

        # Change to first 26 variables
        l2_y_scaler_minmax.min = l2_y_scaler_minmax.min[:26]
        l2_y_scaler_minmax.max = l2_y_scaler_minmax.max[:26]
        l1_y_scaler_minmax.min = l1_y_scaler_minmax.min[:26]
        l1_y_scaler_minmax.max = l1_y_scaler_minmax.max[:26]

        if self.model_type != "multi":
            trainset = l2Dataset(self.l2_x_train, self.l2_y_train, x_scaler=l2_x_scaler_minmax, y_scaler=l2_y_scaler_minmax, variables=26)
            self.trainloader = DataLoader(trainset, batch_size=self.config['batch_size'], shuffle=True, drop_last=False, \
                                            num_workers=4, pin_memory=True)
            validset = l2Dataset(self.l2_x_valid, self.l2_y_valid, x_scaler=l2_x_scaler_minmax, y_scaler=l2_y_scaler_minmax, variables=26)
            self.validloader = DataLoader(validset, batch_size=self.config['batch_size'], shuffle=False, drop_last=False, \
                                            num_workers=4, pin_memory=True)
            testset = l2Dataset(self.l2_x_test, self.l2_y_test, x_scaler=l2_x_scaler_minmax, y_scaler=l2_y_scaler_minmax, variables=26)
            self.testloader = DataLoader(testset, batch_size=self.config['batch_size'], shuffle=False, drop_last=False, \
                                        num_workers=4, pin_memory=True)
        else:
            trainset = MutliDataset(self.l1_x_train, self.l1_y_train, self.l2_x_train, self.l2_y_train,
                                            l1_x_scaler=l1_x_scaler_minmax, l1_y_scaler=l1_y_scaler_minmax,
                                            l2_x_scaler=l2_x_scaler_minmax, l2_y_scaler=l2_y_scaler_minmax, nested=self.config['nested'], variables=[26, 26])
            self.trainloader = DataLoader(trainset, self.config['batch_size'], shuffle=True, drop_last=False, num_workers=0, pin_memory=True)

            validset = MutliDataset(self.l1_x_valid, self.l1_y_valid, self.l2_x_valid, self.l2_y_valid,
                                    l1_x_scaler=l1_x_scaler_minmax, l1_y_scaler=l1_y_scaler_minmax,
                                    l2_x_scaler=l2_x_scaler_minmax, l2_y_scaler=l2_y_scaler_minmax, nested=self.config['nested'], variables=[26, 26])
            self.validloader = DataLoader(validset, self.config['batch_size'], shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
            testset = MutliDataset(self.l1_x_test, self.l1_y_test, self.l2_x_test, self.l2_y_test, l1_x_scaler=l1_x_scaler_minmax, l1_y_scaler=l1_y_scaler_minmax, nested=self.config['nested'], variables=[26, 26])
            self.testloader = DataLoader(testset, self.config['batch_size'], shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

        self.l2_y_scaler_minmax = l2_y_scaler_minmax

    def get_metrics(self, loader):
        self.get_R_stats(loader)
        self.r = self.ssxym / np.sqrt(self.ssxm * self.ssym)
        return self.non_mae, self.nmae, self.r

    def forward_pass(self, data):
        if self.model_type == "multi":
            l1_x, l1_y, l2_x, l2_y = data
            l1_x = l1_x.reshape(-1, 1, l1_x.shape[-1]).to(device)
            l1_y = l1_y.reshape(-1, 1, l1_y.shape[-1]).to(device)
        else:
            l2_x, l2_y = data
            l2_x, l2_y = l2_x.to(device), l2_y.to(device)

        if self.model_type != "attentive":
            l2_x = l2_x.reshape(-1, 1, l2_x.shape[-1]).to(device)
            l2_y = l2_y.reshape(-1, 1, l2_y.shape[-1]).to(device)

        if self.model_type == "forward":
            l2_output_mu = self.model(l2_x)
            l2_truth = l2_y
        elif self.model_type == "multi":
            l1_output_mu, l1_output_cov, l2_output_mu, l2_output_cov, l1_y_truth,\
                l2_truth, l1_z_mu_all, l1_z_cov_all, l1_z_mu_c, l1_z_cov_c, \
                l2_z_mu_all, l2_z_cov_all, l2_z_mu_c, l2_z_cov_c = self.model(l1_x, l1_y, l2_x, l2_y)
        elif self.model_type == "attentive":
            mask, others = split_context_target(l2_x, l2_y, self.config['model']['context_percentage_low'], 
                                                                            self.config['model']['context_percentage_high'])
            context_x, context_y, target_x, l2_truth = l2_x[:, mask], l2_y[:, mask], l2_x[:, others], l2_y[:, others]
            l2_output_mu, l2_output_cov = self.model(context_x, context_y, target_x)
        else:
            l2_output_mu, l2_output_cov, l2_truth, l2_z_mu_all, \
                    l2_z_cov_all, l2_z_mu_c, l2_z_cov_c = self.model(l2_x, l2_y)
        
        non_y_pred = self.l2_y_scaler_minmax.inverse_transform(l2_output_mu.squeeze().cpu().numpy())
        non_y = self.l2_y_scaler_minmax.inverse_transform(l2_truth.squeeze().cpu().numpy())
        return non_y, non_y_pred

    def get_R_stats(self, loader):
        self._get_stats(loader)
        self.ssxm = 0
        self.ssxym = 0
        self.ssym = 0

        with torch.no_grad():
            for i, data in enumerate(tqdm(loader, total=len(loader))):
                if self.model_type == "attentive":
                    n_mb = 9
                    data = [d[:, ::int(n_mb)] for d in data]

                    for b in range(n_mb): # Split into batches due to GPU limit
                        non_y, non_y_pred = self.forward_pass([d[:, b::n_mb] for d in data])

                        self.ssxm += ((non_y - self.y_mean)**2).sum(0)
                        self.ssym += ((non_y_pred - self.y_pred_mean)**2).sum(0)
                        self.ssxym += ((non_y - self.y_mean) * (non_y_pred - self.y_pred_mean)).sum(0)
                else:
                    non_y, non_y_pred = self.forward_pass(data)
                self.ssxm += ((non_y - self.y_mean)**2).sum(0)
                self.ssym += ((non_y_pred - self.y_pred_mean)**2).sum(0)
                self.ssxym += ((non_y - self.y_mean) * (non_y_pred - self.y_pred_mean)).sum(0)
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

        with torch.no_grad():
            for i, data in enumerate(tqdm(loader, total=len(loader))):
                if self.model_type == "attentive":
                    n_mb = 9
                    data = [d[:, ::int(n_mb)] for d in data]
                    for b in range(n_mb): # Split into batches due to GPU limit
                        non_y, non_y_pred = self.forward_pass([d[:, b::n_mb] for d in data])
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
                else:
                    non_y, non_y_pred = self.forward_pass(data)

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

        self.y_mean /= self.n_total
        self.y_pred_mean /= self.n_total
        self.nmae /= self.n_total
        self.non_mae /= self.n_total
        self.nmae = np.abs(np.sqrt(self.non_mae / self.n_total) / np.abs(self.y_mean))


    def get_loss_plot(self):
        
        return self.losses

def plot_with_colorbar(fig, im, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(im, vmin=0, vmax=1, cmap='jet')
    fig.colorbar(im, cax=cax, orientation='vertical')


def plot_variables(x, y, pred=None, loss=None, fidelity="high"):
    col = 2
    if pred is not None:
        col = 3
    fig, axs = plt.subplots(12, col, figsize=(10, 40))
    idx = 0
    n = 32
    for i in range(12):
        if i >= 4:
            n = 1
        if i < 8:
            plot_with_colorbar(fig, x[:, :, idx:idx+n].mean(-1), axs[i, 0])
            axs[i, 0].set_title("X")
        else:
            axs[i, 0].axis('off')
        plot_with_colorbar(fig, y[:, :, idx:idx+n].mean(-1), axs[i, 1])
        axs[i, 1].set_title("Y")
        if pred is not None:
            plot_with_colorbar(fig, pred[:, :, idx:idx+n].mean(-1), axs[i, 2])
            axs[i, 2].set_title("Pred")
            if loss is not None:
                axs[i, 2].set_title(f"Pred, loss: {loss[:, :, idx:idx+n].mean()}")
        idx += n

def plot_one_variable(x, y, y_pred, idx, figsize=(10, 5)):
    fig, axs = plt.subplots(2, 2, figsize=figsize, tight_layout=True)
    axs[0, 0].imshow(y[:, :, idx], vmax=1, vmin=0)
    axs[0, 0].set_title("True 0-1")
    axs[0, 1].imshow(y_pred[:, :, idx], vmax=1, vmin=0)
    axs[0, 1].set_title("Prediction 0-1")

    axs[1, 0].imshow(y[:, :, idx])
    axs[1, 0].set_title("Scaled True")
    axs[1, 1].imshow(y_pred[:, :, idx])
    axs[1, 1].set_title("Scaled Prediction")

    mae_loss = mae_metric(y_pred, y, mean=False).reshape(-1, 136).mean(0)[idx]
    norm_loss = norm_rmse_loss(y_pred, y, metric=True, mean=False).reshape(-1, 136).mean(0)[idx]
    plt.suptitle(f"MAE: {mae_loss:.4f} NRMSE: {norm_loss:.4f}")
    plt.show()

