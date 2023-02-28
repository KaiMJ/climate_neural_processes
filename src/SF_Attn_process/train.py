import sys
sys.path.append('..')
sys.path.append('.')
from model import Model
from lib.dataset import *
from lib.loss import *
from lib.utils import *
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import yaml
import glob
import os
import dill
import random
import time
import argparse


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SeedContext:
    def __init__(self, seed):
        self.seed = seed
        self.state = np.random.get_state()

    def __enter__(self):
        set_seed(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        np.random.set_state(self.state)


def split_context_target(x, y, context_percentage_low, context_percentage_high):
    """Helper function to split randomly into context and target"""
    context_percentage = np.random.uniform(
        context_percentage_low, context_percentage_high)
    n_context = int(x.shape[1]*context_percentage)
    ind = np.arange(x.shape[1])
    mask = np.random.choice(ind, size=n_context, replace=False)
    others = np.delete(ind, mask)

    return mask, others

# TODO: look at different KLD implementations


def kl_div(prior_mu, prior_var, posterior_mu, posterior_var):
    kl_div = (torch.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / \
        torch.exp(prior_var) - 1. + (prior_var - posterior_var)
    kl_div = 0.5 * kl_div.sum()
    return kl_div


class Supervisor():

    def __init__(self, tune=False):
        self.init_config()
        self.init_dataloader()
        if tune:
            self.tune_best_loss = 999999
        else:
            self.init_model()
        self.init_checkpoint()

    def init_config(self):
        config = yaml.safe_load(open("config.yaml"))

        # Load config file from checkpoint, Don't change saved_config.yaml
        if config['load_checkpoint']:
            name = config['load_checkpoint']
            config_path = os.path.join(
                config['log_dir'], name, "saved_config.yaml")
            config = yaml.safe_load(open(config_path))
            config['load_checkpoint'] = True
        else:
            name = f"{config['title']}_seed{seed}_lr{config['lr']}_bs_{config['batch_size']}_{time.ctime()}"

        # Get Logger
        log_dir = os.path.join(config['log_dir'], name)
        logger = get_logger(log_dir=log_dir, name=config['title'])

        if config['load_checkpoint']:
            logger.info("Resuming from checkpoint...")
        else:
            config['checkpoint_path'] = log_dir + "/checkpoint.pt"
            config['best_path'] = log_dir + "/best.pt"
            config['save_config'] = log_dir + "/saved_config.yaml"
            config['writer_dir'] = log_dir + "/runs"
        self.train_writer = SummaryWriter(config['writer_dir'] + "/train")
        self.valid_writer = SummaryWriter(config['writer_dir'] + "/valid")
        self.config = config
        self.logger = logger

    def init_checkpoint(self):
        self.epoch = 0
        self.global_batch_idx = self.config['global_batch_idx']

        if self.config['load_checkpoint']:
            save_dict = torch.load(self.config['checkpoint_path'])
            self.epoch = save_dict['epoch'] + 1
            self.model.load_state_dict(save_dict['model'])
            self.optim.load_state_dict(save_dict['optim'])
            self.scheduler.load_state_dict(save_dict['scheduler'])

        self.best_loss = self.config['best_loss']

    def init_dataloader(self):
        # Train and validation data split
        # 04/01/2023 -- 01/17/2004 | 01/18/2004 - 3/31/2004
        l2_x_data = sorted(
            glob.glob(f"{self.config['data_dir']}/SPCAM5/inputs_*"), key=sort_fn)
        l2_y_data = sorted(
            glob.glob(f"{self.config['data_dir']}/SPCAM5/outputs_*"), key=sort_fn)

        split_n = int(365*0.8)
        l2_x_train = l2_x_data[:split_n]
        l2_y_train = l2_y_data[:split_n]
        l2_x_valid = l2_x_data[split_n:365]
        l2_y_valid = l2_y_data[split_n:365]

        self.x_scaler_minmax = dill.load(
            open(f"../../scalers/x_SPCAM5_minmax_scaler.dill", 'rb'))
        y_scaler_minmax = dill.load(
            open(f"../../scalers/y_SPCAM5_minmax_scaler.dill", 'rb'))

        # Change to first 26 variables
        y_scaler_minmax.min = y_scaler_minmax.min[:26]
        y_scaler_minmax.max = y_scaler_minmax.max[:26]

        train_dataset = l2Dataset(
            l2_x_train, l2_y_train, x_scaler=self.x_scaler_minmax, y_scaler=y_scaler_minmax, variables=26)
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config['batch_size'], shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
        val_dataset = l2Dataset(
            l2_x_valid, l2_y_valid, x_scaler=self.x_scaler_minmax, y_scaler=y_scaler_minmax, variables=26)
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.config['batch_size'], shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
        self.y_scaler_minmax = y_scaler_minmax

    def init_model(self):
        self.model = Model(self.config['model']).to(device)

        self.optim = torch.optim.Adam(self.model.parameters(
        ), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        self.scheduler = StepLR(
            self.optim, step_size=self.config['decay_steps'], gamma=self.config['decay_rate'])
        self.logger.info(
            f"Total trainable parameters {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        self.scaler = torch.cuda.amp.GradScaler()

    def step(self, eval):
        start = time.time()
        if not eval:
            split = "Train"
            loader = self.train_loader
            writer = self.train_writer
        else:
            split = "Valid"
            loader = self.val_loader
            writer = self.valid_writer

        mse_total = 0
        mae_total = 0
        non_mae_total = 0
        norm_rmse_total = 0
        non_norm_rmse_total = 0

        if not eval:
            self.model.train()
        else:
            self.model.eval()

        for i, (x_, y_) in enumerate(pbar := tqdm(loader, total=len(loader))):
            n_mb = 9
            # For Tuning
            rand_idxs = np.random.permutation(
                np.arange(x_.shape[1] / 144 * n_mb))
            x_ = x_[:, rand_idxs]
            y_ = y_[:, rand_idxs]

            # mini batch gradients
            mae_mb = 0
            mse_mb = 0
            non_mae_mb = 0
            norm_rmse_mb = 0

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
                    x, y, self.config['model']['context_percentage_low'], self.config['model']['context_percentage_high'])
                x_context = x[:, context_idxs, :].to(device)
                y_context = y[:, context_idxs].to(device)
                x_target = x[:, target_idxs, :].to(device)
                y_target = y[:, target_idxs].to(device)
                l2_truth = y_target

                with torch.cuda.amp.autocast():
                    l2_output_mu, l2_output_cov, l2_z_mu_c, l2_z_cov_c, l2_z_mu_all, l2_z_cov_all = self.model(
                        x_context, y_context, x_target, y_target)

                    if torch.any(torch.isnan(l2_output_mu)):
                        self.logger.info(
                            "Prediction returned NAN. Learning rate is too high...")
                        continue

                    nll = nll_loss(l2_output_mu, l2_output_cov, l2_truth)
                    mae = mae_loss(l2_output_mu, l2_truth)
                    kld = kl_div(l2_z_mu_all, l2_z_cov_all,
                                 l2_z_mu_c, l2_z_cov_c)
                    loss = nll + mae + kld

                    if not eval:
                        self.optim.zero_grad()
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        if i == 0:
                            self.scheduler.step()

                mse = mse_loss(l2_output_mu, l2_truth, mean=True).detach()
                norm_rmse = norm_rmse_loss(l2_output_mu, l2_truth).detach()
                non_y_pred = self.y_scaler_minmax.inverse_transform(
                    l2_output_mu.unsqueeze(1).detach().cpu().numpy())
                non_y = self.y_scaler_minmax.inverse_transform(
                    l2_truth.unsqueeze(1).detach().cpu().numpy())
                non_mae = mae_metric(non_y_pred, non_y)

                mae_mb += mae
                mse_mb += mse
                non_mae_mb += non_mae
                norm_rmse_mb += norm_rmse

            mae_mb /= n_mb
            mse_mb /= n_mb
            non_mae_mb /= n_mb
            norm_rmse_mb /= n_mb

            if eval is not False:
                writer.add_scalar("mse", mse_mb.item(), self.global_batch_idx)
                writer.add_scalar("mae", mae_mb.item(), self.global_batch_idx)
                writer.add_scalar("non_mae", non_mae_mb.item(),
                                  self.global_batch_idx)
                writer.add_scalar("norm_rmse", norm_rmse_mb,
                                  self.global_batch_idx)
                writer.flush()

            pbar.set_description(f"Epoch {self.epoch} {split}")
            pbar.set_postfix_str(
                f"MSE: {mse_mb.item():.6f} MAE: {mae_mb.item():.6f} NON-MAE: {non_mae_mb.item():.6f}")
            if not eval:
                self.global_batch_idx += 1

            mae_total += mae_mb.item()
            mse_total += mse_mb.item()
            norm_rmse_total += norm_rmse_mb
            non_mae_total += non_mae_mb.item()

        mse_total /= (i+1)
        mae_total /= (i+1)
        non_mae_total /= (i+1)
        norm_rmse_total /= (i+1)
        non_norm_rmse_total /= (i+1)

        end = time.time()
        total_time = end - start

        self.logger.info(
            f"EPOCH: {self.epoch} {split} {total_time:.4f} sec - NON-MAE: {non_mae_total:.6f}"
            + f" MSE: {mse_total:.6f} MAE: {mae_total:.6f} NRMSE: {norm_rmse:.6f}"
            + f"LR: {self.scheduler.get_last_lr()[0]:6f}")

        return non_mae_total

    def train(self):
        try:
            while True:
                self.step(eval=False)
                with torch.no_grad():
                    valid_loss = self.step(eval=True)

                self.config['epoch'] = self.epoch
                self.config['global_batch_idx'] = self.global_batch_idx

                save_best = False
                if valid_loss < self.best_loss:
                    self.best_loss = valid_loss
                    self.config['best_loss'] = self.best_loss
                    save_best = True
                    self.logger.info(
                        f"Best validation Loss: {self.best_loss:.6f}")
                elif self.config['patience'] != -1:  # if patience == -1, run forever
                    self.config['patience'] = self.config['patience'] - 1
                    self.logger.info(f"Patience: {self.config['patience']}")

                save_dict = {
                    'epoch': self.epoch,
                    'model': self.model.state_dict(),
                    'optim': self.optim.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'config': self.config
                }

                yaml.dump(self.config, open(
                    self.config['save_config'], 'w'), default_flow_style=False)

                if save_best:
                    torch.save(save_dict, self.config['best_path'])
                torch.save(save_dict, self.config['checkpoint_path'])

                if self.config['patience'] <= 0:
                    self.logger.info("Early stopping")
                    break
                self.epoch += 1

        except KeyboardInterrupt as e:
            self.logger.info("Interrupted")

    def hyper_tune(self, max_epochs, hyper_params, hyper_model_params):
        self.config.update(hyper_params)
        self.config["model"].update(hyper_model_params)
        with SeedContext(seed):
            self.init_model()
            best_loss = 100000
            self.logger.info("Tuning: " + str(hyper_params) +
                             " " + str(hyper_model_params))

            self.epoch = 0
            for e in range(max_epochs):
                self.step(eval=False)
                with torch.no_grad():
                    valid_loss = self.step(eval=True)
                if valid_loss < best_loss:
                    best_loss = valid_loss
                self.epoch += 1
            self.logger.info("Tuning loss: " + str(best_loss))

            if best_loss < self.tune_best_loss:
                self.tune_best_loss = best_loss
                self.best_results = {
                    "hyper_params": hyper_params,
                    "hyper_model_params": hyper_model_params,
                    "loss": best_loss
                }


if __name__ == "__main__":
    tune = False

    seed = 0
    # parser = argparse.ArgumentParser()
    # parser.add_argument("seed", type=int, help="seed for random number generator")
    # args = parser.parse_args()
    # seed = args.seed

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    with SeedContext(seed):
        supervisor = Supervisor(tune)

    def qloguniform(low, high, base, q):
        return np.power(base, np.random.uniform(low, high)) // q * q

    # Hyper parameter tuning
    if tune:
        num_samples = 30
        max_epochs = 2

        try:
            for i in range(num_samples):
                with SeedContext(np.random.randint(1000)):
                    hyper_config = {
                        "weight_decay": np.random.uniform(0, 0.1),
                        "lr": qloguniform(-6, -3, 7, 1e-6),
                    }
                    hyper_model_config = {
                        "num_heads": int(np.random.choice([4, 8, 16])),
                        "attention_layers": np.random.choice([4, 8, 12]),
                        "n_embd": int(np.random.choice([32, 48, 64, 96, 128])),
                        "hidden_dim": np.random.choice([32, 64, 96, 128, 160]),
                        "dropout": np.random.uniform(0, 0.4),
                    }
                supervisor.hyper_tune(
                    max_epochs, hyper_config, hyper_model_config)
        except Exception as e:
            print(e)
        finally:
            supervisor.logger.info(
                "Best tuning results: " + str(supervisor.best_results))

    else:
        supervisor.train()
