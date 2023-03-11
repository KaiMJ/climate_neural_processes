import argparse
from ray import tune, air
import time
import random
import dill
import os
import glob
import yaml
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from lib.utils import *
from lib.loss import *
from lib.dataset import *
from lib.model import Model
from scipy.stats import linregress

cwd = os.getcwd()


class Supervisor(tune.Trainable):
    """
        setup and step is for ray tune.
    """

    def setup(self, config=None):
        self.init_config(config)
        self.init_dataloader()
        self.init_model()
        self.init_checkpoint()

    def step(self):
        self.model_step(eval=False)
        with torch.no_grad():
            valid_loss = self.model_step(eval=True)
        return {"loss": valid_loss}

    def init_config(self, ray_config=None):
        """
            ray config is a dict with {"train": {...}, "model": {...}} parameters that override config.yaml
        """
        config = yaml.safe_load(open(f"{cwd}/config.yaml"))

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

        # update with ray config
        if ray_config:
            config.update(ray_config["train"])
            config["model"].update(ray_config["model"])

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

        x_scaler_minmax = dill.load(
            open(f"{cwd}/../../scalers/x_SPCAM5_minmax_scaler.dill", 'rb'))
        y_scaler_minmax = dill.load(
            open(f"{cwd}/../../scalers/y_SPCAM5_minmax_scaler.dill", 'rb'))

         # Change to first 26 variables
        # Follow Azis's process. X -> X/(max(abs(X))
        x_scaler_minmax.min = x_scaler_minmax.min * 0
        x_scaler_minmax.max = np.abs(x_scaler_minmax.max)
        y_scaler_minmax.min = y_scaler_minmax.min[:26] * 0
        y_scaler_minmax.max = np.abs(y_scaler_minmax.max[:26])       # Change to first 26 variables
        self.x_scaler_minmax = x_scaler_minmax
        self.y_scaler_minmax = y_scaler_minmax

        train_dataset = l2Dataset(
            l2_x_train, l2_y_train, x_scaler=x_scaler_minmax, y_scaler=y_scaler_minmax, variables=26)
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config['batch_size'], shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
        val_dataset = l2Dataset(
            l2_x_valid, l2_y_valid, x_scaler=x_scaler_minmax, y_scaler=y_scaler_minmax, variables=26)
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.config['batch_size'], shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    def init_model(self):
        self.model = Model(self.config['model']).to(device)

        self.optim = torch.optim.Adam(self.model.parameters(
        ), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        self.scheduler = StepLR(
            self.optim, step_size=self.config['decay_steps'], gamma=self.config['decay_rate'])
        self.logger.info(
            f"Total trainable parameters {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        self.scaler = torch.cuda.amp.GradScaler()

    def model_step(self, eval):
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
        r2_mean_total = 0

        if not eval:
            self.model.train()
        else:
            self.model.eval()

        for i, (x_, y_) in enumerate(pbar := tqdm(loader, total=len(loader))):
            # mini batch gradients
            n_mb = 4
            mae_mb = 0
            mse_mb = 0
            non_mae_mb = 0
            norm_rmse_mb = 0
            r2_mean_mb = 0

            # Positional embedding
            # x_idxs, y_idxs = np.meshgrid(np.arange(96), np.arange(144))
            # x_idxs = x_idxs / 96 * np.pi
            # y_idxs = y_idxs / 144 * np.pi
            # pos_idxs = np.dstack((np.sin(x_idxs), np.cos(x_idxs), np.sin(y_idxs), np.cos(y_idxs)))
            # pos_idxs = pos_idxs.reshape(96*144, 1, 4)
            # pos_idxs = np.arange(96 * 144).reshape(-1, 1, 1)
            # pos_idxs = torch.from_numpy(pos_idxs).int()
            # pos_idxs = pos_idxs.repeat(24, 1, 1).to(device)

            for bg_idx in range(n_mb):
                x = x_[:, bg_idx::n_mb]
                y = y_[:, bg_idx::n_mb]
                # pos_idxs_mb = pos_idxs[bg_idx::n_mb]

                if eval is False:  # Random dropout during training
                    dropout = self.config['batch_dropout']
                    n = int(x.shape[1] * (1 - dropout))
                    idxs = np.random.permutation(np.arange(x.shape[1]))[:n]
                    x = x[:, idxs, :]
                    y = y[:, idxs, :]
                    # pos_idxs_mb = pos_idxs_mb[idxs]

                x = x.reshape(-1, 1, x.shape[-1]).to(device)
                y = y.reshape(-1, 1, y.shape[-1]).to(device)

                with torch.cuda.amp.autocast():
                    l2_output_mu = self.model(x)

                    mae = mae_loss(l2_output_mu, y)
                    loss = mae

                    if not eval:
                        self.optim.zero_grad()
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        if i == 0:
                            self.scheduler.step()

                mse = mse_loss(l2_output_mu, y, mean=True).detach()
                norm_rmse = norm_rmse_loss(l2_output_mu, y).detach()
                non_y_pred = self.y_scaler_minmax.inverse_transform(
                    l2_output_mu.squeeze().detach().cpu().numpy())
                non_y = self.y_scaler_minmax.inverse_transform(
                    y.squeeze().detach().cpu().numpy())
                non_mae = mae_metric(non_y_pred, non_y)
                r2_mean = 0
                for v in range(non_y.shape[-1]):
                    result = linregress(non_y[:, v], non_y_pred[:, v])
                    r2_mean += result.rvalue ** 2
                r2_mean = r2_mean / (non_y_pred.shape[-1])

                mae_mb += mae
                mse_mb += mse
                non_mae_mb += non_mae
                norm_rmse_mb += norm_rmse
                r2_mean_mb += r2_mean

            mae_mb /= n_mb
            mse_mb /= n_mb
            non_mae_mb /= n_mb
            norm_rmse_mb /= n_mb
            r2_mean_mb /= n_mb

            if not eval:
                writer.add_scalar("mse", mse_mb.item(), self.global_batch_idx)
                writer.add_scalar("mae", mae_mb.item(), self.global_batch_idx)
                writer.add_scalar("norm_rmse", norm_rmse_mb,
                                  self.global_batch_idx)
                writer.add_scalar("non_mae", non_mae_mb.item(),
                                  self.global_batch_idx)
                writer.add_scalar("r2", r2_mean_mb, self.global_batch_idx)
                writer.flush()
                self.global_batch_idx += 1

            pbar.set_description(f"Epoch {self.epoch} {split}")
            pbar.set_postfix_str(
                f"R2: {r2_mean_mb.item():.6f} MAE: {mae_mb.item():.6f} NON-MAE: {non_mae_mb.item():.6f}")

            mse_total += mae_mb
            mae_total += mae.item()
            norm_rmse_total += norm_rmse
            non_mae_total += non_mae.item()
            r2_mean_mb += r2_mean

        mse_total /= (i+1) 
        mae_total /= (i+1) 
        non_mae_total /= (i+1) 
        norm_rmse_total /= (i+1) 
        non_norm_rmse_total /= (i+1) 
        r2_mean_total /= (i+1) 

        if eval:
            writer.add_scalar("mse", mse_total, self.global_batch_idx)
            writer.add_scalar("mae", mae_total, self.global_batch_idx)
            writer.add_scalar("norm_rmse", norm_rmse_total, self.global_batch_idx)
            writer.add_scalar("non_mae", non_mae_total, self.global_batch_idx)
            writer.add_scalar("r2", r2_mean_total, self.global_batch_idx)
            writer.flush()

        end = time.time()
        total_time = end - start

        self.logger.info(
            f"EPOCH: {self.epoch} {split} {total_time:.4f} sec - R2: {r2_mean_total:.6f} NON-MAE: {non_mae_total:.6f}"
            + f" MSE: {mse_total:.6f} MAE: {mae_total:.6f} NRMSE: {norm_rmse:.6f}"
            + f"LR: {self.scheduler.get_last_lr()[0]:6f}")

        return non_mae_total

    def train_model(self):
        try:
            while True:
                self.model_step(eval=False)
                with torch.no_grad():
                    valid_loss = self.model_step(eval=True)

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


def main():
    if TUNE:
        num_samples = 30
        max_iter = 2

        param_space = {
            "train": {
                "lr": tune.qloguniform(1e-6, 1e-3, 1e-6),
                "batch_dropout": tune.uniform(0, 0.5),
                "weight_decay": tune.uniform(0, 0.2),
                # "decay_steps": 10,
                # "decay_rate": 0.9
            },
            "model": {
                "num_heads": tune.choice([4, 8, 16]),
                "attention_layers": tune.choice([4, 8, 12]),
                "n_embd": tune.choice([32, 48, 64, 96, 128]),
                "dropout": tune.uniform(0, 0.4),
            },
        }
        tuner = tune.Tuner(
            tune.with_resources(Supervisor, {"cpu": 1, "gpu": 0.5}),
            tune_config=tune.TuneConfig(
                scheduler=tune.schedulers.ASHAScheduler(
                    mode="min", metric="loss", max_t=max_iter),
                num_samples=num_samples,
            ),
            run_config=air.RunConfig(
                local_dir="./ray_results",
                stop={"training_iteration": max_iter},
                verbose=3,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_at_end=False
                ),
            ),
            param_space=param_space,
        )
        results = tuner.fit()
    else:
        set_seed(seed)
        supervisor = Supervisor()
        supervisor.init_config()
        supervisor.init_dataloader()
        supervisor.init_model()
        supervisor.init_checkpoint()

        supervisor.train_model()


if __name__ == "__main__":
    TUNE = False

    seed = 0
    # parser = argparse.ArgumentParser()
    # parser.add_argument("seed", type=int, help="seed for random number generator")
    # args = parser.parse_args()
    # seed = args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    main()
