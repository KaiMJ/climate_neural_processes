from lib.model import Model
from lib.dataset import *
from lib.loss import *
from ray import tune, air
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
import time
import argparse
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

        n = 365
        split_n = int(n*0.8)
        l2_x_train = l2_x_data[:split_n]
        l2_y_train = l2_y_data[:split_n]
        l2_x_valid = l2_x_data[split_n:n]
        l2_y_valid = l2_y_data[split_n:n]

        x_scaler_minmax = dill.load(
            open(f"{cwd}/../../scalers/x_SPCAM5_minmax_scaler.dill", 'rb'))
        y_scaler_minmax = dill.load(
            open(f"{cwd}/../../scalers/y_SPCAM5_minmax_scaler.dill", 'rb'))

        # Change to first 26 variables
        # Follow Azis's process. X -> X/(max(abs(X))
        x_scaler_minmax.min = x_scaler_minmax.min * 0
        x_scaler_minmax.max = np.abs(x_scaler_minmax.max)
        y_scaler_minmax.min = y_scaler_minmax.min[:26] * 0
        y_scaler_minmax.max = np.abs(y_scaler_minmax.max[:26])
        self.x_scaler_minmax = x_scaler_minmax
        self.y_scaler_minmax = y_scaler_minmax

        train_dataset = l2Dataset(
            l2_x_train, l2_y_train, x_scaler=x_scaler_minmax, y_scaler=y_scaler_minmax, variables=26)
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config['batch_size'], shuffle=True, drop_last=False, num_workers=2, pin_memory=True)
        val_dataset = l2Dataset(
            l2_x_valid, l2_y_valid, x_scaler=x_scaler_minmax, y_scaler=y_scaler_minmax, variables=26)
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.config['batch_size'], shuffle=False, drop_last=False, num_workers=2, pin_memory=True)

    def init_model(self):
        self.model = Model(self.config['model']).to(device)

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=self.config['lr'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optim, gamma=self.config['decay_rate'])
        self.logger.info(
            f"Total trainable parameters {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} Device: {device}")
        self.scaler = torch.cuda.amp.GradScaler()
        self.loss_fn = NegRLoss()

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

        nll_total = 0
        mse_total = 0
        mae_total = 0
        non_mae_total = 0
        norm_rmse_total = 0
        non_norm_rmse_total = 0
        r2_total = 0

        if not eval:
            self.model.train()
        else:
            self.model.eval()

        mb = 48
        for i, (x_, y_) in enumerate(pbar := tqdm(loader, total=len(loader))):
            nll_mb_total = 0
            mse_mb_total = 0
            mae_mb_total = 0
            non_mae_mb_total = 0
            norm_rmse_mb_total = 0
            r2_mb_total = 0
            self.optim.zero_grad()

            for b in range(mb):
                x = x_[:, b::mb]
                y = y_[:, b::mb]

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
                l2_truth = y_target

                with torch.cuda.amp.autocast():
                    l2_output_mu, l2_output_cov, l2_z_mu_c, l2_z_cov_c, l2_z_mu_all, l2_z_cov_all = self.model(
                        x_context, y_context, x_target, y_target)

                    if torch.any(torch.isnan(l2_output_mu)):
                        self.logger.info("Prediction returned NAN...")
                        continue

                    nll = nll_loss(l2_output_mu, l2_output_cov, l2_truth)
                    mse = mse_loss(l2_output_mu, l2_truth).detach()
                    kld = kl_div(l2_z_mu_all, l2_z_cov_all,
                                 l2_z_mu_c, l2_z_cov_c)
                    r_score = self.loss_fn(
                        l2_output_mu[0], y_target[0]).detach()

                    loss = nll + mse + kld

                    try:
                        self.scaler.scale(loss).backward()
                    except Exception as e:
                        self.logger.info(e)
                        continue
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                    if not eval:
                        if i == 0:
                            self.scheduler.step()
                        if (b == mb-1):
                            self.scaler.step(self.optim)
                        self.scaler.update()

                    mae = mae_loss(l2_output_mu, l2_truth).detach()
                    norm_rmse = norm_rmse_loss(l2_output_mu, l2_truth).detach()
                    non_y_pred = self.y_scaler_minmax.inverse_transform(
                        l2_output_mu.squeeze().detach().cpu().numpy())
                    non_y = self.y_scaler_minmax.inverse_transform(
                        l2_truth.squeeze().detach().cpu().numpy())
                    non_mae = mae_metric(non_y_pred, non_y)

                    r_score[r_score > 1.0] = 1.0
                    r_score[r_score < -1.0] = -1.0
                    r2 = (r_score ** 2).mean()

                    pbar.set_description(f"Epoch {self.epoch} {split}")
                    pbar.set_postfix_str(
                        f"R2: {r2.item():.6f} MAE: {mae.item():.6f} NON-MAE: {non_mae.item():.6f}")

                    nll_mb_total += nll.detach()
                    mse_mb_total += mse.detach()
                    mae_mb_total += mae
                    non_mae_mb_total += non_mae
                    norm_rmse_mb_total += norm_rmse
                    r2_mb_total += r2.item()

            nll_mb_total /= mb
            mse_mb_total /= mb
            mae_mb_total /= mb
            non_mae_mb_total /= mb
            norm_rmse_mb_total /= mb
            r2_mb_total /= mb
            if not eval:
                writer.add_scalar("nll", nll_mb_total.item(),
                                    self.global_batch_idx)
                writer.add_scalar("mse", mse_mb_total.item(),
                                    self.global_batch_idx)
                writer.add_scalar("mae", mae_mb_total.item(),
                                    self.global_batch_idx)
                writer.add_scalar("norm_rmse", norm_rmse_mb_total,
                                    self.global_batch_idx)
                writer.add_scalar("non_mae", non_mae_mb_total.item(),
                                    self.global_batch_idx)
                writer.add_scalar("r2", r2_mb_total, self.global_batch_idx)
                writer.flush()
                self.global_batch_idx += 1

            nll_total += nll_mb_total
            mse_total += mse_mb_total
            mae_total += mae_mb_total
            non_mae_total += non_mae_mb_total
            norm_rmse_total += norm_rmse_mb_total
            r2_total += r2_mb_total

        nll_total /= i+1
        mse_total /= i+1
        mae_total /= i+1
        non_mae_total /= i+1
        norm_rmse_total /= i+1
        non_norm_rmse_total /= i+1
        r2_total /= i+1

        if eval:
            writer.add_scalar("nll", nll_total, self.global_batch_idx)
            writer.add_scalar("mse", mse_total, self.global_batch_idx)
            writer.add_scalar("mae", mae_total, self.global_batch_idx)
            writer.add_scalar("norm_rmse", norm_rmse_total,
                              self.global_batch_idx)
            writer.add_scalar("non_mae", non_mae_total, self.global_batch_idx)
            writer.add_scalar("r2", r2_total, self.global_batch_idx)
            writer.flush()

        end = time.time()
        total_time = end - start

        self.logger.info(
            f"EPOCH: {self.epoch} {split} {total_time:.4f} sec - R2: {r2_total:.6f} NON-MAE: {non_mae_total:.6f}"
            + f" MSE: {mse_total:.6f} MAE: {mae_total:.6f} NRMSE: {norm_rmse_total:.6f}"
            + f" LR: {self.scheduler.get_last_lr()[0]:6f}")

        return r2_total

    def train_model(self):
        try:
            while True:
                self.model_step(eval=False)

                with torch.no_grad():
                    valid_loss = self.model_step(eval=True)

                self.config['epoch'] = self.epoch
                self.config['global_batch_idx'] = self.global_batch_idx

                save_best = False
                if valid_loss > self.best_loss:
                    self.best_loss = valid_loss
                    self.config['best_loss'] = self.best_loss
                    save_best = True
                    self.logger.info(
                        f"Best validation Loss: {self.best_loss:.6f}")
                    self.config['patience'] = 0
                elif self.config['patience'] != -1:  # if patience == -1, run forever
                    self.config['patience'] = self.config['patience'] + 1
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

                if self.config['patience'] == self.config['max_patience']:
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
                "hidden_layers": tune.choice([3, 5, 7]),
                "z_hidden_layers": tune.choice([3, 5, 7]),
                "z_hidden_dim": tune.choice([32, 64, 96]),
                "z_dim": tune.choice([64, 96, 128, 160]),
                "hidden_dim": tune.choice([32, 64, 96])
            },
        }
        tuner = tune.Tuner(
            tune.with_resources(Supervisor, {"cpu": 1, "gpu": 1}),
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

    device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')

    # CUDA_VISIBLE_DEVICES=1,2,3,5 python train.py

    with SeedContext(seed):
        main()
