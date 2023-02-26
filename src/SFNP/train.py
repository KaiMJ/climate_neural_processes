import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.utils.tensorboard import SummaryWriter
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import yaml, glob, os, dill, random, time, torch, argparse
import os, sys
sys.path.append('..')
sys.path.append('.')

from tqdm import tqdm
from lib.utils import *
from lib.loss import *
from lib.dataset import *
from model import Model

scaler = torch.cuda.amp.GradScaler()

# parser = argparse.ArgumentParser()
# parser.add_argument("seed", type=int, help="seed for random number generator")
# args = parser.parse_args()
# seed = args.seed

seed = 0

device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')

def step(model, optim, scheduler, loader, writer, logger, eval=False):
    start = time.time()
    global global_batch_idx
    split = "Train" if not eval else "Valid"

    mse_total = 0
    mae_total = 0
    non_mae_total = 0
    norm_rmse_total = 0
    non_norm_rmse_total = 0
    if not eval:
        model.train()
    else:
        model.eval()

    for i, (x, y) in enumerate(pbar:= tqdm(loader, total=len(loader))):
        
        if eval is False: # Random dropout during trainig
            dropout = config['batch_dropout']
            n = int(x.shape[1] * (1 - dropout))
            idxs = np.random.permutation(np.arange(x.shape[1]))[:n]
            x = x[:, idxs, :]
            y = y[:, idxs, :]
        
        x = x.reshape(-1, 1, x.shape[-1]).to(device)
        y = y.reshape(-1, 1, y.shape[-1]).to(device)

        with torch.cuda.amp.autocast():
            l2_output_mu, l2_output_cov, l2_truth, l2_z_mu_all, l2_z_cov_all, l2_z_mu_c, l2_z_cov_c = model(x, y)

            nll = nll_loss(l2_output_mu, l2_output_cov, l2_truth)
            mae = mae_loss(l2_output_mu, l2_truth)
            kld = kld_gaussian_loss(l2_z_mu_all, l2_z_cov_all, l2_z_mu_c, l2_z_cov_c)

            loss = nll + mae + kld

        if not eval:
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            if i == 0:
                scheduler.step()

        mse = mse_loss(l2_output_mu, l2_truth, mean=True).detach()
        norm_rmse = norm_rmse_loss(l2_output_mu, l2_truth).detach()
        non_y_pred = y_scaler_minmax.inverse_transform(l2_output_mu.unsqueeze(1).detach().cpu().numpy())
        non_y = y_scaler_minmax.inverse_transform(l2_truth.unsqueeze(1).detach().cpu().numpy())
        non_mae = mae_metric(non_y_pred, non_y)

        mse_total += mse.item()
        mae_total += mae.item()
        norm_rmse_total += norm_rmse
        non_mae_total += non_mae.item()

        if eval is not False:
            writer.add_scalar("mse", mse.item(), global_batch_idx)
            writer.add_scalar("mae", mae.item(), global_batch_idx)
            writer.add_scalar("norm_rmse", norm_rmse, global_batch_idx)
            writer.add_scalar("non_mae", non_mae.item(), global_batch_idx)
            writer.flush()

        pbar.set_description(f"Epoch {epoch} {split}")
        pbar.set_postfix_str(f"MSE: {mse.item():.6f} MAE: {mae.item():.6f} NRMSE: {norm_rmse:.6f}")
        if not eval:
            global_batch_idx += 1

    mse_total /= i+1
    mae_total /= i+1
    non_mae_total /= i+1
    norm_rmse_total /= i+1
    non_norm_rmse_total /= i+1

    end = time.time()
    total_time = end - start

    logger.info(f"EPOCH: {epoch} {split} {total_time:.4f} sec - NON-MAE: {non_mae_total:.6f} MSE: {mse_total:.6f} MAE: {mae_total:.6f} NRMSE: {norm_rmse:.6f}")
                    # LR: {scheduler.get_last_lr()[0]:.6f}")

    return non_mae_total

if __name__ == "__main__":
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    config = yaml.safe_load(open("config.yaml"))

    # Load config file from checkpoint, Don't change saved_config.yaml
    if config['load_checkpoint']:
        name = config['load_checkpoint']
        config_path = os.path.join(config['log_dir'], name, "saved_config.yaml")
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

    train_writer = SummaryWriter(config['writer_dir'] + "/train")
    valid_writer = SummaryWriter(config['writer_dir'] + "/valid")
    batch_size = config['batch_size']

    # Train and validation data split
    # 04/01/2023 -- 01/17/2004 | 01/18/2004 - 3/31/2004
    l2_x_data = sorted(glob.glob(f"{config['data_dir']}/inputs_*"), key=sort_fn)
    l2_y_data = sorted(glob.glob(f"{config['data_dir']}/outputs_*"), key=sort_fn)

    split_n = int(365*0.8)
    l2_x_train = l2_x_data[:split_n]
    l2_y_train = l2_y_data[:split_n]
    l2_x_valid = l2_x_data[split_n:]
    l2_y_valid = l2_y_data[split_n:]
    
    x_scaler_minmax = dill.load(open(f"../../scalers/x_SPCAM5_minmax_scaler.dill", 'rb'))
    y_scaler_minmax = dill.load(open(f"../../scalers/y_SPCAM5_minmax_scaler.dill", 'rb'))

    # Change to first 26 variables
    y_scaler_minmax.min = y_scaler_minmax.min[:26]
    y_scaler_minmax.max = y_scaler_minmax.max[:26]

    train_dataset = l2Dataset(l2_x_train, l2_y_train, x_scaler=x_scaler_minmax, y_scaler=y_scaler_minmax, variables=26)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    val_dataset = l2Dataset(l2_x_valid, l2_y_valid, x_scaler=x_scaler_minmax, y_scaler=y_scaler_minmax, variables=26)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    model = Model(config['model']).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = StepLR(optim, step_size=config['decay_steps'], gamma=config['decay_rate'])

    ray_dir = os.path.join(config["log_dir"], "ray")
    ray_config = {
        "lr": tune.loguniform(1e-6, 1e-1)
    }
    ray_scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=10,
            grace_period=3,
            reduction_factor=2)
    ray_reporter = CLIReporter(metric_columns=["loss", "training_iteration"])
    result = tune.run(
        


    logger.info(f"Total trainable parameters {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    epoch = 0
    global_batch_idx = config['global_batch_idx']

    if config['load_checkpoint']:
        save_dict = torch.load(config['checkpoint_path'])
        epoch = save_dict['epoch'] + 1
        model.load_state_dict(save_dict['model'])
        optim.load_state_dict(save_dict['optim'])
        scheduler.load_state_dict(save_dict['scheduler'])
    
    best_loss = config['best_loss']

    try:
        while True:
            step(model, optim, scheduler, train_loader, train_writer, logger, eval=False)
            with torch.no_grad():
                valid_loss = step(model, optim, scheduler, val_loader, valid_writer, logger, eval=True)
        
            config['epoch'] = epoch
            config['global_batch_idx'] = global_batch_idx          

            save_best = False
            if valid_loss < best_loss:
                best_loss = valid_loss
                config['best_loss'] = best_loss
                save_best = True
                logger.info(f"Best validation Loss: {best_loss:.6f}")
            elif config['patience'] != -1: # if patience == -1, run forever
                config['patience'] = config['patience'] - 1
                logger.info(f"Patience: {config['patience']}")

            save_dict = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optim': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': config
            }

            yaml.dump(config, open(config['save_config'], 'w'), default_flow_style=False)
            
            if save_best:
                torch.save(save_dict, config['best_path'])
            torch.save(save_dict, config['checkpoint_path'])

            if config['patience'] <= 0:
                logger.info("Early stopping")
                break
            epoch += 1

    except KeyboardInterrupt as e:
        logger.info("Interrupted")
