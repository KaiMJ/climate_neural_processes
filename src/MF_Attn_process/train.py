import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.utils.tensorboard import SummaryWriter

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
from model import MultiLatentModel

scaler = torch.cuda.amp.GradScaler()

# parser = argparse.ArgumentParser()
# parser.add_argument("seed", type=int, help="seed for random number generator")
# args = parser.parse_args()
# seed = args.seed
seed = 0


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def split_context_target(x, y, context_percentage_low, context_percentage_high):
    """Helper function to split randomly into context and target"""
    context_percentage = np.random.uniform(context_percentage_low,context_percentage_high)
    n_context = int(x.shape[1]*context_percentage)
    ind = np.arange(x.shape[1])
    mask = np.random.choice(ind, size=n_context, replace=False)
    others = np.delete(ind,mask)

    return mask, others

# TODO: look at different KLD implementations
def kl_div(prior_mu, prior_var, posterior_mu, posterior_var):
    kl_div = (torch.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / torch.exp(prior_var) - 1. + (prior_var - posterior_var)
    kl_div = 0.5 * kl_div.sum()
    return kl_div

def step(model, optim, scheduler, loader, writer, logger, eval=False):
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

    for i, (l1_x, l1_y, l2_x, l2_y) in enumerate(pbar:= tqdm(loader, total=len(loader))):
        
        l1_x = l1_x[:, ::2].to(device)
        l1_y = l1_y[:, ::2].to(device)
        l2_x = l2_x[:, ::2].to(device)
        l2_y = l2_y[:, ::2].to(device)  
    
        batch_loss = 0
        batch_nll = 0
        batch_mae = 0
        batch_mse = 0
        batch_norm_rmse = 0

        n = 6
        with torch.cuda.amp.autocast():
            context_idxs, target_idxs = split_context_target(l1_x, l1_y, config['model']['context_percentage_low'], \
                                                                        config['model']['context_percentage_high'])
            l1_x_context = l1_x[:, context_idxs, :]
            l1_y_context = l1_y[:, context_idxs]
            l1_x_target = l1_x[:, target_idxs, :]
            l2_x_context = l2_x[:, context_idxs, :]
            l2_y_context = l2_y[:, context_idxs]
            l2_x_target = l2_x[:, target_idxs, :]

            l2_output_mu, l2_output_cov, l2_z_mu_c, l2_z_cov_c, l2_z_mu_all, l2_z_cov_all = \
                model(l1_x_context, l1_y_context, l1_x_target, l2_x_context, l2_y_context, l2_x_target, l1_x, l1_y, l2_x, l2_y)

            # TODO: FIX both levels
            nll = nll_loss(l2_output_mu, l2_output_cov, y, mean=True)
            
            mae = mae_loss(l2_output_mu, y)
            kld = kl_div(l2_z_mu_all, l2_z_cov_all, l2_z_mu_c, l2_z_cov_c)
            loss = nll + mae + kld

        if not eval:
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            if i == 0:
                scheduler.step()

        mse = mse_loss(l2_output_mu, y, mean=True).detach()
        non_y_pred = l2_y_scaler_minmax.inverse_transform(l2_output_mu.unsqueeze(1).detach().cpu().numpy())
        non_y = l2_y_scaler_minmax.inverse_transform(y.unsqueeze(1).detach().cpu().numpy())
        non_mae = mae_metric(non_y_pred, non_y)
        norm_rmse = norm_rmse_loss(l2_output_mu, y).detach()
        non_norm_rmse = norm_rmse_loss(non_y_pred, non_y, metric=True).mean()

        mse_total += mse.item()
        mae_total += mae.item()
        non_mae_total += non_mae.item()
        norm_rmse_total += norm_rmse
        non_norm_rmse_total += non_norm_rmse

        writer.add_scalar("mse", mse.item(), global_batch_idx)
        writer.add_scalar("mae", mae.item(), global_batch_idx)
        writer.add_scalar("nll", nll.item(), global_batch_idx)
        writer.add_scalar("non_mae", non_mae.item(), global_batch_idx)
        writer.add_scalar("norm_rmse", norm_rmse, global_batch_idx)
        writer.add_scalar("non_norm_rmse", non_norm_rmse, global_batch_idx)
        writer.close()

        pbar.set_description(f"Epoch {epoch} {split}")
        pbar.set_postfix_str(f"MSE: {mse.item():.6f} NON-MAE: {non_mae.item():.6f} NON-RMSE: {non_norm_rmse:.0f}")
        if not eval:
            global_batch_idx += 1

    mse_total /= i+1
    mae_total /= i+1
    non_mae_total /= i+1
    norm_rmse_total /= i+1
    non_norm_rmse_total /= i+1

    logger.info(f"EPOCH: {epoch} {split} - MSE: {mse_total:.6f} MAE: {mae_total:.6f} NON-MAE: {non_mae_total:.6f} " + \
            f"NRMSE: {norm_rmse_total:.6f} NON-NRMSE: {non_norm_rmse_total:.6f} LR: {scheduler.get_last_lr()[0]:.6f}")

    return non_norm_rmse_total.item()

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

    train_idx_start = 1872
    train_idx_end = 10080
    valid_idx_end = 12240

    # 4/6/2023 -- 3/31/2004 / 6/30/2004 - 9/22/2004
    l1_x_data = sorted(glob.glob(f"{config['data_dir']}/batch_low_data/inputs_*"), key=sort_batch_fn)
    l1_y_data = sorted(glob.glob(f"{config['data_dir']}/batch_low_data/outputs_*"), key=sort_batch_fn)
    l2_x_data = sorted(glob.glob(f"{config['data_dir']}/batch_high_data/inputs_*"), key=sort_batch_fn)
    l2_y_data = sorted(glob.glob(f"{config['data_dir']}/batch_high_data/outputs_*"), key=sort_batch_fn)

    l1_x_train = l1_x_data[train_idx_start:train_idx_end]
    l1_y_train = l1_y_data[train_idx_start:train_idx_end]
    l2_x_train = l2_x_data[train_idx_start:train_idx_end]
    l2_y_train = l2_y_data[train_idx_start:train_idx_end]

    l1_x_val = l1_x_data[train_idx_end:valid_idx_end]
    l1_y_val = l1_y_data[train_idx_end:valid_idx_end]
    l2_x_val = l2_x_data[train_idx_end:valid_idx_end]
    l2_y_val = l2_y_data[train_idx_end:valid_idx_end]

    l1_x_scaler_minmax = dill.load(open(f"{config['data_dir']}/scaler/low_x_minmax_scaler.pkl", 'rb'))
    l1_y_scaler_minmax = dill.load(open(f"{config['data_dir']}/scaler/low_y_minmax_scaler.pkl", 'rb'))
    l2_x_scaler_minmax = dill.load(open(f"{config['data_dir']}/scaler/high_x_minmax_scaler.pkl", 'rb'))
    l2_y_scaler_minmax = dill.load(open(f"{config['data_dir']}/scaler/high_y_minmax_scaler.pkl", 'rb'))

    l1_y_scaler_minmax.min = l1_y_scaler_minmax.min[:26]
    l1_y_scaler_minmax.max = l1_y_scaler_minmax.max[:26]
    l2_y_scaler_minmax.min = l2_y_scaler_minmax.min[:32]
    l2_y_scaler_minmax.max = l2_y_scaler_minmax.max[:32]

    train_dataset = MutliDataset(l1_x_train, l1_y_train, l2_x_train, l2_y_train, \
                                l1_x_scaler_minmax, l1_y_scaler_minmax, l2_x_scaler_minmax, l2_y_scaler_minmax, \
                                nested=True, variables=[26, 32])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    val_dataset = MutliDataset(l1_x_val, l1_y_val, l2_x_val, l2_y_val, \
                                l1_x_scaler_minmax, l1_y_scaler_minmax, l2_x_scaler_minmax, l2_y_scaler_minmax, \
                                nested=True, variables=[26, 32])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiLatentModel(config['model']).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    # optim = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = StepLR(optim, step_size=config['decay_steps'], gamma=config['decay_rate'])
    # scheduler = CyclicLR(optim, config['base_lr'], config['max_lr'], step_size_up=2000, step_size_down=None, mode='exp_range', \
                        # gamma=config['gamma'], scale_fn=None, scale_mode='iterations', cycle_momentum=False, base_momentum=0.8, \
                        # max_momentum=0.9, last_epoch=- 1, verbose=False)

    logger.info(f"Total trainable parameters {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    epoch = 0
    best_loss = 999999999
    global_batch_idx = config['global_batch_idx']

    if config['load_checkpoint']:
        save_dict = torch.load(config['checkpoint_path'])
        epoch = save_dict['epoch'] + 1
        model.load_state_dict(save_dict['model'])
        optim.load_state_dict(save_dict['optim'])
        scheduler.load_state_dict(save_dict['scheduler'])
        best_loss = config['best_loss']
    
    patience = config['patience'] # if -1, run forever

    try:
        while True:
            step(model, optim, scheduler, train_loader, train_writer, logger, eval=False)
            with torch.no_grad():
                valid_loss = step(model, optim, scheduler, val_loader, valid_writer, logger, eval=True)
        
            config['epoch'] = epoch
            config['global_batch_idx'] = global_batch_idx
            save_dict = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optim': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': config
            }
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                config['best_loss'] = best_loss
                torch.save(save_dict, config['best_path'])
                logger.info(f"Best validation Loss: {best_loss:.6f}")
                patience = config['patience']
            elif patience != -1:
                patience -= 1
                logger.info(f"Patience: {patience}")
            
            yaml.dump(config, open(config['save_config'], 'w'), default_flow_style=False)
            torch.save(save_dict, config['checkpoint_path'])

            if patience <= 0:
                logger.info("Early stopping")
                break
            epoch += 1

    except KeyboardInterrupt as e:
        logger.info("Interrupted")
