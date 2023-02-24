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
from model import LatentModel

scaler = torch.cuda.amp.GradScaler()

# parser = argparse.ArgumentParser()
# parser.add_argument("seed", type=int, help="seed for random number generator")
# args = parser.parse_args()
# seed = args.seed
seed = 0

device = torch.device("cuda:7" if torch.cuda.is_available() else 'cpu')
print(device)

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
    norm_rmse_total = 0
    if not eval:
        model.train()
    else:
        model.eval()

    pbar = tqdm(loader, total=len(loader))
    for i, (x_, y_) in enumerate(pbar):
        x_ = x_[:, ::2]
        y_ = y_[:, ::2]
        batch_loss = 0
        batch_nll = 0
        batch_mae = 0
        batch_mse = 0
        batch_norm_rmse = 0

        n = 6
        for b in range(n):
            x = x_[:, b::n].to(device)
            y = y_[:, b::n].to(device)

            context_idxs, target_idxs = split_context_target(x, y, config['model']['context_percentage_low'], config['model']['context_percentage_high'])
            x_context = x[:, context_idxs, :]
            y_context = y[:, context_idxs]
            x_target = x[:, target_idxs, :]
            # y_target = y[:, target_idxs]

            with torch.cuda.amp.autocast():
                l2_output_mu, l2_output_cov, l2_z_mu_c, l2_z_cov_c, l2_z_mu_all, l2_z_cov_all = model(x_context, y_context, x_target, all_x=x, all_y=y)
                nll = nll_loss(l2_output_mu, l2_output_cov, y)
                mae = mae_loss(l2_output_mu, y)
                kld = kl_div(l2_z_mu_all, l2_z_cov_all, l2_z_mu_c, l2_z_cov_c)
                batch_loss += (nll + mae + kld) / (b + 1)

                batch_nll += nll.detach() / (b + 1)
                batch_mae += mae.detach() / (b + 1)
                batch_mse += mse_loss(l2_output_mu, y).detach() / (b + 1)
                batch_norm_rmse += norm_rmse_loss(l2_output_mu, y).detach() / (b + 1)

        if not eval:
            optim.zero_grad()
            scaler.scale(batch_loss).backward()
            scaler.step(optim)
            scaler.update()
            if i == 0:
                scheduler.step()

        mse_total += batch_mse.item()
        mae_total += batch_mae.item()
        norm_rmse_total += batch_norm_rmse.item()
        
        writer.add_scalar("nll", batch_nll.item(), global_batch_idx)
        writer.add_scalar("mae", batch_mae.item(), global_batch_idx)
        writer.add_scalar("mse", batch_mse.item(), global_batch_idx)
        writer.add_scalar("norm_rmse", batch_norm_rmse, global_batch_idx)
        writer.flush()

        pbar.set_description(f"Epoch {epoch} {split}")
        pbar.set_postfix_str(f"MSE: {batch_mse.item():.6f} MAE: {batch_mae.item():.6f} NRMSE: {batch_norm_rmse:.6f}")
        if not eval:
            global_batch_idx += 1

    mse_total /= i+1
    mae_total /= i+1
    norm_rmse_total /= i+1
    logger.info(f"EPOCH: {epoch} {split} - MSE: {mse_total:.6f} MAE: {mae_total:.6f} NRMSE: {norm_rmse_total:.6f}")

    return mae_total


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

    # Format like Aziz
    fidelity = "low"

    # 4/6/2023 -- 3/31/2004 / 6/30/2004 - 9/22/2004
    l1_x_data = sorted(glob.glob(f"{config['data_dir']}/batch_{fidelity}_data/inputs_*"), key=sort_batch_fn)
    l1_y_data = sorted(glob.glob(f"{config['data_dir']}/batch_{fidelity}_data/outputs_*"), key=sort_batch_fn)
    l1_x_train = l1_x_data[train_idx_start:train_idx_end]
    l1_y_train = l1_y_data[train_idx_start:train_idx_end]
    l1_x_val = l1_x_data[train_idx_end:valid_idx_end]
    l1_y_val = l1_y_data[train_idx_end:valid_idx_end]

    x_scaler_minmax = dill.load(open(f"{config['data_dir']}/scaler/{fidelity}_x_minmax_scaler.pkl", 'rb'))
    y_scaler_minmax = dill.load(open(f"{config['data_dir']}/scaler/{fidelity}_y_minmax_scaler.pkl", 'rb'))

    # x_scaler_minmax = None
    # y_scaler_minmax = None
    # l1_x_scaler_minmax = dill.load(open(f"{config['data_dir']}/scaler/{fidelity}_x_minmax_scaler.pkl", 'rb'))
    # l1_y_scaler_minmax = dill.load(open(f"{config['data_dir']}/scaler/{fidelity}_y_minmax_scaler.pkl", 'rb'))

    # Change to first 32 variables
    y_scaler_minmax.min = y_scaler_minmax.min[:26]
    y_scaler_minmax.max = y_scaler_minmax.max[:26]

    train_dataset = l2Dataset(l1_x_train, l1_y_train, x_scaler=x_scaler_minmax, y_scaler=y_scaler_minmax, variables=26)
    # train_dataset = l2Dataset(l1_x_train, l1_y_train, x_scaler=x_scaler_minmax, y_scaler=y_scaler_minmax)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    val_dataset = l2Dataset(l1_x_val, l1_y_val, x_scaler=x_scaler_minmax, y_scaler=y_scaler_minmax, variables=26)
    # val_dataset = l2Dataset(l1_x_val, l1_y_val, x_scaler=x_scaler_minmax, y_scaler=y_scaler_minmax)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    model = LatentModel(config['model']).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    # optim = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = StepLR(optim, step_size=config['decay_steps'], gamma=config['decay_rate'])
    # scheduler = CyclicLR(optim, config['base_lr'], config['max_lr'], step_size_up=2000, step_size_down=None, mode='exp_range', \
                        # gamma=config['gamma'], scale_fn=None, scale_mode='iterations', cycle_momentum=False, base_momentum=0.8, \
                        # max_momentum=0.9, last_epoch=- 1, verbose=False)

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
