import torch
import torch.nn as nn
import numpy as np
import warnings

def nll_loss(pred_mu, pred_cov, y, mean=True):
    pred_std = torch.sqrt(pred_cov)
    gaussian = torch.distributions.Normal(pred_mu, pred_std)
    nll = -gaussian.log_prob(y)
    if mean:
        nll = torch.mean(nll)

    return nll

def nll_metric(pred_mu, pred_cov, y):
    pred_mu = torch.from_numpy(pred_mu)
    pred_cov = torch.from_numpy(pred_cov)
    y = torch.from_numpy(y)
    pred_std = torch.sqrt(pred_cov)
    gaussian = torch.distributions.Normal(pred_mu, pred_std)
    nll = -gaussian.log_prob(y)
    nll = torch.mean(nll).cpu().detach().numpy()
    return nll

def mae_loss(y_pred, y_true, mean=True):
    loss = torch.abs(y_pred - y_true)
    loss[loss != loss] = 0
    if mean:
        loss = loss.mean()
    return loss

def mae_metric(y_pred, y_true, mean=True):
    loss = np.abs(y_pred - y_true)
    loss[loss != loss] = 0
    if mean:
        loss = loss.mean()
    return loss

def mse_loss(y_pred, y_true, mean=True):
    loss = (y_pred - y_true) ** 2
    loss[loss != loss] = 0
    if mean:
        loss = loss.mean()
    return loss

def mse_metric(y_pred, y_true, mean=True):
    loss = (y_pred - y_true)**2
    loss[loss != loss] = 0
    if mean:
        loss = loss.mean()
    return loss


def kl_div(prior_mu, prior_var, posterior_mu, posterior_var):
    kl_div = (torch.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / torch.exp(prior_var) - 1. + (prior_var - posterior_var)
    kl_div = 0.5 * kl_div.sum()
    return kl_div
    
def kld_gaussian_loss(z_mean_all, z_var_all, z_mean_context, z_var_context): 
    """Analytical KLD between 2 Gaussians."""
    mean_q, var_q, mean_p, var_p = z_mean_all, z_var_all, z_mean_context,  z_var_context
    std_q = torch.sqrt(var_q)
    std_p = torch.sqrt(var_p)
    p = torch.distributions.Normal(mean_p, std_p)
    q = torch.distributions.Normal(mean_q, std_q)
    return torch.mean(torch.sum(torch.distributions.kl_divergence(q, p),dim=1))


def norm_rmse_loss(y_pred, y_true, metric=False, mean=True):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if metric:
            norm_rmse = np.abs(np.sqrt((y_pred - y_true)**2 / y_true.shape[0]) / np.abs(y_true.mean(0)))
            norm_rmse[np.isinf(norm_rmse) | np.isnan(norm_rmse)] = 0
        else:
            norm_rmse = torch.abs(torch.sqrt((y_pred - y_true)**2 / y_true.shape[0]) / torch.abs(y_true.mean(0)))
            norm_rmse[torch.isinf(norm_rmse) | torch.isnan(norm_rmse)] = 0
            
        norm_rmse[norm_rmse > 1e8] = 0
        if mean:
            norm_rmse = norm_rmse.mean()
        return norm_rmse

class NegRLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(NegRLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y, np=False):
        if np:
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)

        xmean = torch.mean(x, dim=0)
        ymean = torch.mean(y, dim=0)
        ssxm = torch.mean( (x-xmean)**2, dim=0)
        ssym = torch.mean( (y-ymean)**2, dim=0 )
        ssxym = torch.mean( (x-xmean) * (y-ymean), dim=0 )
        r = ssxym / torch.sqrt(ssxm * ssym)

        return r
