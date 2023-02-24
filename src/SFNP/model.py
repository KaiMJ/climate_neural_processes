import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath("__file__")), '../..'))

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")
class MLP_Encoder(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_layers=2, hidden_dim=32):
        super().__init__()

        layers = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
        for _ in range(hidden_layers - 1):
            # layers.append(nn.LayerNorm(hidden_dim))
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        # layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        output = self.model(x)
        mean = self.mean_out(output)
        return mean

class MLP_ZEncoder(nn.Module):
    """Takes an r representation and produces the mean & variance of the 
    normally distributed function encoding, z."""
    def __init__(self, 
            in_dim, 
            out_dim,
            hidden_layers=2,
            hidden_dim=32):

        nn.Module.__init__(self)

        layers = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
        for _ in range(hidden_layers - 1):
            # layers.append(nn.LayerNorm(hidden_dim))
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        # layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(hidden_dim, out_dim)
        self.cov_out = nn.Linear(hidden_dim, out_dim)
        self.cov_m = nn.Sigmoid()


    def forward(self, inputs):
        output = self.model(inputs)
        mean = self.mean_out(output)
        cov = 0.1+0.9*self.cov_m(self.cov_out(output))

        return mean, cov

class MLP_Decoder(nn.Module):

    def __init__(self, 
            in_dim, 
            out_dim, 
            hidden_layers=2,
            hidden_dim=32):

        nn.Module.__init__(self)

        layers = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
        for _ in range(hidden_layers - 1):
            # layers.append(nn.LayerNorm(hidden_dim))
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        # layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(hidden_dim, out_dim)
        self.cov_out = nn.Linear(hidden_dim, out_dim)
        self.cov_m = nn.Softplus()

    def forward(self, x):
        output = self.model(x)
        mean = self.mean_out(output)
        cov = self.cov_m(self.cov_out(output))
        return mean, cov


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_layers = int(config['hidden_layers'])
        self.z_hidden_layers = int(config['z_hidden_layers'])
        self.z_hidden_dim = int(config['z_hidden_dim'])
        self.z_dim = int(config['z_dim'])

        self.input_dim = int(config['input_dim']) #fully connected, 50+3
        self.output_dim = int(config['output_dim'])
        self.hidden_dim = int(config['hidden_dim'])
        self.encoder_output_dim = self.z_dim
        self.decoder_input_dim = self.z_dim + self.input_dim
        self.context_percentage_low = float(config['context_percentage_low'])
        self.context_percentage_high = float(config['context_percentage_high'])

        self.l2_encoder_model = MLP_Encoder(self.input_dim+self.output_dim, self.encoder_output_dim, self.hidden_layers, self.hidden_dim)
        self.l2_z_encoder_model = MLP_ZEncoder(self.z_dim, self.z_dim, self.z_hidden_layers, self.z_hidden_dim)
        self.l2_decoder_model = MLP_Decoder(self.decoder_input_dim, self.output_dim, self.hidden_layers, self.hidden_dim)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def split_context_target(self, x, y, context_percentage_low, context_percentage_high):
        """Helper function to split randomly into context and target"""
        context_percentage = np.random.uniform(context_percentage_low,context_percentage_high)

        n_context = int(x.shape[0]*context_percentage)
        ind = np.arange(x.shape[0])
        mask = np.random.choice(ind, size=n_context, replace=False)

        return x[mask], y[mask], mask


    def sample_z(self, mean, var, n=1):
        """Reparameterisation trick."""
        eps = torch.autograd.Variable(var.data.new(n,var.size(0),var.size(1)).normal_()).to(mean.device)
        std = torch.sqrt(var)

        return torch.unsqueeze(mean, dim=0) + torch.unsqueeze(std, dim=0) * eps

    def xy_to_r(self, x, y):
        r_mu = self.l2_encoder_model(torch.cat([x, y],dim=-1))

        return r_mu

    def z_to_y(self, x, zs):
        output = self.l2_decoder_model(torch.cat([x,zs], dim=-1))

        return output

    def mean_z_agg(self, r):
        # r_mu = torch.swapaxes(r_mu,0,1)
        # r_cov = torch.swapaxes(r_cov,0,1)

        r_agg = torch.mean(r,dim=0)
        z_mu, z_cov = self.l2_z_encoder_model(r_agg)
        return z_mu, z_cov


    def forward(self, l2_x_all=None, l2_y_all=None, l2_z_mu_all=None, l2_z_cov_all=None, return_idxs=False):

        if l2_y_all is not None:
            l2_x_c,l2_y_c,l2_x_t,l2_y_t, idxs = self.split_context_target(l2_x_all,l2_y_all, self.context_percentage_low, self.context_percentage_high)

            l2_r_all = self.xy_to_r(l2_x_all, l2_y_all)
            l2_r_c = self.xy_to_r(l2_x_c, l2_y_c)
            l2_z_mu_all, l2_z_cov_all = self.mean_z_agg(l2_r_all)
            l2_z_mu_c, l2_z_cov_c = self.mean_z_agg(l2_r_c)

            l2_zs = self.sample_z(l2_z_mu_all, l2_z_cov_all, l2_x_t.size(0))
            l2_output_mu, l2_output_cov = self.z_to_y(l2_x_t,l2_zs)
            l2_truth = l2_y_t
            if not return_idxs:
                return l2_output_mu, l2_output_cov, l2_z_mu_all, l2_z_cov_all, l2_z_mu_c, l2_z_cov_c
            if return_idxs:
                return l2_output_mu, l2_output_cov, l2_z_mu_all, l2_z_cov_all, l2_z_mu_c, l2_z_cov_c, idxs

        else:
            l2_output_mu, l2_output_cov = None, None
            if l2_x_all is not None:
                l2_zs = self.sample_z(l2_z_mu_all, l2_z_cov_all, l2_x_all.size(0))
                l2_output_mu, l2_output_cov = self.z_to_y(l2_x_all, l2_zs)

            return l2_output_mu, l2_output_cov
