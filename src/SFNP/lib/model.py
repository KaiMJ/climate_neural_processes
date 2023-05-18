import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath("__file__")), '../..'))


class MLP_Encoder(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_layers,
                 hidden_dim,
                 leaky_relu_slope,
                 dropout_rate):

        nn.Module.__init__(self)

        layers = []
        for i in range(hidden_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(1))
            layers.append(nn.LeakyReLU(leaky_relu_slope))
            layers.append(nn.Dropout(dropout_rate))
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
                 hidden_layers,
                 hidden_dim,
                 leaky_relu_slope,
                 dropout_rate):

        nn.Module.__init__(self)

        layers = []
        for i in range(hidden_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(leaky_relu_slope))
            layers.append(nn.Dropout(dropout_rate))
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
                 hidden_layers,
                 hidden_dim,
                 leaky_relu_slope,
                 dropout_rate):

        nn.Module.__init__(self)

        layers = []
        for i in range(hidden_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(1))
            layers.append(nn.LeakyReLU(leaky_relu_slope))
            layers.append(nn.Dropout(dropout_rate))
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

        self.input_dim = int(config['input_dim'])  # fully connected, 50+3
        self.output_dim = int(config['output_dim'])
        self.hidden_dim = int(config['hidden_dim'])
        self.encoder_output_dim = self.z_dim
        self.decoder_input_dim = self.z_dim + self.input_dim

        self.leaky_relu_slope = config['leaky_relu_slope']
        self.dropout_rate = config['dropout_rate']

        self.l2_encoder_model = MLP_Encoder(
            self.input_dim+self.output_dim, self.encoder_output_dim, self.hidden_layers, self.hidden_dim, self.leaky_relu_slope, self.dropout_rate)
        self.l2_z_encoder_model = MLP_ZEncoder(
            self.z_dim, self.z_dim, self.z_hidden_layers, self.z_hidden_dim,
            self.leaky_relu_slope, self.dropout_rate)
        self.l2_decoder_model = MLP_Decoder(self.decoder_input_dim, self.output_dim, self.hidden_layers, self.hidden_dim,
                                            self.leaky_relu_slope, self.dropout_rate)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def sample_z(self, mean, var, n=1):
        """Reparameterisation trick."""
        eps = torch.autograd.Variable(var.data.new(
            n, var.size(0), var.size(1)).normal_()).to(mean.device)
        std = torch.sqrt(var)

        return torch.unsqueeze(mean, dim=0) + torch.unsqueeze(std, dim=0) * eps

    def xy_to_r(self, x, y):
        r_mu = self.l2_encoder_model(torch.cat([x, y], dim=-1))

        return r_mu

    def z_to_y(self, x, zs):
        output = self.l2_decoder_model(torch.cat([x, zs], dim=-1))

        return output

    def mean_z_agg(self, r):
        r_agg = torch.mean(r, dim=0)
        z_mu, z_cov = self.l2_z_encoder_model(r_agg)
        return z_mu, z_cov

    def forward(self, x_context, y_context, x_target, x_all=None, y_all=None):
        l2_r_c = self.xy_to_r(x_context, y_context)
        l2_z_mu_c, l2_z_cov_c = self.mean_z_agg(l2_r_c)

        if x_all is not None:
            l2_r_all = self.xy_to_r(x_all, y_all)
            l2_z_mu_all, l2_z_cov_all = self.mean_z_agg(l2_r_all)
            l2_zs = self.sample_z(l2_z_mu_all, l2_z_cov_all, x_target.size(0))
            l2_output_mu, l2_output_cov = self.z_to_y(x_target, l2_zs)
            return l2_output_mu, l2_output_cov, l2_z_mu_all, l2_z_cov_all, l2_z_mu_c, l2_z_cov_c

        else:
            l2_zs = self.sample_z(l2_z_mu_c, l2_z_cov_c, x_target.size(0))
            l2_output_mu, l2_output_cov = self.z_to_y(x_target, l2_zs)
            return l2_output_mu, l2_output_cov
