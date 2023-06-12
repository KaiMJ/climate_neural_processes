import torch
import torch.nn as nn
import torch.optim as optim

# Create the model
class Model(nn.Module):
    def __init__(self, config):
        num_layers = config['num_layers']
        num_nodes = config['num_nodes']
        dropout_rate = config['dropout_rate']
        leaky_relu_slope = config['leaky_relu_slope']
        input_dim = config["l2_input_dim"]
        output_dim = config["l2_output_dim"]

        super(Model, self).__init__()

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, num_nodes))
            else:
                layers.append(nn.Linear(num_nodes, num_nodes))
            layers.append(nn.BatchNorm1d(1))
            layers.append(nn.LeakyReLU(leaky_relu_slope))
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(num_nodes, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
