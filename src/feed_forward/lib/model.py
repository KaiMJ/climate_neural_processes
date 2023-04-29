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
