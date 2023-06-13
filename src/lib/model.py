import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN_Model(nn.Module):
    def __init__(self, config):
        num_layers = config['num_layers']
        num_nodes = config['num_nodes']
        dropout_rate = config['dropout_rate']
        leaky_relu_slope = config['leaky_relu_slope']
        input_dim = config["l2_input_dim"]
        output_dim = config["l2_output_dim"]

        super(DNN_Model, self).__init__()

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

class MF_DNN_Model(nn.Module):
    def __init__(self, config):
        num_layers = config['num_layers']
        num_nodes = config['num_nodes']
        dropout_rate = config['dropout_rate']
        leaky_relu_slope = config['leaky_relu_slope']
        input_dim = config["l2_input_dim"]
        output_dim = config["l2_output_dim"]

        super(MF_DNN_Model, self).__init__()

        low_layers = []
        for i in range(num_layers):
            if i == 0:
                low_layers.append(nn.Linear(input_dim, num_nodes))
            else:
                low_layers.append(nn.Linear(num_nodes, num_nodes))
            low_layers.append(nn.BatchNorm1d(1))
            low_layers.append(nn.LeakyReLU(leaky_relu_slope))
            low_layers.append(nn.Dropout(dropout_rate))

        self.low_model = nn.Sequential(*low_layers)
        
        self.low_final = nn.Linear(num_nodes, output_dim)

        high_layers = []
        for i in range(num_layers):
            if i == 0:
                high_layers.append(nn.Linear(input_dim + num_nodes, num_nodes))
            else:
                high_layers.append(nn.Linear(num_nodes, num_nodes))
            high_layers.append(nn.BatchNorm1d(1))
            high_layers.append(nn.LeakyReLU(leaky_relu_slope))
            high_layers.append(nn.Dropout(dropout_rate))

        high_layers.append(nn.Linear(num_nodes, output_dim))
        self.high_model = nn.Sequential(*high_layers)

    def forward(self, l1_x, l2_x):
        l1_out = self.low_model(l1_x)
        out = torch.cat((l1_out, l2_x), dim=-1)
        l1_out = self.low_final(l1_out)
        l2_out = self.high_model(out)

        return l1_out, l2_out


# ---------------------------------------------------- Transformer ----------------------------------------------------

class Head(nn.Module):

    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape # 24, , 132
        k = self.key(x)   # B, T, C
        q = self.query(x) # B, T, C
        
        wei = q @ k.transpose(-2, -1) * C ** 0.5 # B, T, T
        wei = F.softmax(wei, dim=-1)  # B, T, T
        wei = self.dropout(wei)       # B, T, T

        v = self.value(x) # B, T, C
        out = wei @ v     # B, T, C
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        num_heads = config['num_heads']
        n_embd = config['n_embd']
        dropout = config['dropout']
        head_size = n_embd // num_heads
        
        self.embed_input = nn.Linear(132, n_embd)
        self.heads = nn.ModuleList([Head(n_embd, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, 136)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Layer norm before Mutli-Head
        out = self.embed_input(x)
        out = torch.cat([h(out) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# https://github.com/gazelle93/Transformer-Various-Positional-Encoding/blob/main/positional_encoders.py
class AbsolutePositionalEncoder(nn.Module):
    def __init__(self, emb_dim, max_position=512):
        super(AbsolutePositionalEncoder, self).__init__()
        self.position = torch.arange(max_position).unsqueeze(1)

        self.positional_encoding = torch.zeros(1, max_position, emb_dim)

        _2i = torch.arange(0, emb_dim, step=2).float()

        # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 0::2] = torch.sin(self.position / (10000 ** (_2i / emb_dim)))

        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 1::2] = torch.cos(self.position / (10000 ** (_2i / emb_dim)))

    def forward(self, x):
        # batch_size, input_len, embedding_dim
        batch_size, seq_len, _ = x.size()

        return self.positional_encoding[:batch_size, :seq_len, :]

# https://github.com/karpathy/nanoGPT/blob/master/model.py#L17

@torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['num_heads'] == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'])
        # output projection
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'])
        # regularization
        self.attn_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("bias", torch.tril(torch.ones(config['block_size'], config['block_size']))
        #               max_iter              .view(1, 1, config['block_size'], config['block_size']))
        self.num_heads = config['num_heads']
        self.n_embd = config['n_embd']

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config['n_embd'], 4 * config['n_embd'])
        self.c_proj  = nn.Linear(4 * config['n_embd'], config['n_embd'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['n_embd'])
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['n_embd'])
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer_Model(nn.Module):

    def __init__(self, config):
        super().__init__()

        input_dim = config['input_dim']
        output_dim = config['output_dim']
        n_embd = config['n_embd']
        n_layers = config['attention_layers']
        dropout = config['dropout']

        # (B, 96*144, 132)
        self.embedding = nn.Linear(input_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        # self.positional_embedding = nn.Linear(4, n_embd)
        self.heads = nn.ModuleList([Block(config) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(n_embd)
        self.proj = nn.Linear(n_embd, output_dim)

    def forward(self, x, pos=None):
        # B, T, C
        out = self.embedding(x) # [B T C]
        out = self.dropout(out)
        for block in self.heads:
            out = block(out)
        out = self.layer_norm(out)
        out = self.proj(out)

        return out


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

# ---------------------------------------------------- SFNP ----------------------------------------------------

class SFNP_Model(nn.Module):
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


class MLP_Encoder(nn.Module):

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

    def forward(self, x):
        output = self.model(x)
        mean = self.mean_out(output)
        return mean

class L1_MLP_ZEncoder(nn.Module):
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

class L2_MLP_ZEncoder(nn.Module):
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


    def forward(self, inputs, z):
        inputs = torch.cat([inputs, z], dim=-1)
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
        # cov = torch.exp(self.cov_out(output))

        return mean, cov

class MFNP_MLP_Z1Z2_Encoder(nn.Module):

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
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(hidden_dim, out_dim)
        self.cov_out = nn.Linear(hidden_dim, out_dim)
        self.cov_m = nn.Sigmoid()
        # self.cov_m = nn.ELU()

    def forward(self, x):
        # x = torch.swapaxes(x, -2, -1)
        output = self.model(x)
        mean = self.mean_out(output)
        cov = 0.1+ 0.9*self.cov_m(self.cov_out(output))
        # mean = torch.swapaxes(mean, -2, -1)
        # cov = torch.swapaxes(cov, -2, -1)

        return mean, cov

# ---------------------------------------------------- MFNP ----------------------------------------------------

class MFNP_Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_layers = config['hidden_layers']
        self.z_hidden_layers = config['z_hidden_layers']
        self.z_hidden_dim = config['z_hidden_dim']
        self.z_dim = config['z_dim']

        self.l1_input_dim = config['l1_input_dim']
        self.l2_input_dim = config['l2_input_dim']

        self.l1_output_dim = config['l1_output_dim']
        self.l2_output_dim = config['l2_output_dim']

        self.hidden_dim = config['hidden_dim']

        self.encoder_output_dim = self.z_dim
        self.l1_decoder_input_dim = self.z_dim + self.l1_input_dim
        self.l2_decoder_input_dim = self.z_dim + self.l2_input_dim
        self.l1_input_dim = config['l1_input_dim']
        self.l2_input_dim = config['l2_input_dim']
        self.l1_encoder_model = MLP_Encoder(self.l1_input_dim+self.l1_output_dim, self.encoder_output_dim, self.hidden_layers, self.hidden_dim)
        self.l2_encoder_model = MLP_Encoder(self.l2_input_dim+self.l2_output_dim, self.encoder_output_dim, self.hidden_layers, self.hidden_dim)
        self.l1_decoder_model = MLP_Decoder(self.l1_decoder_input_dim, self.l1_output_dim, self.hidden_layers, self.hidden_dim)
        self.l2_decoder_model = MLP_Decoder(self.l2_decoder_input_dim, self.l2_output_dim, self.hidden_layers, self.hidden_dim)

        self.l1_z_encoder_model = L1_MLP_ZEncoder(self.z_dim, self.z_dim, self.z_hidden_layers, self.z_hidden_dim)
        self.l2_z_encoder_model = L2_MLP_ZEncoder(self.z_dim+self.z_dim, self.z_dim, self.z_hidden_layers, self.z_hidden_dim)
        self.z2_z1_agg = MFNP_MLP_Z1Z2_Encoder(self.z_dim, self.z_dim)

    def sample_z(self, mean, var, n=1):
        """Reparameterisation trick."""
        eps = torch.autograd.Variable(var.data.new(n,var.size(0),var.size(1)).normal_()).to(mean.device)

        std = torch.sqrt(var)
        return torch.unsqueeze(mean, dim=0) + torch.unsqueeze(std, dim=0) * eps

    def xy_to_r(self, x, y, level):
        if level == 1:
            r_mu = self.l1_encoder_model(torch.cat([x, y],dim=-1))
        elif level == 2:
            r_mu = self.l2_encoder_model(torch.cat([x, y],dim=-1))

        return r_mu

    def z_to_y(self, x, zs, level):
        if level == 1:
            output = self.l1_decoder_model(torch.cat([x,zs], dim=-1))

        elif level == 2:
            output = self.l2_decoder_model(torch.cat([x,zs], dim=-1))

        return output

    def mean_z_agg(self, r, level, z_mu=None):
        if level == 1:
            r_agg = torch.mean(r,dim=0)
            z_mu, z_cov = self.l1_z_encoder_model(r_agg)
        if level == 2:
            r_agg = torch.mean(r,dim=0)
            z_mu, z_cov = self.l2_z_encoder_model(r_agg, z_mu)

        return z_mu, z_cov


    def forward(self, l2_x_context, l2_y_context, l2_x_target, l1_x_context, l1_y_context, l1_x_target, \
                l2_x_all=None, l2_y_all=None, l1_x_all=None, l1_y_all=None):
        """
            Return idxs for visualization purposes.

            If l2_y_all is not None, then we are training the model.
            If l2_y_all is None, then we are testing the model.

            Prediction is made across both context and target.
        """
        l1_r_c = self.xy_to_r(l1_x_context, l1_y_context, level=1)
        l1_z_mu_c, l1_z_cov_c = self.mean_z_agg(l1_r_c,level=1)
        l2_z_mu_c_0, l2_z_cov_c_0 = self.z2_z1_agg(l1_z_mu_c)

        l2_r_c = self.xy_to_r(l2_x_context, l2_y_context, level=2)
        l2_z_mu_c, l2_z_cov_c = self.mean_z_agg(l2_r_c,level=2, z_mu=l2_z_mu_c_0)

        if l2_y_all is not None:
            l1_r_all = self.xy_to_r(l1_x_all, l1_y_all, level=1)
            l1_z_mu_all, l1_z_cov_all = self.mean_z_agg(l1_r_all,level=1)
            l2_z_mu_all_0, l2_z_cov_all_0 = self.z2_z1_agg(l1_z_mu_all)
            l2_r_all = self.xy_to_r(l2_x_all, l2_y_all, level=2)
            l2_z_mu_all, l2_z_cov_all = self.mean_z_agg(l2_r_all,level=2, z_mu=l2_z_mu_all_0)

            #sample z
            l1_zs = self.sample_z(l1_z_mu_all, l1_z_cov_all, l1_x_target.size(0))
            l2_zs = self.sample_z(l2_z_mu_all, l2_z_cov_all, l2_x_target.size(0))
            #l1_decoder, l2_decoder
            l1_output_mu, l1_output_cov = self.z_to_y(l1_x_target, l1_zs, level=1)
            l2_output_mu, l2_output_cov = self.z_to_y(l2_x_target, l2_zs, level=2)

            return l2_output_mu, l2_output_cov, l1_output_mu, l1_output_cov,  \
                    l1_z_mu_all, l1_z_cov_all, l1_z_mu_c, l1_z_cov_c, \
                    l2_z_mu_all, l2_z_cov_all, l2_z_mu_c, l2_z_cov_c
        else:
            l2_zs = self.sample_z(l2_z_mu_c, l2_z_cov_c, l2_x_target.size(0))
            l2_output_mu, l2_output_cov = self.z_to_y(l2_x_target, l2_zs, level=2)

            return l2_output_mu, l2_output_cov
        
# ---------------------------------------------------- SFATNP ----------------------------------------------------

class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class SF_LatentEncoder(nn.Module):
    """
    Latent Encoder [For prior, posterior]
    """

    def __init__(self, config):
        super(SF_LatentEncoder, self).__init__()
        input_dim = config['l2_input_dim']
        output_dim = config['l2_output_dim']
        hidden_dim = config['hidden_dim']
        attention_layers = config['attention_layers']

        self.input_projection = Linear(input_dim+output_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attentions = nn.ModuleList(
            [Attention(config) for _ in range(attention_layers)])
        self.penultimate_layer = Linear(hidden_dim, hidden_dim, w_init='relu')
        self.mu = Linear(hidden_dim, hidden_dim)
        self.log_sigma = Linear(hidden_dim, hidden_dim)

    def forward(self, x, y):
        # concat location (x) and value (y)
        encoder_input = torch.cat([x, y], dim=-1)

        # project vector with dimension 132+136 --> hidden_dim
        encoder_input = self.input_projection(encoder_input)
        encoder_input = self.layer_norm(encoder_input)

        # self attention layer
        for attention in self.self_attentions:
            encoder_input, _ = attention(
                encoder_input, encoder_input, encoder_input)

        # mean
        hidden = encoder_input.mean(dim=1)
        hidden = torch.relu(self.penultimate_layer(hidden))

        # get mu and sigma
        mu = self.mu(hidden)
        log_sigma = self.log_sigma(hidden)

        # reparameterization trick
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        # return distribution
        return mu, log_sigma, z

class SF_DeterministicEncoder(nn.Module):
    """
    Deterministic Encoder [r]
    """

    def __init__(self, config):
        super(SF_DeterministicEncoder, self).__init__()
        input_dim = config['l2_input_dim']
        output_dim = config['l2_output_dim']
        hidden_dim = config['hidden_dim']
        attention_layers = config['attention_layers']

        self.self_attentions = nn.ModuleList(
            [Attention(config) for _ in range(attention_layers)])
        self.cross_attentions = nn.ModuleList(
            [Attention(config) for _ in range(attention_layers)])
        self.input_projection = Linear(input_dim+output_dim, hidden_dim)
        self.context_projection = Linear(input_dim, hidden_dim)
        self.target_projection = Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, context_x, context_y, target_x):
        # concat context location (x), context value (y)
        encoder_input = torch.cat([context_x, context_y], dim=-1)

        # project vector with dimension 132+136 --> num_hidden
        encoder_input = self.input_projection(encoder_input)
        encoder_input = self.layer_norm(encoder_input)

        # self attention layer
        for attention in self.self_attentions:
            encoder_input, _ = attention(
                encoder_input, encoder_input, encoder_input)

        # query: target_x, key: context_x, value: representation
        query = self.target_projection(target_x)
        keys = self.context_projection(context_x)

        # cross attention layer
        for attention in self.cross_attentions:
            query, _ = attention(keys, encoder_input, query)

        return query

class SF_Decoder(nn.Module):
    """
    Decoder for generation 
    """

    def __init__(self, config):
        num_hidden = config['hidden_dim']
        input_dim = config['l2_input_dim']
        output_dim = config['l2_output_dim']
        hidden_layers = config['hidden_layers']
        attention_layers = config['attention_layers']

        super(SF_Decoder, self).__init__()
        self.target_projection = Linear(input_dim, num_hidden)
        self.linears = nn.ModuleList([Linear(
            num_hidden * 3, num_hidden * 3, w_init='relu') for _ in range(hidden_layers-1)])
        self.hidden_projection = Linear(num_hidden * 3, num_hidden)
        # self.layer_norm = nn.LayerNorm(num_hidden * 3)
        self.self_attentions = nn.ModuleList(
            [Attention(config) for _ in range(attention_layers)])
        self.mean_out = nn.Linear(num_hidden, output_dim)
        self.cov_out = nn.Linear(num_hidden, output_dim)
        self.cov_m = nn.Softplus()
        self.relu = nn.LeakyReLU(config['leaky_relu_slope'])

    def forward(self, r, z, target_x):
        batch_size, num_targets, _ = target_x.size()
        # project vector with dimension 2 --> num_hidden
        target_x = self.target_projection(target_x)

        # concat all vectors (r,z,target_x)
        hidden = torch.cat([torch.cat([r, z], dim=-1), target_x], dim=-1)

        # mlp layers
        for linear in self.linears:
            hidden = self.relu(linear(hidden))
        hidden = self.hidden_projection(hidden)

        # self attention layer
        for attention in self.self_attentions:
            hidden, _ = attention(hidden, hidden, hidden)

        # get mu and sigma
        mean = self.mean_out(hidden)
        cov = self.cov_m(self.cov_out(hidden))
        return mean, cov

class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism (dot attention)
    """

    def __init__(self, num_hidden_k, dropout):
        """
        :param num_hidden_k: dimension of hidden 
        """
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=dropout)

    def forward(self, key, value, query):
        # Get attention score
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)
        attn = torch.softmax(attn, dim=-1)

        # Dropout
        attn = self.attn_dropout(attn)
        # Get Context Vector
        result = torch.bmm(attn, value)

        return result, attn

class Attention(nn.Module):
    """
    Attention Network
    """

    def __init__(self, config):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads 
        """
        super(Attention, self).__init__()

        hidden_dim = config['hidden_dim']
        num_heads = config['num_heads']
        dropout = config['dropout_rate']
        self.num_hidden = hidden_dim
        self.num_hidden_per_attn = hidden_dim // num_heads
        self.h = num_heads

        self.key = Linear(hidden_dim, hidden_dim, bias=False)
        self.value = Linear(hidden_dim, hidden_dim, bias=False)
        self.query = Linear(hidden_dim, hidden_dim, bias=False)
        self.multihead = MultiheadAttention(self.num_hidden_per_attn, dropout)

        self.residual_dropout = nn.Dropout(p=dropout)

        self.final_linear = Linear(hidden_dim * 2, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, key, value, query):

        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        residual = query

        # Make multihead
        key = self.key(key).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        value = self.value(value).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)

        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)

        # Get context vector
        result, attns = self.multihead(key, value, query)

        # Concatenate all multihead context vector
        result = result.view(self.h, batch_size, seq_q, self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)

        # Concatenate context vector with input (most important)
        result = torch.cat([residual, result], dim=-1)

        # Final linear
        result = self.final_linear(result)

        # Residual dropout & connection
        result = self.residual_dropout(result)
        result = result + residual

        # Layer normalization
        result = self.layer_norm(result)

        return result, attns

class SF_ATTN_Model(nn.Module):
    """
    Latent Model (Attentive Neural Process)
    """

    def __init__(self, config):
        super(SF_ATTN_Model, self).__init__()
        self.latent_encoder = SF_LatentEncoder(config)
        self.deterministic_encoder = SF_DeterministicEncoder(config)
        self.decoder = SF_Decoder(config)

    def forward(self, x_context, y_context, x_target, y_target=None):
        l2_z_mu_c, l2_z_cov_c, z = self.latent_encoder(x_context, y_context)

        if y_target is not None:
            l2_z_mu_all, l2_z_cov_all, z = self.latent_encoder(
                x_target, y_target)

        z = z.unsqueeze(1).repeat(1, x_target.size(1), 1)  # [B, T_target, H]
        r = self.deterministic_encoder(
            x_context, y_context, x_target)  # [B, T_target, H]
        l2_output_mu, l2_output_cov = self.decoder(r, z, x_target)
        import torch
        if torch.isnan(l2_output_mu).any():
            print('here')
        if y_target is not None:
            return l2_output_mu, l2_output_cov, l2_z_mu_c, l2_z_cov_c, l2_z_mu_all, l2_z_cov_all
        else:
            return l2_output_mu, l2_output_cov

# ---------------------------------------------------- MF_ATTNNP ----------------------------------------------------

class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class LatentEncoder(nn.Module):
    """
    Latent Encoder [For prior, posterior]
    """

    def __init__(self, config, level=1):
        super(LatentEncoder, self).__init__()
        if level == 1:
            input_dim = config['l1_input_dim']
            output_dim = config['l1_output_dim']
        if level == 2:
            input_dim = config['l2_input_dim']
            output_dim = config['l2_output_dim']
            self.l1z_l2z_encoder = MLP_Z1Z2_Encoder(config)
        hidden_dim = config['hidden_dim']

        attention_layers = config['attention_layers']

        self.input_projection = Linear(input_dim+output_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attentions = nn.ModuleList(
            [Attention(config) for _ in range(attention_layers)])
        self.penultimate_layer = Linear(hidden_dim, hidden_dim, w_init='relu')
        self.mu = Linear(hidden_dim, hidden_dim)
        self.log_sigma = Linear(hidden_dim, hidden_dim)

    def forward(self, x, y, l_z=None):
        # concat location (x) and value (y)
        encoder_input = torch.cat([x, y], dim=-1)

        # project vector with dimension 132+136 --> hidden_dim
        encoder_input = self.input_projection(encoder_input)
        encoder_input = self.layer_norm(encoder_input)

        # self attention layer
        for attention in self.self_attentions:
            encoder_input, _ = attention(
                encoder_input, encoder_input, encoder_input)

        # mean
        hidden = encoder_input.mean(dim=1)
        hidden = torch.relu(self.penultimate_layer(hidden))

        # z_mu combine with hidden if level==2
        if l_z is not None:
            hidden = self.l1z_l2z_encoder(hidden, l_z)

        # get mu and sigma
        mu = self.mu(hidden)
        log_sigma = self.log_sigma(hidden)

        # reparameterization trick
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        # return distribution
        return mu, log_sigma, z


class DeterministicEncoder(nn.Module):
    """
    Deterministic Encoder [r]
    """

    def __init__(self, config, level=1):
        super(DeterministicEncoder, self).__init__()
        if level == 1:
            input_dim = config['l1_input_dim']
            output_dim = config['l1_output_dim']
        if level == 2:
            input_dim = config['l2_input_dim']
            output_dim = config['l2_output_dim']
            self.l1r_l2r_encoder = MLP_Z1Z2_Encoder(config)
        hidden_dim = config['hidden_dim']
        attention_layers = config['attention_layers']

        self.self_attentions = nn.ModuleList(
            [Attention(config) for _ in range(attention_layers)])
        self.cross_attentions = nn.ModuleList(
            [Attention(config) for _ in range(attention_layers)])
        self.input_projection = Linear(input_dim+output_dim, hidden_dim)
        self.context_projection = Linear(input_dim, hidden_dim)
        self.target_projection = Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, context_x, context_y, target_x, l_r=None):
        # concat context location (x), context value (y)
        encoder_input = torch.cat([context_x, context_y], dim=-1)

        # project vector with dimension 132+136 --> num_hidden
        encoder_input = self.input_projection(encoder_input)
        encoder_input = self.layer_norm(encoder_input)

        # self attention layer
        for attention in self.self_attentions:
            encoder_input, _ = attention(
                encoder_input, encoder_input, encoder_input)

        # query: target_x, key: context_x, value: representation
        query = self.target_projection(target_x)
        keys = self.context_projection(context_x)

        # cross attention layer
        for attention in self.cross_attentions:
            query, _ = attention(keys, encoder_input, query)

        if l_r is not None:
            query = self.l1r_l2r_encoder(query, l_r)

        return query


class Decoder(nn.Module):
    """
    Decoder for generation 
    """

    def __init__(self, config, level=1):
        num_hidden = config['hidden_dim']
        if level == 1:
            input_dim = config['l1_input_dim']
            output_dim = config['l1_output_dim']
        if level == 2:
            input_dim = config['l2_input_dim']
            output_dim = config['l2_output_dim']
        hidden_layers = config['hidden_layers']
        attention_layers = config['attention_layers']

        super(Decoder, self).__init__()
        self.target_projection = Linear(input_dim, num_hidden)
        self.linears = nn.ModuleList([Linear(
            num_hidden * 3, num_hidden * 3, w_init='relu') for _ in range(hidden_layers-1)])
        self.hidden_projection = Linear(num_hidden * 3, num_hidden)
        # self.layer_norm = nn.LayerNorm(num_hidden * 3)
        self.self_attentions = nn.ModuleList(
            [Attention(config) for _ in range(attention_layers)])
        self.mean_out = nn.Linear(num_hidden, output_dim)
        self.cov_out = nn.Linear(num_hidden, output_dim)
        self.cov_m = nn.Softplus()

    def forward(self, r, z, target_x):
        # project vector with dimension 2 --> num_hidden
        target_x = self.target_projection(target_x)

        # concat all vectors (r,z,target_x)
        hidden = torch.cat([torch.cat([r, z], dim=-1), target_x], dim=-1)

        # mlp layers
        for linear in self.linears:
            hidden = torch.relu(linear(hidden))
        hidden = self.hidden_projection(hidden)

        # self attention layer
        for attention in self.self_attentions:
            hidden, _ = attention(hidden, hidden, hidden)

        # get mu and sigma
        mean = self.mean_out(hidden)
        cov = self.cov_m(self.cov_out(hidden))
        return mean, cov


class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism (dot attention)
    """

    def __init__(self, num_hidden_k, dropout):
        """
        :param num_hidden_k: dimension of hidden 
        """
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=dropout)

    def forward(self, key, value, query):
        # Get attention score
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)

        attn = torch.softmax(attn, dim=-1)

        # Dropout
        attn = self.attn_dropout(attn)

        # Get Context Vector
        result = torch.bmm(attn, value)

        return result, attn


class Attention(nn.Module):
    """
    Attention Network
    """

    def __init__(self, config):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads 
        """
        super(Attention, self).__init__()

        hidden_dim = config['hidden_dim']
        num_heads = config['num_heads']
        dropout = config['dropout_rate']
        self.num_hidden = hidden_dim
        self.num_hidden_per_attn = hidden_dim // num_heads
        self.h = num_heads

        self.key = Linear(hidden_dim, hidden_dim, bias=False)
        self.value = Linear(hidden_dim, hidden_dim, bias=False)
        self.query = Linear(hidden_dim, hidden_dim, bias=False)
        self.multihead = MultiheadAttention(self.num_hidden_per_attn, dropout)

        self.residual_dropout = nn.Dropout(p=dropout)

        self.final_linear = Linear(hidden_dim * 2, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, key, value, query):

        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        residual = query

        # Make multihead
        key = self.key(key).view(batch_size, seq_k,self.h, self.num_hidden_per_attn)
        value = self.value(value).view(batch_size, seq_k,self.h, self.num_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q,self.h, self.num_hidden_per_attn)

        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)

        # Get context vector
        result, attns = self.multihead(key, value, query)

        # Concatenate all multihead context vector
        result = result.view(self.h, batch_size, seq_q,
                             self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(
            batch_size, seq_q, -1)

        # Concatenate context vector with input (most important)
        result = torch.cat([residual, result], dim=-1)

        # Final linear
        result = self.final_linear(result)

        # Residual dropout & connection
        result = self.residual_dropout(result)
        result = result + residual

        # Layer normalization
        result = self.layer_norm(result)

        return result, attns


class MLP_Z1Z2_Encoder(nn.Module):

    def __init__(self, config):
        dim = config['hidden_dim']
        in_dim = dim * 2  # hidden, l_r/l_z
        hidden_dim = dim
        out_dim = dim
        hidden_layers = config['hidden_layers']

        nn.Module.__init__(self)
        layers = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
        for _ in range(hidden_layers - 1):
            # layers.append(nn.LayerNorm(hidden_dim))
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, l_z, l_r):
        output = self.model(torch.cat([l_z, l_r], dim=-1))
        mean = self.mean_out(output)

        return mean


class MF_ATTN_Model(nn.Module):
    """
    Latent Model (Attentive Neural Process)
    """

    def __init__(self, config):
        super(MF_ATTN_Model, self).__init__()

        self.l1_latent_encoder = LatentEncoder(config, level=1)
        self.l2_latent_encoder = LatentEncoder(config, level=2)
        self.l1_deterministic_encoder = DeterministicEncoder(config, level=1)
        self.l2_deterministic_encoder = DeterministicEncoder(config, level=2)
        self.l1_decoder = Decoder(config, level=1)
        self.l2_decoder = Decoder(config, level=2)

    def forward(self, l1_x_context, l1_y_context, l1_x_target, l2_x_context, l2_y_context, l2_x_target,
                l1_y_target=None, l2_y_target=None):
        """ l1_r and l1_z --> l1_l2_latent """

        l1_z_mu_c, l1_z_cov_c, l1_prior_z = self.l1_latent_encoder(
            l1_x_context, l1_y_context)
        l2_z_mu_c, l2_z_cov_c, l2_prior_z = self.l2_latent_encoder(
            l2_x_context, l2_y_context, l1_z_mu_c)

        if l1_y_target is not None:
            l1_z_mu_all, l1_z_cov_all, l1_posterior_z = self.l1_latent_encoder(
                l1_x_target, l1_y_target)
            l1_z = l1_posterior_z
        else:
            l1_z = l1_prior_z

        if l2_y_target is not None:
            l2_z_mu_all, l2_z_cov_all, l2_posterior_z = self.l2_latent_encoder(
                l2_x_target, l2_y_target, l1_z_mu_all)
            l2_z = l2_posterior_z
        else:
            l2_z = l2_prior_z

        l1_r = self.l1_deterministic_encoder(
            l1_x_context, l1_y_context, l1_x_target)  # [B, T_target, H]
        l2_r = self.l2_deterministic_encoder(
            l2_x_context, l2_y_context, l2_x_target, l1_r)

        l1_z = l1_z_mu_c.unsqueeze(1).repeat(1, l1_x_target.size(1), 1)
        l2_z = l2_z_mu_c.unsqueeze(1).repeat(1, l2_x_target.size(1), 1)
        l1_output_mu, l1_output_cov = self.l1_decoder(l1_r, l1_z, l1_x_target)
        l2_output_mu, l2_output_cov = self.l2_decoder(l2_r, l2_z, l2_x_target)

        if l2_y_target is not None:
            return l2_output_mu, l2_output_cov, l1_output_mu, l1_output_cov, \
                l2_z_mu_c, l2_z_cov_c, l2_z_mu_all, l2_z_cov_all, \
                l1_z_mu_c, l1_z_cov_c, l1_z_mu_all, l1_z_cov_all
        else:
            return l2_output_mu, l2_output_cov

