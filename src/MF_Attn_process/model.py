import torch as t
import torch.nn as nn
import math


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
        encoder_input = t.cat([x, y], dim=-1)

        # project vector with dimension 132+136 --> hidden_dim
        encoder_input = self.input_projection(encoder_input)
        encoder_input = self.layer_norm(encoder_input)

        # self attention layer
        for attention in self.self_attentions:
            encoder_input, _ = attention(
                encoder_input, encoder_input, encoder_input)

        # mean
        hidden = encoder_input.mean(dim=1)
        hidden = t.relu(self.penultimate_layer(hidden))

        # z_mu combine with hidden if level==2
        if l_z is not None:
            hidden = self.l1z_l2z_encoder(hidden, l_z)
            print(hidden.shape)

        # get mu and sigma
        mu = self.mu(hidden)
        log_sigma = self.log_sigma(hidden)

        # reparameterization trick
        std = t.exp(0.5 * log_sigma)
        eps = t.randn_like(std)
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
        encoder_input = t.cat([context_x, context_y], dim=-1)

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
        hidden = t.cat([t.cat([r, z], dim=-1), target_x], dim=-1)

        # mlp layers
        for linear in self.linears:
            hidden = t.relu(linear(hidden))
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
        attn = t.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)

        attn = t.softmax(attn, dim=-1)

        # Dropout
        attn = self.attn_dropout(attn)

        # Get Context Vector
        result = t.bmm(attn, value)

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
        dropout = config['dropout']
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
        key = self.key(key).view(batch_size, seq_k,
                                 self.h, self.num_hidden_per_attn)
        value = self.value(value).view(batch_size, seq_k,
                                       self.h, self.num_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q,
                                       self.h, self.num_hidden_per_attn)

        key = key.permute(2, 0, 1, 3).contiguous().view(-1,
                                                        seq_k, self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous(
        ).view(-1, seq_k, self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous(
        ).view(-1, seq_q, self.num_hidden_per_attn)

        # Get context vector
        result, attns = self.multihead(key, value, query)

        # Concatenate all multihead context vector
        result = result.view(self.h, batch_size, seq_q,
                             self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(
            batch_size, seq_q, -1)

        # Concatenate context vector with input (most important)
        result = t.cat([residual, result], dim=-1)

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
        output = self.model(t.cat([l_z, l_r], dim=-1))
        mean = self.mean_out(output)

        return mean


class Model(nn.Module):
    """
    Latent Model (Attentive Neural Process)
    """

    def __init__(self, config):
        super(Model, self).__init__()

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

    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        kl_div = (t.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / \
            t.exp(prior_var) - 1. + (prior_var - posterior_var)
        kl_div = 0.5 * kl_div.sum()
        return kl_div
