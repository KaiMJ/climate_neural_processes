import torch
import torch.nn as nn
import torch.nn.functional as F


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

import math
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

class Model(nn.Module):

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
