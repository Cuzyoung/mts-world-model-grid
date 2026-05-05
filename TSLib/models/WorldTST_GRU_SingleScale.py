"""
WorldTST-GRU Ablation: Single-Scale GRU (no slow GRU, no gating).
Used to prove the value of multi-scale dynamics.
"""

import torch
from torch import nn
import torch.nn.functional as F
from layers.Embed import PatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class SingleScaleGRULayer(nn.Module):
    """Single GRU only — no slow path, no gating."""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.gru = nn.GRUCell(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        h = x[:, 0, :]
        outputs = []
        for t in range(L):
            x_t = x[:, t, :]
            h_new = self.gru(x_t, h)
            h = self.layer_norm(h_new + x_t)
            h = self.dropout(h)
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class PatchCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        B, L, D = query.shape
        H, hd = self.n_heads, self.head_dim
        Q = self.q_proj(query).view(B, L, H, hd).transpose(1, 2)
        K = self.k_proj(key_value).view(B, L, H, hd).transpose(1, 2)
        V = self.v_proj(key_value).view(B, L, H, hd).transpose(1, 2)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (hd ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class GRUEncoderLayerSingleScale(nn.Module):
    """Single-Scale GRU + CrossAttention + FFN."""
    def __init__(self, d_model, n_heads=4, d_ff=None, slow_interval=2,
                 dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.gru_dynamics = SingleScaleGRULayer(d_model, dropout)
        self.cross_attn = PatchCrossAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        x_orig = x
        gru_out = self.gru_dynamics(x)
        cross_out = self.cross_attn(gru_out, x_orig)
        x = self.norm1(x + self.dropout(cross_out))
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = self.norm2(x + y)
        return x, None


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
    def forward(self, x):
        return self.dropout(self.linear(self.flatten(x)))


class Model(nn.Module):
    """WorldTST-GRU ablation: Single-Scale (no slow GRU)."""
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride
        slow_interval = getattr(configs, 'slow_interval', 2)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        from layers.Transformer_EncDec import Encoder
        self.encoder = Encoder(
            [GRUEncoderLayerSingleScale(
                d_model=configs.d_model, n_heads=configs.n_heads, d_ff=configs.d_ff,
                slow_interval=slow_interval, dropout=configs.dropout,
                activation=configs.activation,
            ) for _ in range(configs.e_layers)],
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
        )

        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out).permute(0, 2, 1)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]
        return None
