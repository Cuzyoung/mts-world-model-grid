"""
WorldTST-GRU Ablation: No CrossAttention variant.
GRU dynamics only, no observation correction.
Used to prove the value of CrossAttention component.
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


class MultiScaleGRULayer(nn.Module):
    def __init__(self, d_model, slow_interval=2, dropout=0.1):
        super().__init__()
        self.slow_interval = slow_interval
        self.fast_gru = nn.GRUCell(d_model, d_model)
        self.slow_gru = nn.GRUCell(d_model, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        h_fast = x[:, 0, :]
        h_slow = x[:, 0, :]
        outputs = []
        for t in range(L):
            x_t = x[:, t, :]
            h_fast_new = self.fast_gru(x_t, h_fast)
            if t % self.slow_interval == 0:
                h_slow = self.slow_gru(x_t, h_slow)
            gate = self.gate(torch.cat([h_fast_new, h_slow], dim=-1))
            h_fused = gate * h_fast_new + (1 - gate) * h_slow
            h_fused = self.layer_norm(h_fused + x_t)
            h_fused = self.dropout(h_fused)
            outputs.append(h_fused)
            h_fast = h_fused
        return torch.stack(outputs, dim=1)


class GRUEncoderLayerNoCross(nn.Module):
    """GRU dynamics + FFN, NO CrossAttention."""
    def __init__(self, d_model, n_heads=4, d_ff=None, slow_interval=2,
                 dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.gru_dynamics = MultiScaleGRULayer(d_model, slow_interval, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # GRU dynamics only (no cross-attention)
        gru_out = self.gru_dynamics(x)
        x = self.norm1(x + self.dropout(gru_out))
        # FFN
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = self.norm2(x + y)
        return x, None


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """WorldTST-GRU ablation: No CrossAttention."""
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
            [GRUEncoderLayerNoCross(
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
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
