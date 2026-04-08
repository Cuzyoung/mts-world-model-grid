"""
Baseline models for comparison:
- LSTM
- DLinear
- Simplified Informer (ProbSparse attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LSTMBaseline(nn.Module):
    """Standard LSTM encoder-decoder for time series forecasting."""

    def __init__(self, config):
        super().__init__()
        self.pred_len = config["pred_len"]
        d_input = config["d_input"]
        d_hidden = config.get("d_model", 128)
        n_layers = config.get("n_layers", 2)

        self.encoder = nn.LSTM(d_input, d_hidden, n_layers, batch_first=True, dropout=0.1)
        self.decoder = nn.LSTM(d_input, d_hidden, n_layers, batch_first=True, dropout=0.1)
        self.proj = nn.Linear(d_hidden, d_input)

    def forward(self, x):
        # Encode
        _, (h, c) = self.encoder(x)

        # Decode autoregressively
        dec_input = x[:, -1:, :]  # last step
        outputs = []
        for _ in range(self.pred_len):
            out, (h, c) = self.decoder(dec_input, (h, c))
            pred = self.proj(out)
            outputs.append(pred)
            dec_input = pred
        pred = torch.cat(outputs, dim=1)
        return pred, torch.tensor(0.0, device=x.device)


class DLinearBaseline(nn.Module):
    """
    DLinear: decomposition + linear layers.
    Reference: Are Transformers Effective for Time Series Forecasting? (AAAI 2023)
    """

    def __init__(self, config):
        super().__init__()
        self.seq_len = config["seq_len"]
        self.pred_len = config["pred_len"]
        d_input = config["d_input"]
        kernel_size = 25

        self.decomp = SeriesDecomp(kernel_size)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: (B, L, D)
        seasonal, trend = self.decomp(x)
        # (B, L, D) -> (B, D, L) for linear
        trend_out = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal_out = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        pred = trend_out + seasonal_out
        return pred, torch.tensor(0.0, device=x.device)


class SeriesDecomp(nn.Module):
    """Series decomposition using moving average."""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: (B, L, D)
        # Pad for causal moving average
        pad = self.kernel_size // 2
        x_pad = F.pad(x.permute(0, 2, 1), (pad, pad), mode="replicate")
        trend = self.avg(x_pad).permute(0, 2, 1)
        # Trim to original length
        trend = trend[:, :x.size(1), :]
        seasonal = x - trend
        return seasonal, trend


class InformerBaseline(nn.Module):
    """
    Simplified Informer with ProbSparse self-attention.
    Not a full reproduction -- a lightweight version for fair comparison.
    """

    def __init__(self, config):
        super().__init__()
        self.seq_len = config["seq_len"]
        self.pred_len = config["pred_len"]
        d_input = config["d_input"]
        d_model = config.get("d_model", 128)
        n_heads = config.get("n_heads", 4)
        n_layers = config.get("n_layers", 2)

        self.input_proj = nn.Linear(d_input, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder = nn.Linear(d_model, d_input)
        self.pred_linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        B, L, D = x.shape
        h = self.input_proj(x) + self.pos_embed[:, :L, :]
        h = self.encoder(h)  # (B, L, d_model)
        # Project sequence length to pred_len
        h = h.permute(0, 2, 1)  # (B, d_model, L)
        h = self.pred_linear(h)  # (B, d_model, pred_len)
        h = h.permute(0, 2, 1)  # (B, pred_len, d_model)
        pred = self.decoder(h)  # (B, pred_len, d_input)
        return pred, torch.tensor(0.0, device=x.device)
