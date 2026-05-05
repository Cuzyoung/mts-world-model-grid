"""
WorldTST-Causal: PatchTST with Causal Attention World Model Encoder

Paper B: World Model approach to time series forecasting using causal dynamics.
Core idea: Replace PatchTST's bidirectional self-attention with:
  1. Causal Self-Attention (unidirectional: patch_i only sees patches 1..i)
  2. Cross-Attention to original patches (observation correction)
  3. FFN (unchanged from PatchTST)

World Model narrative:
  - Patch Embedding = observation encoding (raw time series → structured tokens)
  - Causal Self-Attention = parallel dynamics transition model
    (each patch position evolves based on all prior states — autoregressive dynamics)
  - Cross-Attention to patches = posterior update (correct state using full observations)
  - Flatten + Linear head = prediction decoder (decode future from final representation)

Why Causal Attention is "dynamics evolution":
  - Bidirectional attention: each patch sees past AND future — no temporal causality
  - Causal attention: patch_i's representation is built ONLY from patch_1..i
    → this is equivalent to parallel dynamics: s_t = f(s_1, ..., s_t)
  - Information flows strictly forward in time (like GRU, but parallelizable)

Architecture:
  Input → PatchEmbedding → [CausalEncoderLayer × e_layers] → BatchNorm → FlattenHead → Output

  Each CausalEncoderLayer:
    x_orig = x  (save for cross-attention)
    h = CausalSelfAttention(x)  (causal dynamics: each token only attends to past)
    h = CrossAttention(query=h, key/value=x_orig)  (observation correction)
    h = FFN(h)  (same as PatchTST)
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from layers.Embed import PatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class CausalSelfAttention(nn.Module):
    """
    Causal (unidirectional) Self-Attention.

    patch_i can only attend to patches 1..i (not future patches).
    This enforces temporal causality — information flows strictly forward.

    World Model analogy: parallel dynamics transition function.
    Each state s_t is computed from all prior observations o_1..o_t.

    Implementation: standard multi-head attention with a lower-triangular causal mask.
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Causal mask will be registered as buffer on first forward call
        self._causal_mask = None

    def _get_causal_mask(self, L, device):
        """Lower-triangular causal mask: position i can attend to positions 0..i"""
        if self._causal_mask is None or self._causal_mask.shape[-1] < L:
            # (1, 1, L, L) — broadcastable over batch and heads
            mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
            self._causal_mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        return self._causal_mask[:, :, :L, :L]

    def forward(self, x):
        """
        Args:
            x: (B, L, d_model) — patch sequence
        Returns:
            out: (B, L, d_model)
        """
        B, L, D = x.shape
        H = self.n_heads
        head_dim = self.head_dim

        Q = self.q_proj(x).view(B, L, H, head_dim).transpose(1, 2)  # (B, H, L, hd)
        K = self.k_proj(x).view(B, L, H, head_dim).transpose(1, 2)  # (B, H, L, hd)
        V = self.v_proj(x).view(B, L, H, head_dim).transpose(1, 2)  # (B, H, L, hd)

        # Scaled dot-product attention with causal mask
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)

        # Apply causal mask: set future positions to -inf
        causal_mask = self._get_causal_mask(L, x.device)
        attn = attn.masked_fill(causal_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, L, hd)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        return out


class PatchCrossAttention(nn.Module):
    """
    Cross-Attention: causal states attend to original patch embeddings.

    World Model analogy: posterior update — correct the dynamics-evolved state
    using the full observation sequence.

    Note: cross-attention is NOT causal (states can attend to all patches).
    This is by design: the "observation" is the full input window,
    and the posterior should incorporate all available observations.

    query = causal dynamics output (evolved states)
    key/value = original patch embeddings (observations)
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        """
        Args:
            query: (B, L, D) — causal dynamics states
            key_value: (B, L, D) — original patch embeddings
        Returns:
            out: (B, L, D)
        """
        B, L, D = query.shape
        _, S, _ = key_value.shape
        H = self.n_heads
        head_dim = self.head_dim

        Q = self.q_proj(query).view(B, L, H, head_dim).transpose(1, 2)      # (B, H, L, hd)
        K = self.k_proj(key_value).view(B, S, H, head_dim).transpose(1, 2)  # (B, H, S, hd)
        V = self.v_proj(key_value).view(B, S, H, head_dim).transpose(1, 2)  # (B, H, S, hd)

        # Standard attention (no causal mask — cross-attention sees all patches)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, L, hd)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        return out


class CausalEncoderLayer(nn.Module):
    """
    Replaces PatchTST's EncoderLayer (bidirectional Self-Attention + FFN) with:
      1. Causal Self-Attention (unidirectional dynamics)
      2. Cross-Attention to original patches (observation correction)
      3. FFN (identical to PatchTST: Conv1d sandwich)

    Matches EncoderLayer interface: forward(x) → (x, None)
    """

    def __init__(self, d_model, n_heads=4, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        # 1. Causal Self-Attention (replaces bidirectional Self-Attention)
        self.causal_attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 2. Cross-Attention to original patches (observation correction)
        self.cross_attn = PatchCrossAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # 3. FFN (identical to PatchTST EncoderLayer)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        Args:
            x: (B, L, d_model) — patch sequence
        Returns:
            (x, None) — matches Encoder interface
        """
        # Save original for cross-attention
        x_orig = x

        # 1. Causal Self-Attention: causal dynamics evolution
        causal_out = self.causal_attn(x)  # (B, L, D)
        x = self.norm1(x + self.dropout(causal_out))

        # 2. Cross-Attention: observation correction
        cross_out = self.cross_attn(x, x_orig)  # (B, L, D)
        x = self.norm2(x + self.dropout(cross_out))

        # 3. FFN (identical to PatchTST)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = self.norm3(x + y)

        return x, None  # None for compatibility with Encoder


class FlattenHead(nn.Module):
    """Identical to PatchTST's FlattenHead."""

    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    WorldTST-Causal: PatchTST with Causal Attention World Model Encoder.

    Identical to PatchTST except:
    - Bidirectional Self-Attention → Causal Self-Attention + Cross-Attention
    - Everything else (patching, embedding, head, normalization) unchanged

    This ensures fair comparison: same embedding, same head, same training pipeline.
    Only the "how patches interact" mechanism differs:
    PatchTST: bidirectional (all-to-all) attention
    WorldTST-Causal: causal (past-to-present) attention + observation correction
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # Patching and embedding (identical to PatchTST)
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Causal Encoder (replaces Transformer Encoder)
        from layers.Transformer_EncDec import Encoder
        self.encoder = Encoder(
            [
                CausalEncoderLayer(
                    d_model=configs.d_model,
                    n_heads=configs.n_heads,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
        )

        # Prediction Head (identical to PatchTST)
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer (identical to PatchTST)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Patching + Embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        # enc_out: (B*nvars, patch_num, d_model)

        # Causal Encoder
        enc_out, attns = self.encoder(enc_out)
        # enc_out: (B*nvars, patch_num, d_model)

        # Reshape: (B, nvars, patch_num, d_model) → (B, nvars, d_model, patch_num)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Prediction Head
        dec_out = self.head(enc_out)  # (B, nvars, pred_len)
        dec_out = dec_out.permute(0, 2, 1)  # (B, pred_len, nvars)

        # De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
