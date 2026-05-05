"""
WorldTST-GRU: PatchTST with World-Model-inspired GRU Dynamics Encoder

Paper A: World Model approach to time series forecasting.
Core idea: Replace PatchTST's bidirectional self-attention with:
  1. Multi-Scale GRU dynamics (sequential state evolution along patch sequence)
  2. Cross-Attention to original patches (observation correction)
  3. FFN (unchanged from PatchTST)

World Model narrative:
  - Patch Embedding = observation encoding (raw time series → structured tokens)
  - GRU rollout along patches = dynamics transition model (state evolves sequentially)
  - Cross-Attention to patches = posterior update (correct state using observations)
  - Flatten + Linear head = prediction decoder (decode future from final representation)

Architecture:
  Input → PatchEmbedding → [GRUEncoderLayer × e_layers] → BatchNorm → FlattenHead → Output

  Each GRUEncoderLayer:
    x_orig = x  (save for cross-attention)
    h = GRU_rollout(x)  (sequential dynamics: fast GRU + slow GRU + gate)
    h = CrossAttention(query=h, key/value=x_orig)  (observation correction)
    h = FFN(h)  (same as PatchTST)
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
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class MultiScaleGRULayer(nn.Module):
    """
    Multi-Scale GRU: processes patch sequence sequentially.
    Fast GRU updates every step, Slow GRU every K steps.
    Learned gating fuses fast and slow paths.

    This replaces Self-Attention — instead of all-to-all attention,
    patches are processed left-to-right with recurrent state transition.

    World Model analogy: state transition function p(s_t | s_{t-1}, o_t)
    """

    def __init__(self, d_model, slow_interval=2, dropout=0.1):
        super().__init__()
        self.slow_interval = slow_interval

        self.fast_gru = nn.GRUCell(d_model, d_model)
        self.slow_gru = nn.GRUCell(d_model, d_model)

        # Learned gate to fuse fast and slow
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, L, d_model) — patch sequence
        Returns:
            out: (B, L, d_model) — GRU hidden states for each patch position
        """
        B, L, D = x.shape

        # Initialize hidden states from first patch
        h_fast = x[:, 0, :]  # (B, D)
        h_slow = x[:, 0, :]  # (B, D)

        outputs = []
        for t in range(L):
            x_t = x[:, t, :]  # (B, D) — current patch as input

            # Fast GRU: every step
            h_fast_new = self.fast_gru(x_t, h_fast)

            # Slow GRU: every K steps
            if t % self.slow_interval == 0:
                h_slow = self.slow_gru(x_t, h_slow)

            # Gated fusion of fast and slow
            gate = self.gate(torch.cat([h_fast_new, h_slow], dim=-1))
            h_fused = gate * h_fast_new + (1 - gate) * h_slow

            # Residual + LayerNorm (residual from input, not previous hidden)
            h_fused = self.layer_norm(h_fused + x_t)
            h_fused = self.dropout(h_fused)

            outputs.append(h_fused)
            h_fast = h_fused

        return torch.stack(outputs, dim=1)  # (B, L, D)


class PatchCrossAttention(nn.Module):
    """
    Cross-Attention: GRU hidden states attend to original patch embeddings.

    World Model analogy: posterior update — correct the dynamics state
    using the original observations.

    query = GRU hidden states (dynamics output)
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
            query: (B, L, D) — GRU hidden states
            key_value: (B, L, D) — original patch embeddings
        Returns:
            out: (B, L, D)
        """
        B, L, D = query.shape
        H = self.n_heads
        head_dim = self.head_dim

        Q = self.q_proj(query).view(B, L, H, head_dim).transpose(1, 2)    # (B, H, L, hd)
        K = self.k_proj(key_value).view(B, L, H, head_dim).transpose(1, 2)  # (B, H, L, hd)
        V = self.v_proj(key_value).view(B, L, H, head_dim).transpose(1, 2)  # (B, H, L, hd)

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, L, hd)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        return out


class GRUEncoderLayer(nn.Module):
    """
    Replaces PatchTST's EncoderLayer (Self-Attention + FFN) with:
      1. Multi-Scale GRU dynamics (sequential processing)
      2. Cross-Attention to original patches (observation correction)
      3. FFN (identical to PatchTST: Conv1d sandwich)

    Matches EncoderLayer interface: forward(x) → (x, None)
    """

    def __init__(self, d_model, n_heads=4, d_ff=None, slow_interval=2,
                 dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        # 1. Multi-Scale GRU dynamics (replaces Self-Attention)
        self.gru_dynamics = MultiScaleGRULayer(d_model, slow_interval, dropout)

        # 2. Cross-Attention to original patches (observation correction)
        self.cross_attn = PatchCrossAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 3. FFN (identical to PatchTST EncoderLayer)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        Args:
            x: (B, L, d_model) — patch sequence
        Returns:
            (x, None) — matches Encoder interface (None = no attention weights)
        """
        # Save original for cross-attention key/value
        x_orig = x

        # 1. GRU dynamics: sequential state evolution
        gru_out = self.gru_dynamics(x)  # (B, L, D)

        # 2. Cross-Attention: GRU states attend to original patches
        cross_out = self.cross_attn(gru_out, x_orig)  # (B, L, D)

        # Residual + Norm (residual from input x, like standard transformer)
        x = self.norm1(x + self.dropout(cross_out))

        # 3. FFN (identical to PatchTST)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = self.norm2(x + y)

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
    WorldTST-GRU: PatchTST with GRU-based World Model Encoder.

    Identical to PatchTST except:
    - Self-Attention in each EncoderLayer → Multi-Scale GRU + Cross-Attention
    - Everything else (patching, embedding, head, normalization) unchanged

    This ensures fair comparison: same capacity, same head, same training pipeline.
    Only the "how patches interact" mechanism differs.
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # Slow interval for multi-scale GRU (default 2)
        slow_interval = getattr(configs, 'slow_interval', 2)

        # Patching and embedding (identical to PatchTST)
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # GRU Encoder (replaces Transformer Encoder)
        # We reuse TSLib's Encoder wrapper but with GRUEncoderLayer instead of EncoderLayer
        from layers.Transformer_EncDec import Encoder
        self.encoder = Encoder(
            [
                GRUEncoderLayer(
                    d_model=configs.d_model,
                    n_heads=configs.n_heads,
                    d_ff=configs.d_ff,
                    slow_interval=slow_interval,
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

        # GRU Encoder
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
