"""
PatchTST-Dreamer v2: World Model Dynamics with Observation-Guided Rollout

Key improvements over v1:
  1. Attention Pooling replaces Flatten→MLP (no information bottleneck)
  2. Cross-Attention moved INSIDE GRU loop (every step gets "observation" from encoder)
  3. Gated Linear Skip (linear path provides performance floor)
  4. Multi-Scale GRU preserved (ablation-proven critical)

World Model narrative:
  - Attention Pooling  = "initialize belief from observation"
  - In-loop CrossAttn  = "posterior update with observation at each step"
  - Multi-Scale GRU    = "multi-time-scale dynamics transition"
  - Deep Decoder       = "decode prediction from imagined states"
  - Linear Skip + Gate = "model-free baseline for robustness"
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


# =====================================================================
#  Component 1: Attention Pooling  (replaces Flatten→MLP bottleneck)
# =====================================================================

class AttentionPooling(nn.Module):
    """
    Learnable query attends to encoder patches → initial latent state z_0.
    Eliminates the hard dependency on patch_num (works with any seq_len).
    """

    def __init__(self, d_model, d_latent, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_latent // n_heads

        # Single learnable query token
        self.query = nn.Parameter(torch.randn(1, 1, d_latent) * 0.02)

        self.q_proj = nn.Linear(d_latent, d_latent)
        self.k_proj = nn.Linear(d_model, d_latent)
        self.v_proj = nn.Linear(d_model, d_latent)
        self.out_proj = nn.Linear(d_latent, d_latent)
        self.layer_norm = nn.LayerNorm(d_latent)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_patches):
        """
        Args:
            enc_patches: (B, patch_num, d_model)
        Returns:
            z_0: (B, d_latent)
        """
        B = enc_patches.shape[0]
        H = self.n_heads
        head_dim = self.head_dim

        # Expand query for batch
        q = self.query.expand(B, -1, -1)  # (B, 1, d_latent)

        Q = self.q_proj(q).view(B, 1, H, head_dim).transpose(1, 2)       # (B, H, 1, hd)
        K = self.k_proj(enc_patches).view(B, -1, H, head_dim).transpose(1, 2)  # (B, H, P, hd)
        V = self.v_proj(enc_patches).view(B, -1, H, head_dim).transpose(1, 2)  # (B, H, P, hd)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, 1, hd)
        out = out.transpose(1, 2).contiguous().view(B, 1, -1)  # (B, 1, d_latent)
        out = self.out_proj(out).squeeze(1)  # (B, d_latent)

        # Residual from query + LayerNorm
        z_0 = self.layer_norm(out + self.query.squeeze(0).expand(B, -1))
        return z_0


# =====================================================================
#  Component 2: In-Loop Cross-Attention  (observation at each step)
# =====================================================================

class StepCrossAttention(nn.Module):
    """
    Single-query cross-attention: current latent z_t attends to encoder patches.
    Produces the "observation" signal c_t that feeds into the GRU.
    """

    def __init__(self, d_latent, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_latent // n_heads

        self.q_proj = nn.Linear(d_latent, d_latent)
        self.k_proj = nn.Linear(d_model, d_latent)
        self.v_proj = nn.Linear(d_model, d_latent)
        self.out_proj = nn.Linear(d_latent, d_latent)
        self.layer_norm = nn.LayerNorm(d_latent)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_t, enc_patches):
        """
        Args:
            z_t: (B, d_latent) — current latent state
            enc_patches: (B, patch_num, d_model) — encoder features
        Returns:
            c_t: (B, d_latent) — observation-guided context
        """
        B = z_t.shape[0]
        H = self.n_heads
        head_dim = self.head_dim

        q = z_t.unsqueeze(1)  # (B, 1, d_latent)
        Q = self.q_proj(q).view(B, 1, H, head_dim).transpose(1, 2)
        K = self.k_proj(enc_patches).view(B, -1, H, head_dim).transpose(1, 2)
        V = self.v_proj(enc_patches).view(B, -1, H, head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, -1)  # (B, d_latent)
        out = self.out_proj(out)

        # Residual + LayerNorm
        c_t = self.layer_norm(out + z_t)
        return c_t


# =====================================================================
#  Component 3: Multi-Scale Dynamics v2  (observation-guided GRU)
# =====================================================================

class MultiScaleDynamicsV2(nn.Module):
    """
    Multi-time-scale latent dynamics with observation-guided input.

    Key change from v1: GRU input is cross-attention output c_t (data-dependent),
    NOT learnable step_embed (data-independent).

    Fast GRU: updates every step (short-term patterns).
    Slow GRU: updates every K steps (long-term trends).
    Learned gating fuses fast and slow, with residual bypass.
    """

    def __init__(self, d_latent, d_model, num_steps, slow_interval=2,
                 n_attn_heads=4, dropout=0.1):
        super().__init__()
        self.slow_interval = slow_interval
        self.num_steps = num_steps

        # Per-step cross-attention: z_t → attend to encoder → c_t
        self.step_cross_attn = StepCrossAttention(
            d_latent, d_model, n_heads=n_attn_heads, dropout=dropout)

        # Optional: additive step embeddings for positional signal
        self.step_embeds = nn.Parameter(torch.randn(num_steps, d_latent) * 0.02)

        self.fast_gru = nn.GRUCell(d_latent, d_latent)
        self.slow_gru = nn.GRUCell(d_latent, d_latent)

        self.gate = nn.Sequential(
            nn.Linear(d_latent * 2, d_latent),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(d_latent)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_init, enc_patches, num_steps):
        """
        Args:
            z_init: (B, d_latent) initial latent state from AttentionPooling
            enc_patches: (B, patch_num, d_model) encoder features
            num_steps: int, number of rollout steps
        Returns:
            z_seq: (B, num_steps, d_latent)
        """
        z_fast = z_init
        z_slow = z_init
        z_seq = []

        for t in range(num_steps):
            # === KEY CHANGE: observation-guided input ===
            # Cross-attention: z_t attends to encoder patches → c_t
            c_t = self.step_cross_attn(z_fast, enc_patches)

            # Add positional step embedding
            step_emb = self.step_embeds[t].unsqueeze(0).expand(z_fast.shape[0], -1)
            c_t = c_t + step_emb

            # Fast GRU: updates every step
            z_fast_new = self.fast_gru(c_t, z_fast)

            # Slow GRU: updates every slow_interval steps
            if t % self.slow_interval == 0:
                z_slow = self.slow_gru(c_t, z_slow)

            # Gated fusion of fast and slow
            gate = self.gate(torch.cat([z_fast_new, z_slow], dim=-1))
            z_fused = gate * z_fast_new + (1 - gate) * z_slow

            # Residual connection + LayerNorm
            z_fused = self.layer_norm(z_fused + z_fast)
            z_fused = self.dropout(z_fused)

            z_seq.append(z_fused)
            z_fast = z_fused

        return torch.stack(z_seq, dim=1)  # (B, num_steps, d_latent)


# =====================================================================
#  Component 4: Dreamer Head v2  (world model prediction path)
# =====================================================================

class DreamerHeadV2(nn.Module):
    """
    World model prediction head:
      AttentionPooling → MultiScaleDynamicsV2 → Deep Decoder
    """

    MAX_DYNAMICS_STEPS = 8

    def __init__(self, n_vars, d_model, patch_num, pred_len, patch_len=16,
                 d_latent=256, slow_interval=2, n_attn_heads=4, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.pred_len = pred_len
        self.d_latent = d_latent
        self.d_model = d_model

        # Bounded number of dynamics steps
        raw_steps = max(1, pred_len // patch_len)
        self.num_pred_steps = min(raw_steps, self.MAX_DYNAMICS_STEPS)
        self.step_output_len = (pred_len + self.num_pred_steps - 1) // self.num_pred_steps

        # Attention pooling: encoder patches → z_0
        self.attn_pool = AttentionPooling(d_model, d_latent, n_heads=n_attn_heads, dropout=dropout)

        # Multi-scale dynamics with in-loop cross-attention
        self.dynamics = MultiScaleDynamicsV2(
            d_latent, d_model, self.num_pred_steps,
            slow_interval=slow_interval,
            n_attn_heads=n_attn_heads,
            dropout=dropout,
        )

        # Deep decoder with residual connections (same as v1)
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_latent, d_latent * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_latent * 2, d_latent),
            ),
            nn.Sequential(
                nn.Linear(d_latent, d_latent * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_latent * 2, d_latent),
            ),
        ])
        self.decoder_norms = nn.ModuleList([
            nn.LayerNorm(d_latent),
            nn.LayerNorm(d_latent),
        ])
        self.decoder_out = nn.Linear(d_latent, self.step_output_len)

    def forward(self, enc_out):
        """
        Args:
            enc_out: (B, nvars, d_model, patch_num)
        Returns:
            pred: (B, nvars, pred_len)
        """
        B, nvars, d_model, patch_num = enc_out.shape

        # Reshape to per-variable: (B*nvars, patch_num, d_model)
        enc_patches = enc_out.reshape(B * nvars, d_model, patch_num).permute(0, 2, 1)

        # Attention pooling → z_0: (B*nvars, d_latent)
        z_init = self.attn_pool(enc_patches)

        # Dynamics rollout with observation: (B*nvars, num_steps, d_latent)
        z_seq = self.dynamics(z_init, enc_patches, self.num_pred_steps)

        # Deep decoder with residual
        h = z_seq
        for layer, norm in zip(self.decoder, self.decoder_norms):
            h = norm(layer(h) + h)

        # Each step → step_output_len points
        patches = self.decoder_out(h)  # (B*nvars, num_steps, step_output_len)

        # Concatenate and trim to pred_len
        pred = patches.reshape(B * nvars, -1)[:, :self.pred_len]
        pred = pred.reshape(B, nvars, self.pred_len)

        return pred


# =====================================================================
#  Component 5: Gated Dreamer Head  (linear skip + world model)
# =====================================================================

class FlattenHead(nn.Module):
    """Original PatchTST head (linear path for gated fusion)."""

    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class GatedDreamerHead(nn.Module):
    """
    Gated fusion of Linear Skip (PatchTST baseline) + DreamerHeadV2 (world model).

    The gate learns per-timestep blending:
      ŷ = g ⊙ ŷ_dynamics + (1-g) ⊙ ŷ_linear

    Initialization ensures the model starts as pure PatchTST:
      - decoder_out weights = 0  → dynamics path outputs zero
      - gate bias = -2           → σ(-2) ≈ 0.12, favoring linear
    """

    def __init__(self, n_vars, d_model, patch_num, pred_len, patch_len=16,
                 d_latent=256, slow_interval=2, n_attn_heads=4, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        nf = d_model * patch_num

        # PATH 1: Linear (identical to PatchTST)
        self.linear_head = FlattenHead(n_vars, nf, pred_len, head_dropout=dropout)

        # PATH 2: World Model Dynamics
        self.dreamer_head = DreamerHeadV2(
            n_vars=n_vars,
            d_model=d_model,
            patch_num=patch_num,
            pred_len=pred_len,
            patch_len=patch_len,
            d_latent=d_latent,
            slow_interval=slow_interval,
            n_attn_heads=n_attn_heads,
            dropout=dropout,
        )

        # GATE: per-timestep blending
        self.gate_proj = nn.Linear(pred_len * 2, pred_len)

        # === Critical initialization ===
        self._init_for_linear_start()

    def _init_for_linear_start(self):
        """
        Initialize so that the model starts as pure PatchTST:
        - Dreamer decoder output weights → 0 (dynamics path produces zero)
        - Gate bias → -2 (sigmoid ≈ 0.12, heavily favoring linear)
        """
        # Zero-init the dreamer decoder output
        nn.init.zeros_(self.dreamer_head.decoder_out.weight)
        nn.init.zeros_(self.dreamer_head.decoder_out.bias)

        # Gate bias → -2 (favors linear path)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -2.0)

    def forward(self, enc_out):
        """
        Args:
            enc_out: (B, nvars, d_model, patch_num)
        Returns:
            pred: (B, nvars, pred_len)
        """
        # PATH 1: Linear prediction (= PatchTST)
        y_linear = self.linear_head(enc_out)  # (B, nvars, pred_len)

        # PATH 2: World model dynamics prediction
        y_dynamics = self.dreamer_head(enc_out)  # (B, nvars, pred_len)

        # GATE: per-timestep sigmoid blend
        gate_input = torch.cat([y_linear, y_dynamics], dim=-1)  # (B, nvars, pred_len*2)
        gate = torch.sigmoid(self.gate_proj(gate_input))  # (B, nvars, pred_len)

        # Fused prediction
        pred = gate * y_dynamics + (1 - gate) * y_linear

        return pred


# =====================================================================
#  Component 6: Transpose helper
# =====================================================================

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


# =====================================================================
#  Main Model: PatchTST Encoder + Gated Dreamer Head v2
# =====================================================================

class Model(nn.Module):
    """
    PatchTST-Dreamer v2: PatchTST encoder + Gated World Model Dynamics Head.

    The encoder is identical to PatchTST (channel-independent Transformer).
    The prediction head uses observation-guided multi-scale GRU rollout
    with a gated linear skip for robustness.
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # --- PatchTST Encoder (identical to original) ---
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
        )

        # --- Prediction Head ---
        patch_num = int((configs.seq_len - patch_len) / stride + 2)
        d_latent = getattr(configs, 'd_latent', 256)
        slow_interval = getattr(configs, 'slow_interval', 2)
        n_attn_heads = 4  # cross-attention heads in dreamer head

        self.head = GatedDreamerHead(
            n_vars=configs.enc_in,
            d_model=configs.d_model,
            patch_num=patch_num,
            pred_len=configs.pred_len,
            patch_len=patch_len,
            d_latent=d_latent,
            slow_interval=slow_interval,
            n_attn_heads=n_attn_heads,
            dropout=configs.dropout,
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Instance Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Patching + Embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # (B, nvars, d_model, patch_num)

        # Prediction Head
        dec_out = self.head(enc_out)  # (B, nvars, pred_len)
        dec_out = dec_out.permute(0, 2, 1)  # (B, pred_len, nvars)

        # De-Normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
