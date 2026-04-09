"""
PatchTST-Dreamer: PatchTST encoder + Multi-Scale Dreamer Dynamics Head

Architecture:
    PatchTST Encoder (pretrained, frozen or finetuned)
    → Latent Projection
    → Multi-Scale Dynamics (Fast GRU + Slow GRU + Gating + Residual + LayerNorm)
    → Cross-Attention to Encoder Features
    → Deep Patch Decoder (3 layers + residual)
    → Predictions
"""

import torch
from torch import nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class MultiScaleDynamics(nn.Module):
    """
    Multi-time-scale latent dynamics with learnable step embeddings.
    Key fix: GRU receives step embeddings as input (not z itself),
    which lets the GRU distinguish different rollout steps.
    Fast GRU: updates every step (short-term patterns).
    Slow GRU: updates every K steps (long-term trends).
    Learned gating fuses fast and slow, with residual bypass.
    """

    def __init__(self, d_latent, num_steps, slow_interval=2, dropout=0.1):
        super().__init__()
        self.slow_interval = slow_interval

        # Learnable step embeddings — the "action" signal for each rollout step
        self.step_embeds = nn.Parameter(torch.randn(num_steps, d_latent) * 0.02)

        self.fast_gru = nn.GRUCell(d_latent, d_latent)
        self.slow_gru = nn.GRUCell(d_latent, d_latent)

        self.gate = nn.Sequential(
            nn.Linear(d_latent * 2, d_latent),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(d_latent)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_init, num_steps):
        """
        Args:
            z_init: (B, d_latent) initial latent state
            num_steps: int, number of rollout steps
        Returns:
            z_seq: (B, num_steps, d_latent)
        """
        z_fast = z_init
        z_slow = z_init
        z_seq = []

        for t in range(num_steps):
            # Step embedding as GRU input (like "action" in Dreamer)
            step_emb = self.step_embeds[t].unsqueeze(0).expand(z_fast.shape[0], -1)

            z_fast_new = self.fast_gru(step_emb, z_fast)

            if t % self.slow_interval == 0:
                z_slow = self.slow_gru(step_emb, z_slow)

            gate = self.gate(torch.cat([z_fast_new, z_slow], dim=-1))
            z_fused = gate * z_fast_new + (1 - gate) * z_slow

            # Residual connection + LayerNorm
            z_fused = self.layer_norm(z_fused + z_fast)
            z_fused = self.dropout(z_fused)

            z_seq.append(z_fused)
            z_fast = z_fused

        return torch.stack(z_seq, dim=1)  # (B, num_steps, d_latent)


class CrossAttention(nn.Module):
    """Cross-attention: dynamics states attend to encoder patch features."""

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

    def forward(self, z_seq, enc_features):
        """
        Args:
            z_seq: (B, num_steps, d_latent) — dynamics states
            enc_features: (B, patch_num, d_model) — encoder patch features
        Returns:
            out: (B, num_steps, d_latent) — attended + residual
        """
        B, S, D = z_seq.shape
        _, P, _ = enc_features.shape
        H = self.n_heads
        head_dim = self.head_dim

        Q = self.q_proj(z_seq).view(B, S, H, head_dim).transpose(1, 2)
        K = self.k_proj(enc_features).view(B, P, H, head_dim).transpose(1, 2)
        V = self.v_proj(enc_features).view(B, P, H, head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)

        # Residual + LayerNorm
        out = self.layer_norm(out + z_seq)
        return out


class DreamerHead(nn.Module):
    """
    Dreamer-style prediction head with bounded dynamics steps.
    Key design: cap GRU rollout at max_steps (default 8) to prevent
    error accumulation for long prediction horizons.
    Each step's decoder output size adapts to cover pred_len/num_steps.
    """

    MAX_DYNAMICS_STEPS = 8  # Cap rollout to prevent error accumulation

    def __init__(self, n_vars, d_model, patch_num, pred_len, patch_len=16,
                 d_latent=128, slow_interval=2, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.pred_len = pred_len
        self.d_latent = d_latent
        self.d_model = d_model
        self.patch_num = patch_num

        # Bounded number of dynamics steps
        raw_steps = max(1, pred_len // patch_len)
        self.num_pred_steps = min(raw_steps, self.MAX_DYNAMICS_STEPS)
        # Each step decodes this many time points
        self.step_output_len = (pred_len + self.num_pred_steps - 1) // self.num_pred_steps

        # Latent projection: encoder patches → initial latent state
        self.to_latent = nn.Sequential(
            nn.Flatten(start_dim=-2),  # (B*nvars, d_model * patch_num)
            nn.Linear(d_model * patch_num, d_latent * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_latent * 2, d_latent),
            nn.LayerNorm(d_latent),
        )

        # Multi-scale dynamics with step embeddings + residual + LayerNorm
        self.dynamics = MultiScaleDynamics(d_latent, self.num_pred_steps, slow_interval, dropout)

        # Cross-attention: dynamics states attend to encoder features
        self.cross_attn = CrossAttention(d_latent, d_model, n_heads=4, dropout=dropout)

        # Deep decoder with residual connections
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
        # Decoder outputs variable-length chunks (adapts to pred_len)
        self.decoder_out = nn.Linear(d_latent, self.step_output_len)

    def forward(self, enc_out):
        """
        Args:
            enc_out: (B, nvars, d_model, patch_num) — PatchTST encoder output
        Returns:
            pred: (B, nvars, pred_len)
        """
        B, nvars, d_model, patch_num = enc_out.shape

        # Reshape to per-variable: (B * nvars, d_model, patch_num)
        x = enc_out.reshape(B * nvars, d_model, patch_num)

        # Save encoder features for cross-attention: (B * nvars, patch_num, d_model)
        enc_features = x.permute(0, 2, 1)

        # Project to latent: (B * nvars, d_latent)
        z_init = self.to_latent(x)

        # Dynamics rollout (bounded steps): (B * nvars, num_pred_steps, d_latent)
        z_seq = self.dynamics(z_init, self.num_pred_steps)

        # Cross-attention to encoder: (B * nvars, num_pred_steps, d_latent)
        z_seq = self.cross_attn(z_seq, enc_features)

        # Deep decoder with residual connections
        h = z_seq
        for layer, norm in zip(self.decoder, self.decoder_norms):
            h = norm(layer(h) + h)

        # Each step → step_output_len points: (B*nvars, num_pred_steps, step_output_len)
        patches = self.decoder_out(h)

        # Concatenate and trim to pred_len
        pred = patches.reshape(B * nvars, -1)[:, :self.pred_len]

        # Reshape back: (B, nvars, pred_len)
        pred = pred.reshape(B, nvars, self.pred_len)

        return pred


# --- Ablation heads ---

class SingleScaleHead(DreamerHead):
    """Ablation: single GRU (no multi-scale), but keeps cross-attention, deep decoder, and bounded steps."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_latent = self.d_latent
        # Replace multi-scale with single GRU + step embeddings (same bounded steps)
        self.dynamics = SingleScaleDynamics(d_latent, self.num_pred_steps, kwargs.get('dropout', 0.1))


class SingleScaleDynamics(nn.Module):
    def __init__(self, d_latent, num_steps, dropout=0.1):
        super().__init__()
        self.step_embeds = nn.Parameter(torch.randn(num_steps, d_latent) * 0.02)
        self.gru = nn.GRUCell(d_latent, d_latent)
        self.layer_norm = nn.LayerNorm(d_latent)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_init, num_steps):
        z = z_init
        z_seq = []
        for t in range(num_steps):
            step_emb = self.step_embeds[t].unsqueeze(0).expand(z.shape[0], -1)
            z_new = self.gru(step_emb, z)
            z = self.layer_norm(z_new + z)  # residual + norm
            z = self.dropout(z)
            z_seq.append(z)
        return torch.stack(z_seq, dim=1)


class FlattenHead(nn.Module):
    """Original PatchTST head for comparison."""

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


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class Model(nn.Module):
    """
    PatchTST-Dreamer: PatchTST encoder + Dreamer dynamics head.

    The encoder is identical to PatchTST and can load pretrained weights.
    The prediction head is replaced with a multi-scale latent dynamics model.
    """

    # Supported head variants for ablation
    HEAD_VARIANTS = ['dreamer', 'single_scale', 'flatten']

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = patch_len
        padding = stride

        # --- PatchTST Encoder (identical, can load pretrained weights) ---
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
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
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        patch_num = int((configs.seq_len - patch_len) / stride + 2)

        # Dreamer head config
        d_latent = getattr(configs, 'd_latent', 128)
        slow_interval = getattr(configs, 'slow_interval', 2)
        head_variant = getattr(configs, 'head_variant', 'dreamer')

        if head_variant == 'dreamer':
            self.head = DreamerHead(
                n_vars=configs.enc_in,
                d_model=configs.d_model,
                patch_num=patch_num,
                pred_len=configs.pred_len,
                patch_len=patch_len,
                d_latent=d_latent,
                slow_interval=slow_interval,
                dropout=configs.dropout,
            )
        elif head_variant == 'single_scale':
            self.head = SingleScaleHead(
                n_vars=configs.enc_in,
                d_model=configs.d_model,
                patch_num=patch_num,
                pred_len=configs.pred_len,
                patch_len=patch_len,
                d_latent=d_latent,
                slow_interval=slow_interval,
                dropout=configs.dropout,
            )
        elif head_variant == 'flatten':
            # Original PatchTST head as baseline
            self.head = FlattenHead(
                configs.enc_in, self.head_nf, configs.pred_len,
                head_dropout=configs.dropout)
        else:
            raise ValueError(f"Unknown head_variant: {head_variant}")

        self.head_variant = head_variant

    def load_patchtst_encoder(self, ckpt_path):
        """Load pretrained PatchTST encoder weights (patch_embedding + encoder)."""
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith('patch_embedding.') or k.startswith('encoder.'):
                encoder_state[k] = v
        missing, unexpected = self.load_state_dict(encoder_state, strict=False)
        print(f"Loaded PatchTST encoder: {len(encoder_state)} params loaded, "
              f"{len(missing)} missing (head), {len(unexpected)} unexpected")

    def freeze_encoder(self):
        """Freeze encoder weights, only train the head."""
        for param in self.patch_embedding.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Instance Normalization (same as PatchTST)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Patching + Embedding
        x_enc = x_enc.permute(0, 2, 1)  # (B, nvars, seq_len)
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        enc_out, attns = self.encoder(enc_out)
        # (B*nvars, patch_num, d_model) → (B, nvars, patch_num, d_model)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # (B, nvars, d_model, patch_num)
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Prediction Head
        dec_out = self.head(enc_out)  # (B, nvars, pred_len)
        dec_out = dec_out.permute(0, 2, 1)  # (B, pred_len, nvars)

        # De-Normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
