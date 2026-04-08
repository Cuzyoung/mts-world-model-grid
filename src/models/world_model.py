"""
MTS-WM: Multi-Time-Scale World Model for Time Series Forecasting

Architecture:
    Input -> PatchEmbedding -> TransformerEncoder -> LatentProjection
          -> MultiScaleDynamics (FastGRU + SlowGRU + Fusion)
          -> Decoder -> Predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """Split time series into patches and embed them."""

    def __init__(self, patch_size, d_input, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * d_input, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pos_embed = None

    def forward(self, x):
        B, L, D = x.shape
        P = self.patch_size
        num_patches = L // P
        x = x[:, :num_patches * P, :].reshape(B, num_patches, P * D)
        x = self.norm(self.proj(x))

        if self.pos_embed is None or self.pos_embed.size(1) != num_patches:
            self.pos_embed = self._sinusoidal_pe(num_patches, x.size(-1)).to(x.device)
        x = x + self.pos_embed[:num_patches].unsqueeze(0)
        return x

    @staticmethod
    def _sinusoidal_pe(length, d_model):
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2 + d_model % 2])
        return pe


class MultiScaleDynamics(nn.Module):
    """
    Multi-time-scale latent dynamics model.

    Fast dynamics: updates every step (captures short-term patterns)
    Slow dynamics: updates every K steps (captures long-term trends)
    Fusion: learned gating to combine fast and slow predictions
    """

    def __init__(self, d_latent, slow_interval=4):
        super().__init__()
        self.d_latent = d_latent
        self.slow_interval = slow_interval

        self.fast_gru = nn.GRUCell(d_latent, d_latent)
        self.slow_gru = nn.GRUCell(d_latent, d_latent)

        self.gate = nn.Sequential(
            nn.Linear(d_latent * 2, d_latent),
            nn.Sigmoid(),
        )

        self.prior_mean = nn.Linear(d_latent, d_latent)
        self.prior_logvar = nn.Linear(d_latent, d_latent)

    def forward(self, z_init, num_steps):
        """
        Roll out latent dynamics for num_steps.

        Returns:
            z_seq: (B, num_steps, d_latent) predicted latent trajectory
            kl_loss: KL regularization loss
        """
        z_fast = z_init
        z_slow = z_init
        z_seq = []
        kl_loss = 0.0

        for t in range(num_steps):
            z_fast = self.fast_gru(z_fast, z_fast)

            if t % self.slow_interval == 0:
                z_slow = self.slow_gru(z_slow, z_slow)

            gate = self.gate(torch.cat([z_fast, z_slow], dim=-1))
            z_fused = gate * z_fast + (1 - gate) * z_slow

            prior_mu = self.prior_mean(z_fused)
            prior_logvar = self.prior_logvar(z_fused)
            kl = -0.5 * torch.mean(1 + prior_logvar - prior_mu.pow(2) - prior_logvar.exp())
            kl_loss = kl_loss + kl

            z_seq.append(z_fused)
            z_fast = z_fused

        z_seq = torch.stack(z_seq, dim=1)
        kl_loss = kl_loss / num_steps
        return z_seq, kl_loss


class MTSWorldModel(nn.Module):
    """
    Multi-Time-Scale World Model for time series forecasting.

    Combines:
    - Patch-based temporal encoding (inspired by PatchTST)
    - Multi-scale latent dynamics (inspired by MTS3/Dreamer)
    - Probabilistic latent space with KL regularization
    """

    def __init__(self, config):
        super().__init__()
        self.seq_len = config["seq_len"]
        self.pred_len = config["pred_len"]
        self.d_input = config["d_input"]
        d_model = config.get("d_model", 512)
        d_latent = config.get("d_latent", 256)
        n_heads = config.get("n_heads", 8)
        n_layers = config.get("n_layers", 4)
        patch_size = config.get("patch_size", 16)
        slow_interval = config.get("slow_interval", 4)
        dropout = config.get("dropout", 0.2)
        self.d_latent = d_latent
        self.patch_size = patch_size

        # 1. Patch embedding
        self.patch_embed = PatchEmbedding(patch_size, self.d_input, d_model)

        # 2. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.enc_dropout = nn.Dropout(dropout)

        # 3. Latent projection — use pooled encoder output instead of just last patch
        self.to_latent = nn.Sequential(
            nn.Linear(d_model, d_latent),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_latent, d_latent),
        )

        # 4. Multi-scale dynamics
        num_pred_patches = max(1, self.pred_len // patch_size)
        self.dynamics = MultiScaleDynamics(d_latent, slow_interval=slow_interval)
        self.num_pred_steps = num_pred_patches

        # 5. Decoder — stronger decoder with residual
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, patch_size * self.d_input),
        )

        # 6. Direct prediction head (residual shortcut for better gradient flow)
        self.direct_pred = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        """
        Args:
            x: (B, seq_len, d_input) historical observations

        Returns:
            pred: (B, pred_len, d_input) future predictions
            kl_loss: KL divergence loss for latent regularization
        """
        B = x.size(0)

        # Direct prediction path (residual)
        direct = self.direct_pred(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, pred_len, d_input)

        # World model path
        patches = self.patch_embed(x)
        h = self.enc_dropout(self.encoder(patches))

        # Pool all patches (mean pooling) for richer initial state
        z_init = self.to_latent(h.mean(dim=1))

        # Roll out dynamics
        z_seq, kl_loss = self.dynamics(z_init, self.num_pred_steps)

        # Decode
        decoded = self.decoder(z_seq)
        decoded = decoded.reshape(B, -1, self.d_input)

        # Trim or pad to pred_len
        if decoded.size(1) >= self.pred_len:
            wm_pred = decoded[:, :self.pred_len, :]
        else:
            pad = torch.zeros(B, self.pred_len - decoded.size(1), self.d_input, device=x.device)
            wm_pred = torch.cat([decoded, pad], dim=1)

        # Combine: world model prediction + direct residual
        pred = wm_pred + direct

        return pred, kl_loss


# --- Ablation variants ---

class SingleScaleWorldModel(MTSWorldModel):
    """Ablation: single-scale dynamics (standard Dreamer-style)."""

    def __init__(self, config):
        super().__init__(config)
        d_latent = config.get("d_latent", 64)
        self.dynamics = SingleScaleDynamics(d_latent)


class SingleScaleDynamics(nn.Module):
    def __init__(self, d_latent):
        super().__init__()
        self.gru = nn.GRUCell(d_latent, d_latent)

    def forward(self, z_init, num_steps):
        z = z_init
        z_seq = []
        for _ in range(num_steps):
            z = self.gru(z, z)
            z_seq.append(z)
        z_seq = torch.stack(z_seq, dim=1)
        return z_seq, torch.tensor(0.0, device=z_init.device)


class NoSlowDynamicsModel(MTSWorldModel):
    """Ablation: only fast dynamics, no slow."""

    def __init__(self, config):
        super().__init__(config)
        d_latent = config.get("d_latent", 64)
        self.dynamics = FastOnlyDynamics(d_latent)


class FastOnlyDynamics(nn.Module):
    def __init__(self, d_latent):
        super().__init__()
        self.fast_gru = nn.GRUCell(d_latent, d_latent)

    def forward(self, z_init, num_steps):
        z = z_init
        z_seq = []
        for _ in range(num_steps):
            z = self.fast_gru(z, z)
            z_seq.append(z)
        z_seq = torch.stack(z_seq, dim=1)
        return z_seq, torch.tensor(0.0, device=z_init.device)


class NoFastDynamicsModel(MTSWorldModel):
    """Ablation: only slow dynamics, no fast."""

    def __init__(self, config):
        super().__init__(config)
        d_latent = config.get("d_latent", 64)
        slow_interval = config.get("slow_interval", 4)
        self.dynamics = SlowOnlyDynamics(d_latent, slow_interval)


class SlowOnlyDynamics(nn.Module):
    def __init__(self, d_latent, slow_interval=4):
        super().__init__()
        self.slow_gru = nn.GRUCell(d_latent, d_latent)
        self.slow_interval = slow_interval

    def forward(self, z_init, num_steps):
        z = z_init
        z_seq = []
        for t in range(num_steps):
            if t % self.slow_interval == 0:
                z = self.slow_gru(z, z)
            z_seq.append(z)
        z_seq = torch.stack(z_seq, dim=1)
        return z_seq, torch.tensor(0.0, device=z_init.device)


class NoDirect(MTSWorldModel):
    """Ablation: remove direct prediction shortcut."""

    def forward(self, x):
        B = x.size(0)
        patches = self.patch_embed(x)
        h = self.encoder(patches)
        z_init = self.to_latent(h.mean(dim=1))
        z_seq, kl_loss = self.dynamics(z_init, self.num_pred_steps)
        decoded = self.decoder(z_seq)
        decoded = decoded.reshape(B, -1, self.d_input)
        if decoded.size(1) >= self.pred_len:
            pred = decoded[:, :self.pred_len, :]
        else:
            pad = torch.zeros(B, self.pred_len - decoded.size(1), self.d_input, device=x.device)
            pred = torch.cat([decoded, pad], dim=1)
        return pred, kl_loss
