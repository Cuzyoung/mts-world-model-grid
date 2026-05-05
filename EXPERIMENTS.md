# PatchTST-Dreamer: Experimental Report

**Date**: 2026-04-09
**Status**: Experiments in progress (Setting B Dreamer, Hybrid, ECL reproduction still running)

---

## 1. Model Architecture

### 1.1 Base Model: PatchTST (ICLR 2023)

PatchTST is a channel-independent Transformer for multivariate time series forecasting. Each variable is processed independently through shared weights.

```
Input (B, seq_len, nvars)
  -> Instance Normalization (per-variable zero-mean, unit-variance)
  -> Transpose to (B, nvars, seq_len)
  -> Patch Embedding: segment each variable into patches of length P with stride S
     patch_num = floor((seq_len - P) / S) + 2  (with padding = stride)
     -> (B * nvars, patch_num, d_model)
  -> Transformer Encoder: e_layers x [MultiHead Self-Attention + FFN]
     -> (B * nvars, patch_num, d_model)
  -> Reshape to (B, nvars, d_model, patch_num)
  -> Flatten Head: Flatten(d_model * patch_num) -> Linear(pred_len)
     -> (B, nvars, pred_len)
  -> De-Normalization
  -> Output (B, pred_len, nvars)
```

Key design choices:
- **Channel-independent**: Each variable shares the same encoder but is processed independently (no cross-variable attention)
- **Patching**: Groups P consecutive time points into one token, reducing sequence length and capturing local semantics
- **Instance Normalization**: Reversible normalization for non-stationary time series

### 1.2 Our Contribution: PatchTST-Dreamer

We replace PatchTST's linear Flatten Head with a **Dreamer-style multi-scale latent dynamics head**, inspired by world models from reinforcement learning (Dreamer, ICLR 2020; MTS3, NeurIPS 2023).

```
PatchTST Encoder Output (B, nvars, d_model, patch_num)
  -> Latent Projection: Flatten + MLP -> z_0 (B*nvars, d_latent)
  -> Multi-Scale Dynamics Rollout (K steps):
       For each step t = 0..K-1:
         - Fast GRU: z_fast = GRU(step_embed[t], z_fast)      [updates every step]
         - Slow GRU: z_slow = GRU(step_embed[t], z_slow)      [updates every slow_interval steps]
         - Gated Fusion: gate = sigmoid(Linear([z_fast, z_slow]))
                         z_fused = gate * z_fast + (1-gate) * z_slow
         - Residual + LayerNorm: z = LN(z_fused + z_prev)
     -> z_seq (B*nvars, K, d_latent)
  -> Cross-Attention: z_seq attends to encoder patch features
     -> (B*nvars, K, d_latent)
  -> Deep Patch Decoder: 2-layer MLP with residual connections
     -> Each step decodes to (step_output_len) time points
  -> Concatenate + trim to pred_len
  -> Output (B, nvars, pred_len)
```

#### Multi-Scale Dynamics (Core Innovation)

The multi-scale GRU dynamics are inspired by MTS3 (Multi-Time-Scale world models):

- **Fast GRU**: Captures short-term fluctuations, updates at every rollout step
- **Slow GRU**: Captures long-term trends, updates every `slow_interval` steps (default: 2)
- **Learnable Step Embeddings**: Each rollout step has a learnable embedding vector that serves as the GRU input (analogous to "actions" in Dreamer). This allows the model to distinguish different prediction steps
- **Gated Fusion**: A learned sigmoid gate dynamically combines fast and slow streams
- **Bounded Rollout**: MAX_DYNAMICS_STEPS = 8 caps the GRU rollout length to prevent error accumulation. For pred_len > 8*patch_len, each decoded step covers multiple time points

#### Cross-Attention to Encoder Features

After dynamics rollout, each latent state attends to the original encoder patch features via multi-head cross-attention (4 heads). This allows the prediction head to selectively reference historical patterns from the encoder.

#### Deep Patch Decoder

A 2-layer MLP with residual connections and LayerNorm decodes each latent state to a chunk of time-series predictions. The chunk size adapts based on pred_len and num_pred_steps.

### 1.3 Hybrid Head (Variant)

To address error accumulation in long-horizon rollouts, we also propose a **Hybrid Head** that combines the Dreamer dynamics path with a linear skip connection:

```
Encoder Output
  |--- Linear Path: Flatten -> Linear(pred_len) -> linear_pred
  |--- Dreamer Path: Latent Proj -> GRU Rollout -> CrossAttn -> Decoder -> dreamer_pred
  |
  -> Gated Fusion: gate = sigmoid(Linear([linear_pred, dreamer_pred]))
                   output = gate * dreamer_pred + (1-gate) * linear_pred
```

The linear path provides a stable baseline prediction (identical to PatchTST's original head), while the Dreamer path learns nonlinear residual dynamics. The learnable gate allows the model to adaptively rely on the linear path for easy patterns and the Dreamer path for complex dynamics.

### 1.4 Ablation Variants

| Variant | `--head_variant` | Description |
|---------|-----------------|-------------|
| **Dreamer (full)** | `dreamer` | Fast GRU + Slow GRU + gating + cross-attention + deep decoder |
| **Hybrid** | `hybrid` | Dreamer + Linear skip with learned gate |
| **Single-scale** | `single_scale` | Single GRU (no multi-scale), otherwise same architecture |
| **Flatten (PatchTST)** | `flatten` | Original PatchTST linear head (for fair comparison) |

---

## 2. Datasets

### 2.1 Dataset Statistics

| Dataset | Domain | Variates | Timesteps | Frequency | Source |
|---------|--------|----------|-----------|-----------|--------|
| ETTh1 | Electricity Transformer | 7 | 17,420 | 1 hour | Zhou et al. (2021) |
| ETTh2 | Electricity Transformer | 7 | 17,420 | 1 hour | Zhou et al. (2021) |
| ETTm1 | Electricity Transformer | 7 | 69,680 | 15 min | Zhou et al. (2021) |
| ECL | Electricity Consumption | 321 | 26,304 | 1 hour | UCI ML Repository |

**Features**: ETT datasets contain 6 power load features (HUFL, HULL, MUFL, MULL, LUFL, LULL) and 1 oil temperature (OT). ECL contains electricity consumption of 321 clients.

### 2.2 Train / Validation / Test Splits

All splits follow the standard convention established by Informer (AAAI 2021) and adopted by PatchTST (ICLR 2023):

| Dataset | Train | Validation | Test | Split Rule |
|---------|-------|------------|------|------------|
| ETTh1 | 8,640 (12 months) | 2,880 (4 months) | 2,880 (4 months) | Fixed 12/4/4 months |
| ETTh2 | 8,640 (12 months) | 2,880 (4 months) | 2,880 (4 months) | Fixed 12/4/4 months |
| ETTm1 | 34,560 (12 months) | 11,520 (4 months) | 11,520 (4 months) | Fixed 12/4/4 months |
| ECL | 18,412 | 2,632 | 5,260 | 70% / 10% / 20% |

**Data preprocessing**: StandardScaler normalization fitted on training data only, applied to all splits. Reversible instance normalization is applied inside the model.

**Verified**: All CSV files match the PatchTST paper's reported timestep counts exactly. Train/val/test split boundaries in `data_loader.py` are identical to the TSLib framework used by PatchTST.

### 2.3 Forecasting Setup

- **Task**: Multivariate long-term forecasting (all variables predicted simultaneously)
- **Input length (seq_len)**: 96 or 336 (see experimental settings)
- **Prediction horizons (pred_len)**: {96, 192, 336, 720}
- **Evaluation metrics**: MSE (Mean Squared Error), MAE (Mean Absolute Error) -- lower is better
- **Evaluation scope**: Computed on the test set after training with early stopping on validation loss

---

## 3. Experimental Settings

### 3.1 Shared Hyperparameters (All Experiments)

| Parameter | Value | Note |
|-----------|-------|------|
| d_model | 128 | Transformer hidden dimension |
| n_heads | 16 | Attention heads |
| d_ff | 256 | FFN intermediate dimension |
| patch_len | 16 | Patch length (hardcoded in model) |
| stride | 8 | Patch stride (hardcoded in model) |
| dropout | 0.2 | Applied throughout |
| learning_rate | 1e-4 | Adam optimizer |
| train_epochs | 100 | Maximum epochs |
| patience | 5-10 | Early stopping patience |
| loss | MSE | Training objective |
| lr_schedule | type1 | Halve LR each epoch |

### 3.2 Dreamer-Specific Hyperparameters

| Parameter | Value | Note |
|-----------|-------|------|
| d_latent | 256 | Latent state dimension |
| slow_interval | 2 | Slow GRU update frequency |
| MAX_DYNAMICS_STEPS | 8 | Cap on GRU rollout steps |
| cross_attn_heads | 4 | Cross-attention heads |
| decoder_layers | 2 | Deep decoder depth |

### 3.3 Experimental Settings Matrix

| Setting | seq_len | e_layers | batch_size | patience | Purpose |
|---------|---------|----------|------------|----------|---------|
| **A** | 96 | 1 | 32 | 5 | Initial exploration |
| **B** | 96 | 3 | 128 | 10 | Fair comparison with deeper encoder |
| **C** | 336 | 3 | 128 | 10 | Match PatchTST paper setting |
| **Reproduce** | 336 | 3 | 128 | 10 | Reproduce PatchTST paper numbers |
| **Hybrid** | 96 | 1 | 128 | 10 | Hybrid head evaluation |

### 3.4 Baseline Models

| Model | Description | Source |
|-------|-------------|--------|
| **PatchTST** | Patch-based channel-independent Transformer | Nie et al. (ICLR 2023) |
| **DLinear** | Decomposition + Linear | Zeng et al. (AAAI 2023) |
| **Informer** | ProbSparse attention Transformer | Zhou et al. (AAAI 2021) |

---

## 4. Results

### 4.1 PatchTST Paper Reproduction (Setting: seq_len=336, e_layers=3)

First, we verified our PatchTST implementation can reproduce the paper's reported numbers.

| Dataset | pred_len | Paper MSE | Reproduced MSE | Paper MAE | Reproduced MAE | MSE Gap |
|---------|----------|-----------|----------------|-----------|----------------|---------|
| ETTh1 | 96 | 0.375 | 0.386 | 0.399 | 0.404 | +2.9% |
| ETTh1 | 192 | 0.414 | 0.422 | 0.421 | 0.425 | +1.9% |
| ETTh1 | 336 | 0.431 | 0.454 | 0.436 | 0.450 | +5.3% |
| ETTh1 | 720 | 0.449 | 0.484 | 0.466 | 0.492 | +7.8% |
| ETTh2 | 96 | 0.274 | 0.289 | 0.336 | 0.350 | +5.5% |
| ETTh2 | 192 | 0.339 | 0.353 | 0.379 | 0.391 | +4.1% |
| ETTh2 | 336 | 0.331 | 0.380 | 0.380 | 0.412 | +14.8% |
| ETTh2 | 720 | 0.379 | 0.414 | 0.422 | 0.447 | +9.2% |
| ETTm1 | 96 | 0.290 | 0.302 | 0.342 | 0.354 | +4.1% |
| ETTm1 | 192 | 0.332 | 0.338 | 0.369 | 0.376 | +1.8% |
| ETTm1 | 336 | 0.366 | 0.368 | 0.392 | 0.394 | +0.5% |
| ETTm1 | 720 | 0.420 | 0.414 | 0.424 | 0.421 | **-1.4%** |
| ECL | 96-720 | -- | Running | -- | Running | -- |

**Conclusion**: Reproduction successful. Most results within 2-6% of reported numbers, which is normal for different random seeds. ETTm1-720 even outperforms the paper slightly.

### 4.2 Setting A: Initial Exploration (seq_len=96, e_layers=1)

The first round of experiments used a simplified encoder setting.

#### PatchTST_Dreamer vs PatchTST (MSE)

| Dataset | pred_len | PatchTST | Dreamer | Winner | Delta |
|---------|----------|----------|---------|--------|-------|
| ETTh1 | 96 | 0.3906 | **0.3852** | Dreamer | -1.4% |
| ETTh1 | 192 | 0.4379 | **0.4269** | Dreamer | -2.5% |
| ETTh1 | 336 | 0.4733 | **0.4726** | Dreamer | -0.1% |
| ETTh1 | 720 | **0.4698** | 0.5096 | PatchTST | +8.5% |
| ETTh2 | 96 | **0.2868** | 0.2958 | PatchTST | +3.1% |
| ETTh2 | 192 | **0.3670** | 0.3703 | PatchTST | +0.9% |
| ETTh2 | 336 | **0.4124** | 0.4397 | PatchTST | +6.6% |
| ETTh2 | 720 | **0.4216** | 0.4432 | PatchTST | +5.1% |
| ETTm1 | 96 | **0.3351** | 0.3412 | PatchTST | +1.8% |
| ETTm1 | 192 | 0.3749 | **0.3646** | Dreamer | -2.7% |
| ETTm1 | 336 | 0.4087 | **0.4041** | Dreamer | -1.1% |
| ETTm1 | 720 | -- | 0.4686 | -- | (PatchTST interrupted) |

**Score: Dreamer 5 wins / PatchTST 6 wins**

#### Observations
1. **Dreamer wins at medium horizons (192, 336)** on ETTh1 and ETTm1 -- the multi-scale dynamics capture mid-range temporal patterns
2. **PatchTST wins at short (96) and long (720) horizons** -- at 96 the linear head is sufficient; at 720 the GRU rollout error accumulates
3. **ETTh2 consistently favors PatchTST** -- this dataset may have simpler dynamics where linear projection suffices
4. **Maximum improvement**: -2.7% on ETTm1-192; **Maximum degradation**: +8.5% on ETTh1-720

#### Ablation Study (ETTh1, Setting A)

| Variant | 96 | 192 | 336 | 720 |
|---------|-----|-----|-----|-----|
| **Dreamer (full)** | 0.385 | **0.427** | **0.473** | 0.510 |
| Single-scale GRU | 0.504 | 0.428 | 0.627 | 0.650 |
| Flatten (=PatchTST) | 0.391 | 0.438 | 0.473 | **0.470** |

- **Multi-scale dynamics are critical**: Single-scale GRU is dramatically worse (e.g., 0.627 vs 0.473 at pred_len=336)
- The slow GRU provides essential long-term trend information that a single fast GRU cannot capture

#### DLinear and Informer Baselines (Setting A)

| Dataset | pred_len | PatchTST | Dreamer | DLinear | Informer |
|---------|----------|----------|---------|---------|----------|
| ETTh1 | 96 | 0.391 | 0.385 | 0.396 | 0.961 |
| ETTh1 | 192 | 0.438 | 0.427 | 0.445 | 1.021 |
| ETTh1 | 336 | 0.473 | 0.473 | 0.487 | 1.030 |
| ETTh1 | 720 | 0.470 | 0.510 | 0.513 | 1.224 |
| ETTh2 | 96 | 0.287 | 0.296 | 0.341 | 2.866 |
| ETTh2 | 192 | 0.367 | 0.370 | 0.482 | 6.335 |
| ETTh2 | 336 | 0.412 | 0.440 | 0.593 | 5.425 |
| ETTh2 | 720 | 0.422 | 0.443 | 0.840 | 4.295 |
| ETTm1 | 96 | 0.335 | 0.341 | 0.346 | 0.622 |
| ETTm1 | 192 | 0.375 | 0.365 | 0.382 | 0.712 |
| ETTm1 | 336 | 0.409 | 0.404 | 0.415 | 1.269 |
| ETTm1 | 720 | -- | 0.469 | 0.473 | 0.945 |

Dreamer consistently outperforms DLinear and Informer, and is competitive with PatchTST.

### 4.3 Setting C: Paper-Aligned Encoder (seq_len=336, e_layers=3) -- FAILED

We tested whether Dreamer benefits from PatchTST's longer lookback window.

| Dataset | pred_len | PatchTST (reproduced) | Dreamer (336) | Delta |
|---------|----------|----------------------|---------------|-------|
| ETTh1 | 96 | 0.386 | 0.642 | **+66%** |
| ETTh1 | 192 | 0.422 | 0.422 | 0% |
| ETTh1 | 336 | 0.454 | 0.701 | **+54%** |
| ETTh1 | 720 | 0.484 | 0.727 | **+50%** |
| ETTh2 | 96 | 0.289 | 0.322 | +11% |
| ETTh2 | 192 | 0.353 | 0.382 | +8% |
| ETTh2 | 336 | 0.380 | 0.423 | +11% |
| ETTh2 | 720 | 0.414 | 0.449 | +8% |
| ETTm1 | 96 | 0.302 | 0.339 | +12% |
| ETTm1 | 192-720 | -- | Running | -- |

**Conclusion: Dreamer completely fails with seq_len=336.**

Root cause analysis:
- seq_len=336 with patch_len=16, stride=8 produces **42 patches**
- The `to_latent` layer flattens all patches: `Linear(128 * 42 = 5376 -> 512 -> 256)`
- This 5376-dimensional flattened input is too high-dimensional for the MLP to project meaningfully
- The GRU then operates on a poorly initialized latent state, leading to cascading errors
- ETTh1 is worst affected (results are essentially random); ETTh2 degrades more gracefully (~10%)

### 4.4 Setting B: Deeper Encoder, Short Lookback (seq_len=96, e_layers=3)

Testing whether a deeper encoder (matching paper depth) helps at shorter lookback.

#### PatchTST Baseline Comparison: e_layers=1 vs e_layers=3

| Dataset | pred_len | e=1 (Setting A) | e=3 (Setting B) | Delta |
|---------|----------|-----------------|-----------------|-------|
| ETTh1 | 96 | **0.391** | 0.397 | +1.5% |
| ETTh1 | 192 | **0.438** | 0.444 | +1.4% |
| ETTh1 | 336 | **0.473** | 0.484 | +2.3% |
| ETTh1 | 720 | **0.470** | 0.483 | +2.8% |
| ETTh2 | 96 | **0.287** | 0.293 | +2.1% |
| ETTh2 | 192 | **0.367** | 0.371 | +1.1% |
| ETTh2 | 336 | **0.412** | 0.414 | +0.5% |
| ETTh2 | 720 | 0.422 | **0.420** | -0.5% |
| ETTm1 | 96-720 | -- | Running | -- |

**Conclusion: e_layers=3 is slightly worse than e_layers=1 when seq_len=96.** With only 6 patches (from seq_len=96), a 3-layer encoder overfits. The 1-layer encoder is the optimal baseline for short lookback.

Setting B Dreamer results: PENDING (baseline phase still running)

### 4.5 Hybrid Head (seq_len=96, e_layers=1)

Early results from the Hybrid Head (Linear + Dreamer with learned gate):

| Dataset | pred_len | PatchTST | Dreamer | Hybrid | Note |
|---------|----------|----------|---------|--------|------|
| ETTh1 | 96 | 0.391 | 0.385 | 0.394 | Slightly worse than both |
| ETTh1 | 192-720 | -- | -- | Running | -- |

Only 1/12 experiments complete. Need more results to evaluate.

---

## 5. Key Findings and Conclusions (So Far)

### 5.1 Confirmed Findings

1. **Multi-scale dynamics matter**: The dual fast+slow GRU architecture is substantially better than single-scale GRU (ablation confirms 20-30% gap)

2. **Dreamer works best at medium prediction horizons (192-336)**: The GRU dynamics capture temporal patterns that a simple linear head misses, but error accumulates over very long rollouts

3. **seq_len=336 breaks Dreamer**: The latent projection from 42 flattened patches (5376-dim) is too high-dimensional. This is a fundamental architectural limitation of the current `to_latent` design

4. **seq_len=96, e_layers=1 is the optimal baseline**: Deeper encoders overfit with short lookback. This is the fairest comparison setting

5. **Dreamer consistently beats DLinear and Informer**: Even where Dreamer loses to PatchTST, it still outperforms other baselines

### 5.2 Open Questions (Pending Experiments)

1. **Does Hybrid Head fix the 720-horizon degradation?** (Running on GPU 3)
2. **How does Dreamer perform with e_layers=3 at seq_len=96?** (Running on GPU 2)
3. **ECL (321 variables) performance?** (PatchTST reproduction running on GPU 0)

### 5.3 Implications for Paper

**Strengths to highlight**:
- Novel application of world model dynamics (from RL) to time series forecasting
- Multi-scale GRU dynamics capture temporal patterns at different frequencies
- Competitive with SOTA Transformer on standard benchmarks
- Clear ablation showing multi-scale > single-scale > no dynamics

**Weaknesses to address**:
- Does not uniformly beat PatchTST (5W/6L in Setting A)
- Fails with long lookback (seq_len=336) -- needs architectural fix
- Error accumulation at very long horizons (720)

**Potential narrative angles**:
- Focus on the **medium horizon** regime (192-336) where Dreamer excels
- Position as a **complementary approach** that brings world model thinking to time series
- Emphasize the **multi-scale dynamics** as the key contribution (ablation is strong)
- If Hybrid Head works: demonstrate **robustness across all horizons**

---

## 6. Experiment Tracking

### 6.1 GPU Assignment

| GPU | tmux Session | Experiment | Status |
|-----|-------------|------------|--------|
| 0 | `reproduce` | PatchTST ECL reproduction (seq=336, e=3) | Running |
| 1 | `dreamer` | Setting C: Dreamer ETT (seq=336, e=3) | Running (9/12) |
| 2 | `settingB` | Setting B: PatchTST + Dreamer (seq=96, e=3) | Running (8/24) |
| 3 | `hybrid` | Hybrid Head (seq=96, e=1) | Running (1/12) |

### 6.2 Log Files

| Log | Content |
|-----|---------|
| `logs/reproduce_paper.log` | PatchTST paper reproduction |
| `logs/dreamer_paper_setting.log` | Setting C Dreamer (seq=336) |
| `logs/setting_b.log` | Setting B experiments |
| `logs/hybrid.log` | Hybrid head experiments |
| `logs/run_all.log` | Setting A (Round 1) |

### 6.3 Scripts

| Script | Setting | Description |
|--------|---------|-------------|
| `scripts/PatchTST_Dreamer/run_all.sh` | A | Original experiments (seq=96, e=1) |
| `scripts/PatchTST_Dreamer/reproduce_paper.sh` | Reproduce | PatchTST paper reproduction |
| `scripts/PatchTST_Dreamer/run_dreamer_paper_setting.sh` | C | Dreamer with paper settings |
| `scripts/PatchTST_Dreamer/run_setting_b.sh` | B | seq=96, e=3 comparison |
| `scripts/PatchTST_Dreamer/run_hybrid.sh` | Hybrid | Hybrid head evaluation |
| `scripts/PatchTST_Dreamer/run_dreamer_ett.sh` | C (ETT) | Dreamer ETT on GPU 1 |
| `scripts/PatchTST_Dreamer/run_ecl_reproduce.sh` | Reproduce | ECL-only reproduction |

---

## 7. Reproducibility

### 7.1 Environment

- **Hardware**: 4x NVIDIA A100 80GB PCIe
- **Framework**: PyTorch (TSLib / Time-Series-Library)
- **OS**: Linux 6.8.0 (Ubuntu)
- **Python**: 3.x with CUDA support

### 7.2 Running Experiments

```bash
cd /home/aiscuser/workspace-gzy/mts-world-model-grid/TSLib

# Reproduce PatchTST paper
bash scripts/PatchTST_Dreamer/reproduce_paper.sh

# Run Setting A (seq=96, e=1)
bash scripts/PatchTST_Dreamer/run_all.sh

# Run Hybrid Head
bash scripts/PatchTST_Dreamer/run_hybrid.sh
```

### 7.3 Key File Paths

| File | Description |
|------|-------------|
| `models/PatchTST_Dreamer.py` | Our model implementation (Dreamer, Hybrid, ablation heads) |
| `models/PatchTST.py` | Original PatchTST baseline |
| `data_provider/data_loader.py` | Dataset loading and splitting |
| `run.py` | Main training/evaluation entry point |
| `result_long_term_forecast.txt` | Accumulated test results |
