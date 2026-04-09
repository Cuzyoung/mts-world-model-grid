# PatchTST-Dreamer: World Model Dynamics for Power Grid Time Series Forecasting

## Overview

We propose **PatchTST-Dreamer**, which replaces PatchTST's linear prediction head with a **Dreamer-style multi-scale latent dynamics model**. The PatchTST encoder (patch embedding + Transformer) is kept intact and can be initialized from pretrained weights.

**Key idea**: Instead of a single linear projection from encoder features to predictions, we:
1. Project encoder output to a latent state
2. Roll out future latent states using multi-scale GRU dynamics (fast + slow)
3. Decode each latent step back to a time-series patch

This brings **world model** thinking (from RL) into time series forecasting, with potential benefits in interpretability and uncertainty estimation.

### References
- **PatchTST** (ICLR 2023): Base encoder architecture
- **Dreamer** (ICLR 2020): Latent dynamics world model
- **MTS3** (NeurIPS 2023): Multi-time-scale world models
- **Informer** (AAAI 2021): ETT/ECL benchmark datasets

---

## Architecture

```
Original PatchTST:
  Input -> Patch+Embed -> Transformer -> [Flatten -> Linear] -> Prediction
                                       ^ replaced ^

PatchTST-Dreamer:
  Input -> Patch+Embed -> Transformer -> [Latent Proj -> Multi-Scale GRU -> Patch Decoder] -> Prediction
          (pretrained encoder)         (Dreamer head -- our contribution)
```

### Dreamer Head Details
- **Latent Projection**: Flatten encoder patches -> MLP -> z_0 (initial latent, per variable)
- **Multi-Scale Dynamics**: K rollout steps (K = pred_len / patch_len)
  - Fast GRU: updates every step (short-term)
  - Slow GRU: updates every 2 steps (long-term trends)
  - Learned gating fuses fast + slow
- **Patch Decoder**: Each z_k -> MLP -> one patch of predictions

---

## Data Verification (2026-04-08)

### Datasets confirmed present and correct

| Dataset | File | Rows | Features | Matches Paper |
|---------|------|------|----------|---------------|
| ETTh1 | `dataset/ETT-small/ETTh1.csv` | 17,420 | 7 | YES |
| ETTh2 | `dataset/ETT-small/ETTh2.csv` | 17,420 | 7 | YES |
| ETTm1 | `dataset/ETT-small/ETTm1.csv` | 69,680 | 7 | YES |
| ETTm2 | `dataset/ETT-small/ETTm2.csv` | **MISSING** | 7 | Need download |
| ECL | `dataset/electricity/electricity.csv` | 26,304 | 321 | YES |

### Train/Val/Test Splits (verified in `data_loader.py`)

| Dataset | Train | Val | Test | Split Rule | Matches Paper |
|---------|-------|-----|------|------------|---------------|
| ETTh1/h2 | 8,640 | 2,880 | 2,880 | 12/4/4 months | YES |
| ETTm1/m2 | 34,560 | 11,520 | 11,520 | 12/4/4 months | YES |
| ECL | 18,412 | 2,632 | 5,260 | 70/10/20% | YES |

**Conclusion: Data and splits are correct. No issues here.**

---

## Experimental Settings: Current vs PatchTST Paper

### Hyperparameter Comparison

| Parameter | Paper (PatchTST/42) | Our Current Setting | Status |
|-----------|---------------------|---------------------|--------|
| seq_len (lookback) | **336** | 96 | **MISMATCH - critical** |
| e_layers | **3** | 1 | **MISMATCH - critical** |
| pred_len | {96, 192, 336, 720} | {96, 192, 336, 720} | OK |
| d_model | 128 | 128 | OK |
| n_heads | 16 | 16 | OK |
| d_ff | 256 | 256 | OK |
| patch_len | 16 | 16 (hardcoded) | OK |
| stride | 8 | 8 (hardcoded) | OK |
| dropout | 0.2 | 0.2 | OK |
| learning_rate | 1e-4 | 1e-4 | OK |
| batch_size | 128 | 32 | **MISMATCH** |
| d_latent (Dreamer) | N/A | 256 | Our param |

### Dataset Coverage

| Dataset | Paper | Ours | Status |
|---------|-------|------|--------|
| ETTh1 | YES | YES | OK |
| ETTh2 | YES | YES | OK |
| ETTm1 | YES | YES | OK |
| ETTm2 | YES | NO | TODO (download data) |
| ECL | YES | Script ready, not run | TODO |
| Weather | YES | NO | Not required for now |
| Traffic | YES | NO | Not required for now |
| ILI | YES | NO | Not required for now |

---

## Current Results (Round 1: seq_len=96, e_layers=1)

> **WARNING**: These results use seq_len=96 and e_layers=1, which do NOT match the PatchTST paper settings (seq_len=336, e_layers=3). The PatchTST baseline here is weaker than the paper reports. These results are **NOT directly comparable** to the paper.

### PatchTST_Dreamer vs PatchTST (MSE, lower is better)

| Dataset | pred_len | Dreamer | PatchTST | Winner | Delta |
|---------|----------|---------|----------|--------|-------|
| ETTh1 | 96 | **0.3852** | 0.3906 | Dreamer | -1.4% |
| ETTh2 | 96 | 0.2958 | **0.2868** | PatchTST | +3.1% |
| ETTm1 | 96 | 0.3412 | **0.3351** | PatchTST | +1.8% |
| ETTh1 | 192 | **0.4269** | 0.4379 | Dreamer | -2.5% |
| ETTh2 | 192 | 0.3703 | **0.3670** | PatchTST | +0.9% |
| ETTm1 | 192 | **0.3646** | 0.3749 | Dreamer | -2.7% |
| ETTh1 | 336 | **0.4726** | 0.4733 | Dreamer | -0.1% |
| ETTh2 | 336 | 0.4397 | **0.4124** | PatchTST | +6.6% |
| ETTm1 | 336 | **0.4041** | 0.4087 | Dreamer | -1.1% |
| ETTh1 | 720 | 0.5096 | **0.4698** | PatchTST | +8.5% |
| ETTh2 | 720 | 0.4432 | **0.4216** | PatchTST | +5.1% |
| ETTm1 | 720 | 0.4686 | N/A (interrupted) | - | - |

**Round 1 Score: Dreamer 5 wins / PatchTST 6 wins** (ETTm1-720 incomplete)

### Observations from Round 1
1. Dreamer tends to win at **medium horizons** (192, 336), especially on ETTh1 and ETTm1
2. PatchTST wins at **short** (96) and **long** (720) horizons
3. **ETTh2 is consistently bad** for Dreamer across all horizons
4. At pred_len=720, Dreamer's GRU rollout accumulates error significantly

### Paper PatchTST/42 Reference (seq_len=336, e_layers=3)

| Dataset | 96 | 192 | 336 | 720 |
|---------|-----|-----|-----|-----|
| ETTh1 | 0.375 | 0.414 | 0.431 | 0.449 |
| ETTh2 | 0.274 | 0.339 | 0.331 | 0.379 |
| ETTm1 | 0.290 | 0.332 | 0.366 | 0.420 |

Our PatchTST (seq_len=96) is 4-25% worse than the paper, confirming the settings mismatch matters.

---

## Ablation Results (ETTh1 only, seq_len=96)

| Variant | 96 | 192 | 336 | 720 |
|---------|-----|-----|-----|-----|
| Dreamer (full) | **0.3852** | **0.4269** | **0.4726** | **0.5096** |
| Single-scale GRU | 0.5042 | 0.4279 | 0.6265 | 0.6499 |
| Flatten (=PatchTST) | 0.3906 | 0.4379 | 0.4733 | 0.4698 |

- **Multi-scale (fast+slow GRU) is critical** -- single-scale GRU is much worse
- Dreamer beats flatten at 96/192/336 but loses at 720

---

## Improvement Roadmap

### Phase 1: Reproduce Paper Baseline (PRIORITY)

**Goal**: Confirm PatchTST can reproduce paper numbers with correct settings.

```bash
# Correct settings to match PatchTST/42
--seq_len 336 --e_layers 3 --d_model 128 --n_heads 16 --d_ff 256 \
--dropout 0.2 --batch_size 128 --patch_len 16 --train_epochs 100 --patience 10
```

Run on: ETTh1, ETTh2, ETTm1, ETTm2, ECL

**Expected MSE** (from paper):
| Dataset | 96 | 192 | 336 | 720 |
|---------|-----|-----|-----|-----|
| ETTh1 | 0.375 | 0.414 | 0.431 | 0.449 |
| ETTh2 | 0.274 | 0.339 | 0.331 | 0.379 |
| ETTm1 | 0.290 | 0.332 | 0.366 | 0.420 |

### Phase 2: Re-run Dreamer with Correct Settings

After confirming PatchTST baseline, re-run PatchTST_Dreamer with:
- seq_len=336, e_layers=3 (same encoder as paper)
- d_latent=256, slow_interval=2
- The longer lookback (336) gives the Dreamer head much richer latent representations

### Phase 3: Address Dreamer Weaknesses

Based on Round 1 observations, potential improvements:

**Idea A: Scheduled Sampling / Teacher Forcing for Long Rollouts**
- Problem: At pred_len=720, GRU rollout (45 steps with patch_len=16) accumulates error
- Solution: During training, mix ground-truth patches with predicted latents
- Expected: Fix the 720-horizon degradation

**Idea B: Adaptive Slow Interval**
- Current: slow_interval=2 is fixed
- Idea: Learn the slow interval or use multiple slow scales (2, 4, 8)
- Expected: Better multi-scale dynamics capture

**Idea C: Latent Regularization (KL / Consistency)**
- Add Dreamer-style KL divergence loss on latent states
- Encourage smooth latent dynamics, reduce rollout drift
- Expected: More stable long-horizon predictions

**Idea D: Variable-Specific Dynamics**
- Current: Same GRU weights shared across all variables (channel-independent)
- Idea: Light variable-specific adaptation layers
- Risk: Overfitting on small datasets like ETT (only 7 variables)

**Idea E: Hybrid Head (Dreamer + Linear skip)**
- Add a residual linear prediction path alongside the Dreamer rollout
- Let linear handle the easy part, Dreamer focuses on nonlinear dynamics
- Expected: More robust across all horizons

---

## Quick Start

```bash
# 1. Install dependencies
pip install torch einops reformer-pytorch scikit-learn pandas matplotlib tqdm patool huggingface_hub sktime datasets

# 2. Download data
bash scripts/download_data.sh

# 3. Run single experiment (paper-aligned settings)
cd TSLib
python -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
    --model_id ETTh1_336_96 --model PatchTST_Dreamer --data ETTh1 \
    --features M --seq_len 336 --label_len 48 --pred_len 96 \
    --e_layers 3 --d_layers 1 --factor 3 \
    --enc_in 7 --dec_in 7 --c_out 7 \
    --d_model 128 --d_ff 256 --n_heads 16 \
    --d_latent 256 --slow_interval 2 --head_variant dreamer \
    --dropout 0.2 --batch_size 128 --des 'Exp' --itr 1

# 4. Run all experiments (in tmux)
tmux new-session -d -s exp \
    "bash scripts/PatchTST_Dreamer/run_all.sh 2>&1 | tee logs/run_all.log"
```

---

## Project Structure

```
mts-world-model-grid/
|-- README.md
|-- data/                          # Downloaded datasets
|   |-- ETT-small/
|   +-- ECL/
+-- TSLib/                         # Time-Series-Library framework
    |-- models/
    |   |-- PatchTST_Dreamer.py   # Our model
    |   |-- PatchTST.py           # Baseline
    |   |-- DLinear.py            # Baseline
    |   |-- Informer.py           # Baseline
    |   +-- ...
    |-- scripts/
    |   +-- PatchTST_Dreamer/
    |       +-- run_all.sh        # Full benchmark script
    |-- exp/                       # Training/evaluation framework
    |-- data_provider/             # Data loading
    |-- layers/                    # Transformer layers
    |-- run.py                     # Main entry point
    |-- results/                   # Saved metrics
    +-- checkpoints/               # Model weights
```

---

## License

MIT
