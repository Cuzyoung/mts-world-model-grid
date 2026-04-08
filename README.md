# MTS-WM: Multi-Time-Scale World Model for Power Grid Time Series Forecasting

## 1. Overview

This project applies **world model** ideas (from RL/control) to **power grid time series forecasting**. We propose **MTS-WM (Multi-Time-Scale World Model)**, a lightweight latent dynamics model that:

- **Encodes** historical observations into a latent state via patch embedding + Transformer encoder
- **Predicts** future latent states via **multi-scale dynamics** (fast + slow GRU at different temporal abstractions)
- **Decodes** latent states back to observation space, combined with a direct prediction shortcut

The core insight: power grid data exhibits **inherent multi-scale temporal structure** (hourly fluctuations, daily patterns, weekly cycles). A world model with explicit multi-timescale latent dynamics can naturally capture these patterns.

### Key References
- **Informer** (AAAI 2021): Long sequence time-series forecasting benchmark (ETT, ECL datasets)
- **PatchTST** (ICLR 2023): Patch-based Transformer for time series — we align all experimental settings with this paper
- **MTS3** (NeurIPS 2023): Multi Time Scale World Models for control
- **Dreamer** (ICLR 2020): Learning latent dynamics for planning

---

## 2. Experimental Settings (Aligned with PatchTST)

All settings follow [PatchTST (ICLR 2023)](https://arxiv.org/abs/2211.14730) exactly for fair comparison.

### 2.1 Forecasting Protocol

| Setting | Value |
|---------|-------|
| **Input length (seq_len)** | **336** |
| **Prediction horizons** | **{96, 192, 336, 720}** |
| **Forecasting type** | Multivariate (all features predicted jointly) |
| **Evaluation metrics** | MSE, MAE (computed on standardized data) |
| **Standardization** | StandardScaler fitted on training set only |

### 2.2 Datasets & Splits

| Dataset | Freq | Features | Total Length | Train | Val | Test |
|---------|------|----------|-------------|-------|-----|------|
| ETTh1 | 1h | 7 | 17,420 | 8,640 (12mo) | 2,880 (4mo) | 2,880 (4mo) |
| ETTh2 | 1h | 7 | 17,420 | 8,640 (12mo) | 2,880 (4mo) | 2,880 (4mo) |
| ETTm1 | 15min | 7 | 69,680 | 34,560 (12mo) | 11,520 (4mo) | 11,520 (4mo) |
| ECL | 1h | 321 | 26,304 | 18,412 (70%) | 2,631 (10%) | 5,261 (20%) |

ETT features: HUFL, HULL, MUFL, MULL, LUFL, LULL (load features) + OT (oil temperature)

### 2.3 Evaluation Details

- **Sliding window** with stride=1 on the test set
- Each sample: input `(seq_len, D)` -> predict `(pred_len, D)`
- Number of test samples = `test_len - seq_len - pred_len + 1`
- **MSE** = mean over all (samples x timesteps x features) squared errors
- **MAE** = mean over all (samples x timesteps x features) absolute errors
- All models compared under **identical** input/output/split/metric settings

### 2.4 Training Hyperparameters

| Parameter | ETTh1 | ETTh2 | ETTm1 | ECL |
|-----------|-------|-------|-------|-----|
| Batch size | 128 | 128 | 128 | 32 |
| Learning rate | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| Epochs (max) | 100 | 100 | 100 | 100 |
| Early stopping patience | 10 | 10 | 10 | 10 |
| Optimizer | Adam | Adam | Adam | Adam |
| LR scheduler | CosineAnnealing | CosineAnnealing | CosineAnnealing | CosineAnnealing |
| Gradient clipping | 1.0 | 1.0 | 1.0 | 1.0 |

---

## 3. Model Architecture (~14.4M parameters)

```
Input: x (B, 336, D)
         |
   [Patch Embedding]        -- patch_size=16, non-overlapping -> 21 patches
   [+ Sinusoidal PE]
         |
   [Transformer Encoder]    -- 4 layers, d_model=512, 8 heads, FFN=2048
   [+ Dropout]
         |
   [Mean Pooling -> Latent]  -- d_latent=256
         |
   [Multi-Scale Dynamics]    -- Core contribution
   |              |
[Fast GRU]    [Slow GRU]    -- fast: every step, slow: every 4 steps
   |              |
   [Learned Gating Fusion]
   [+ KL Regularization]
         |
   [Decoder MLP]             -- 3 layers, outputs patch_size * D
         |
   [+ Direct Linear Shortcut] -- residual path: Linear(336 -> pred_len)
         |
Output: predictions (B, pred_len, D)
```

### Key Design Choices
- **Multi-scale dynamics**: Fast GRU captures step-level transitions, Slow GRU (updated every K=4 steps) captures trends. A learned gate fuses them.
- **Direct shortcut**: Linear residual path provides strong gradient flow (similar to DLinear's insight that linear models are surprisingly strong).
- **KL regularization**: Encourages structured latent space, enables uncertainty estimation.

---

## 4. Baselines

We compare against established methods. Baseline numbers are cited from published papers; we also include our own reimplementations for verification.

| Model | Type | Reference | Params |
|-------|------|-----------|--------|
| **Informer** | ProbSparse attention Transformer | AAAI 2021 | ~11M |
| **Autoformer** | Auto-correlation Transformer | NeurIPS 2021 | ~10M |
| **FEDformer** | Frequency-enhanced Transformer | ICML 2022 | ~15M |
| **TimesNet** | Temporal 2D-variation | ICLR 2023 | ~4M |
| **DLinear** | Decomposition + Linear | AAAI 2023 | ~70K |
| **PatchTST** | Patch-based Transformer | ICLR 2023 | ~1-5M |

### Published Baseline Results (MSE, multivariate)

**ETTh1:**

| Model | 96 | 192 | 336 | 720 |
|-------|----|-----|-----|-----|
| Informer | 0.865 | 1.008 | 1.107 | 1.181 |
| Autoformer | 0.449 | 0.500 | 0.521 | 0.514 |
| FEDformer | 0.395 | 0.469 | 0.530 | 0.598 |
| TimesNet | 0.384 | 0.436 | 0.638 | 0.521 |
| DLinear | 0.397 | 0.446 | 0.489 | 0.513 |
| PatchTST | 0.377 | 0.431 | 0.477 | 0.484 |

**ETTh2:**

| Model | 96 | 192 | 336 | 720 |
|-------|----|-----|-----|-----|
| Informer | 3.755 | 5.602 | 4.721 | 3.647 |
| Autoformer | 0.346 | 0.456 | 0.482 | 0.515 |
| FEDformer | 0.358 | 0.429 | 0.496 | 0.463 |
| TimesNet | 0.340 | 0.402 | 0.452 | 0.462 |
| DLinear | 0.340 | 0.482 | 0.591 | 0.839 |
| PatchTST | 0.296 | 0.382 | 0.425 | 0.432 |

**ETTm1:**

| Model | 96 | 192 | 336 | 720 |
|-------|----|-----|-----|-----|
| Informer | 0.672 | 0.795 | 1.212 | 1.166 |
| Autoformer | 0.505 | 0.553 | 0.621 | 0.671 |
| FEDformer | 0.379 | 0.426 | 0.445 | 0.543 |
| TimesNet | 0.338 | 0.374 | 0.410 | 0.478 |
| DLinear | 0.346 | 0.382 | 0.415 | 0.473 |
| PatchTST | 0.332 | 0.367 | 0.410 | 0.459 |

**Electricity (ECL):**

| Model | 96 | 192 | 336 | 720 |
|-------|----|-----|-----|-----|
| Informer | 0.274 | 0.296 | 0.300 | 0.373 |
| Autoformer | 0.201 | 0.222 | 0.231 | 0.254 |
| FEDformer | 0.193 | 0.201 | 0.214 | 0.246 |
| TimesNet | 0.168 | 0.184 | 0.198 | 0.220 |
| DLinear | 0.210 | 0.210 | 0.223 | 0.258 |
| PatchTST | 0.200 | 0.203 | 0.194 | 0.260 |

---

## 5. Ablation Studies

Conducted on **ETTh1** across all four prediction horizons to validate each component:

| Variant | What Changes |
|---------|-------------|
| **MTS-WM (full)** | Complete model |
| **Single-Scale** | Replace multi-scale dynamics with single GRU |
| **w/o Slow** | Remove slow GRU, keep only fast dynamics |
| **w/o Fast** | Remove fast GRU, keep only slow dynamics |
| **w/o Direct** | Remove direct linear prediction shortcut |

---

## 6. Environment Setup

### Requirements
- Python >= 3.10
- PyTorch >= 2.0 (with CUDA support)

### Installation

```bash
git clone https://github.com/Cuzyoung/mts-world-model-grid.git
cd mts-world-model-grid
pip install -r requirements.txt
```

---

## 7. Data Download

```bash
bash scripts/download_data.sh
```

This downloads:
- **ETT**: from [ETDataset GitHub](https://github.com/zhouhaoyi/ETDataset)
- **ECL**: from [multivariate-time-series-data GitHub](https://github.com/laiguokun/multivariate-time-series-data)

---

## 8. Running Experiments

### Single experiment
```bash
# Train MTS-WM on ETTh1, pred_len=96
python src/train.py --config configs/etth1.yaml

# Train with different prediction horizon
python src/train.py --config configs/etth1.yaml --pred_len 336

# Train a baseline model
python src/train.py --config configs/etth1.yaml --model dlinear

# Run ablation variant
python src/train.py --config configs/etth1.yaml --model single_scale
```

### Full benchmark (all models x datasets x horizons)
```bash
bash scripts/run_all.sh
```

### Evaluate a saved checkpoint
```bash
python src/evaluate.py --config configs/etth1.yaml \
    --checkpoint checkpoints/ETTh1_mts_wm/pred96_best.pt
```

---

## 9. Project Structure

```
mts-world-model-grid/
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- etth1.yaml          # ETTh1 config (aligned with PatchTST)
|   |-- etth2.yaml
|   |-- ettm1.yaml
|   |-- ecl.yaml
|-- data/                    # Downloaded datasets (gitignored)
|   |-- ETT-small/
|   |-- ECL/
|-- src/
|   |-- models/
|   |   |-- world_model.py  # MTS-WM + ablation variants
|   |   |-- baselines.py    # LSTM, DLinear, Informer baselines
|   |-- data_loader.py      # Dataset classes + standard splits
|   |-- train.py            # Training with early stopping
|   |-- evaluate.py         # Load checkpoint + test evaluation
|   |-- metrics.py          # MSE, MAE
|   |-- utils.py            # Config loading, seeding, etc.
|-- scripts/
|   |-- download_data.sh    # One-click data download
|   |-- run_all.sh          # Full benchmark runner
|-- results/                 # JSON results per experiment
|-- checkpoints/             # Saved model weights
```

---

## 10. Available Models

| Name (--model) | Description |
|----------------|-------------|
| `mts_wm` | **MTS-WM (ours)** — full model |
| `single_scale` | Ablation: single GRU dynamics |
| `no_slow` | Ablation: fast dynamics only |
| `no_fast` | Ablation: slow dynamics only |
| `no_direct` | Ablation: without direct shortcut |
| `lstm` | Baseline: LSTM encoder-decoder |
| `dlinear` | Baseline: DLinear |
| `informer` | Baseline: simplified Informer |

---

## License

MIT
