# MTS-WorldModel-Grid: Multi-Time-Scale World Model for Power Grid Time Series Forecasting

## 1. Overview

This project applies **world model** ideas (from RL/control) to **power grid time series forecasting**. We propose **MTS-WM (Multi-Time-Scale World Model)**, a lightweight latent dynamics model that:

- **Encodes** historical observations into a latent state
- **Predicts** future latent states via **multi-scale dynamics** (fast + slow temporal abstractions)
- **Decodes** latent states back to observation space

The core insight: power grid data exhibits **inherent multi-scale temporal structure** (hourly fluctuations, daily patterns, weekly cycles). A world model with explicit multi-timescale latent dynamics can naturally capture these patterns.

### Key References
- **Informer** (AAAI 2021): Long sequence time-series forecasting benchmark (ETT, ECL datasets)
- **MTS3** (NeurIPS 2023): Multi Time Scale World Models for control
- **Dreamer** (ICLR 2020): Learning latent dynamics for planning

---

## 2. Environment Setup

### 2.1 Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.8 (optional, for GPU training)

### 2.2 Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/mts-world-model-grid.git
cd mts-world-model-grid

# Create conda environment (recommended)
conda create -n mtswm python=3.10 -y
conda activate mtswm

# Install dependencies
pip install -r requirements.txt
```

### 2.3 Dependencies (requirements.txt)

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
pyyaml>=6.0
tqdm>=4.65.0
```

---

## 3. Datasets

We evaluate on the standard Informer benchmarks:

### 3.1 ETT (Electricity Transformer Temperature)

| Dataset | Granularity | Features | Length | Source |
|---------|-------------|----------|--------|--------|
| ETTh1   | 1 hour      | 7        | 17,420 | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| ETTh2   | 1 hour      | 7        | 17,420 | Same |
| ETTm1   | 15 min      | 7        | 69,680 | Same |

Features: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT (oil temperature)

### 3.2 ECL (Electricity Consuming Load)

| Dataset | Granularity | Features | Length  | Source |
|---------|-------------|----------|---------|--------|
| ECL     | 1 hour      | 321      | 26,304  | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) |

### 3.3 Download

```bash
# Automatic download
bash scripts/download_data.sh

# Or manual:
# ETT: https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small
# ECL: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
```

---

## 4. Model Architecture

```
Input: x_{t-L:t} (history window, L steps, D features)
         |
   [Patch Embedding]     -- split into patches of size P
         |
   [Temporal Encoder]    -- Transformer encoder, outputs h_t
         |
   [Latent Projection]   -- h_t -> z_t (latent state)
         |
   [Multi-Scale Dynamics] -- Core contribution
   |         |
[Fast GRU]  [Slow GRU]   -- fast: step-level, slow: every K steps
   |         |
   [Fusion Layer]         -- combine fast + slow predictions
         |
   [Decoder MLP]          -- z_{t+1:t+H} -> x_hat_{t+1:t+H}
         |
Output: predictions for H future steps
```

### 4.1 Multi-Scale Dynamics (Core)

The key component operates at two timescales:

- **Fast dynamics** (`FastGRU`): models step-by-step transitions, captures local/short-term patterns
- **Slow dynamics** (`SlowGRU`): updates every K steps, captures global/long-term trends
- **Fusion**: learned gating mechanism combines fast and slow latent predictions

This mirrors how power grids operate: fast load fluctuations overlaid on slow seasonal/daily trends.

---

## 5. Experiment Design

### 5.1 Main Comparison (Table 1 in paper)

Compare MTS-WM against established time series forecasting baselines:

| Baseline | Type | Reference |
|----------|------|-----------|
| **Informer** | Sparse attention Transformer | AAAI 2021 |
| **Autoformer** | Auto-correlation Transformer | NeurIPS 2021 |
| **PatchTST** | Patched Transformer | ICLR 2023 |
| **DLinear** | Simple linear model | AAAI 2023 |
| **LSTM** | Recurrent baseline | Classic |

### 5.2 Forecasting Settings

Follow the standard Informer protocol:

| Setting | Input Length (L) | Prediction Horizons (H) |
|---------|-----------------|------------------------|
| Standard | 96 | {96, 192, 336, 720} |

**Metrics**: MSE (Mean Squared Error), MAE (Mean Absolute Error)

**Data Split**: Train / Val / Test = 60% / 20% / 20% (chronological)

### 5.3 Ablation Studies (Table 2 in paper)

To validate each component's contribution:

| Ablation | Description |
|----------|-------------|
| **w/o Slow Dynamics** | Remove slow GRU, keep only fast dynamics |
| **w/o Fast Dynamics** | Remove fast GRU, keep only slow dynamics |
| **w/o Multi-Scale Fusion** | Replace learned gating with simple concatenation |
| **Single-Scale WM** | Replace multi-scale with single GRU (standard Dreamer) |
| **w/o Latent Prediction Loss** | Remove L_latent, train with reconstruction loss only |

### 5.4 Analysis Experiments

1. **Latent space visualization**: t-SNE of z_t colored by time-of-day / day-of-week
2. **Uncertainty estimation**: Plot prediction intervals from latent variance
3. **Scale-specific analysis**: What does fast vs. slow latent capture?

---

## 6. Training & Evaluation

### 6.1 Training

```bash
# Train on ETTh1 with default config
python src/train.py --config configs/etth1.yaml

# Train on ECL
python src/train.py --config configs/ecl.yaml

# Train with custom horizon
python src/train.py --config configs/etth1.yaml --pred_len 336
```

### 6.2 Evaluation

```bash
# Evaluate a trained model
python src/evaluate.py --config configs/etth1.yaml --checkpoint checkpoints/etth1_best.pt

# Run all experiments (all datasets x all horizons)
bash scripts/run_all.sh
```

### 6.3 Baselines

```bash
# Run baseline models for comparison
python src/train.py --config configs/etth1.yaml --model informer
python src/train.py --config configs/etth1.yaml --model dlinear
python src/train.py --config configs/etth1.yaml --model lstm
```

---

## 7. Project Structure

```
mts-world-model-grid/
|-- README.md                  # This file
|-- requirements.txt           # Python dependencies
|-- configs/                   # YAML experiment configs
|   |-- etth1.yaml
|   |-- etth2.yaml
|   |-- ettm1.yaml
|   |-- ecl.yaml
|-- data/                      # Downloaded datasets
|   |-- ETT-small/
|   |-- ECL/
|-- src/                       # Source code
|   |-- __init__.py
|   |-- models/
|   |   |-- __init__.py
|   |   |-- world_model.py    # MTS-WM (our model)
|   |   |-- baselines.py      # Informer, DLinear, LSTM
|   |-- data_loader.py         # Dataset classes
|   |-- train.py               # Training script
|   |-- evaluate.py            # Evaluation script
|   |-- metrics.py             # MSE, MAE computation
|   |-- utils.py               # Helpers
|-- scripts/
|   |-- download_data.sh       # Data download script
|   |-- run_all.sh             # Run all experiments
|-- results/                   # Saved results & plots
|-- checkpoints/               # Model checkpoints
```

---

## 8. Quick Start

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Download data
bash scripts/download_data.sh

# 3. Train MTS-WM on ETTh1
python src/train.py --config configs/etth1.yaml

# 4. Evaluate
python src/evaluate.py --config configs/etth1.yaml --checkpoint checkpoints/etth1_best.pt

# 5. Run full benchmark
bash scripts/run_all.sh
```

---

## 9. Expected Results (Rough Targets)

We do NOT aim to beat SOTA. The goal is to show world model ideas are applicable to grid forecasting with competitive performance + added benefits (uncertainty, interpretable multi-scale latents).

| Model | ETTh1 (96) MSE | ETTh1 (336) MSE | Note |
|-------|----------------|-----------------|------|
| Informer | ~0.098 | ~0.214 | Published |
| DLinear | ~0.075 | ~0.197 | Published |
| PatchTST | ~0.070 | ~0.187 | Published |
| **MTS-WM (ours)** | ~0.080-0.095 | ~0.195-0.215 | Target range |

---

## License

MIT
