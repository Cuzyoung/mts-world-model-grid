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
  Input → Patch+Embed → Transformer → [Flatten → Linear] → Prediction
                                       ↑ replaced ↑

PatchTST-Dreamer:
  Input → Patch+Embed → Transformer → [Latent Proj → Multi-Scale GRU → Patch Decoder] → Prediction
          (pretrained encoder)         (Dreamer head — our contribution)
```

### Dreamer Head Details
- **Latent Projection**: Flatten encoder patches → MLP → z₀ (initial latent, per variable)
- **Multi-Scale Dynamics**: K rollout steps (K = pred_len / patch_len)
  - Fast GRU: updates every step (short-term)
  - Slow GRU: updates every 2 steps (long-term trends)
  - Learned gating fuses fast + slow
- **Patch Decoder**: Each z_k → MLP → one patch of predictions

---

## Experimental Settings (Aligned with PatchTST / TSLib)

| Setting | Value |
|---------|-------|
| Input length (seq_len) | 96 |
| Prediction horizons | {96, 192, 336, 720} |
| Forecasting type | Multivariate (M) |
| Metrics | MSE, MAE |
| Encoder | 1 layer, d_model=128, n_heads=16, d_ff=256 |
| Dreamer head | d_latent=128, slow_interval=2 |
| Training | Adam, lr=1e-4, 100 epochs, patience=5 |

### Datasets

| Dataset | Freq | Features | Train/Val/Test |
|---------|------|----------|----------------|
| ETTh1 | 1h | 7 | 8640/2880/2880 |
| ETTh2 | 1h | 7 | 8640/2880/2880 |
| ETTm1 | 15min | 7 | 34560/11520/11520 |
| ECL | 1h | 321 | 70%/10%/20% |

---

## Models Compared

| Model | Description |
|-------|-------------|
| **PatchTST_Dreamer** (ours) | PatchTST encoder + Multi-scale Dreamer head |
| PatchTST | Original PatchTST (Flatten + Linear head) |
| DLinear | Decomposition + Linear |
| Informer | ProbSparse attention Transformer |

### Ablation Variants (same encoder, different heads)

| Variant | --head_variant | Description |
|---------|---------------|-------------|
| Full Dreamer | `dreamer` | Fast GRU + Slow GRU + gating |
| Single-scale | `single_scale` | Single GRU (no multi-scale) |
| Flatten (PatchTST) | `flatten` | Original linear head |

---

## Quick Start

```bash
# 1. Install dependencies
pip install torch einops reformer-pytorch scikit-learn pandas matplotlib tqdm patool huggingface_hub sktime datasets

# 2. Download data
bash scripts/download_data.sh

# 3. Run single experiment
cd TSLib
python -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
    --model_id ETTh1_96_96 --model PatchTST_Dreamer --data ETTh1 \
    --features M --seq_len 96 --label_len 48 --pred_len 96 \
    --e_layers 1 --d_layers 1 --factor 3 \
    --enc_in 7 --dec_in 7 --c_out 7 \
    --d_model 128 --d_ff 256 --n_heads 16 \
    --d_latent 128 --slow_interval 2 --head_variant dreamer \
    --dropout 0.2 --des 'Exp' --itr 1

# 4. Run all experiments (in tmux)
tmux new-session -d -s exp \
    "bash scripts/PatchTST_Dreamer/run_all.sh 2>&1 | tee logs/run_all.log"
```

---

## Project Structure

```
mts-world-model-grid/
├── README.md
├── data/                          # Downloaded datasets
│   ├── ETT-small/
│   └── ECL/
└── TSLib/                         # Time-Series-Library framework
    ├── models/
    │   ├── PatchTST_Dreamer.py   # Our model
    │   ├── PatchTST.py           # Baseline
    │   ├── DLinear.py            # Baseline
    │   ├── Informer.py           # Baseline
    │   └── ...
    ├── scripts/
    │   └── PatchTST_Dreamer/
    │       └── run_all.sh        # Full benchmark script
    ├── exp/                       # Training/evaluation framework
    ├── data_provider/             # Data loading
    ├── layers/                    # Transformer layers
    ├── run.py                     # Main entry point
    ├── results/                   # Saved metrics
    └── checkpoints/               # Model weights
```

---

## License

MIT
