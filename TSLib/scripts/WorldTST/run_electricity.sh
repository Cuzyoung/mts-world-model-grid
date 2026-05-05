#!/bin/bash
# Electricity benchmark — 3 phases:
#   Phase 1: PatchTST baseline (TSLib official defaults: d_model=512, d_ff=2048, etc.)
#   Phase 2: WorldTST_GRU (our settings: d_model=128, d_ff=256)
#   Phase 3: WorldTST_Causal (our settings: d_model=128, d_ff=256)
#
# PatchTST uses its OWN best settings for fair reproduction.
# Our models use our common settings consistent with ETT experiments.
#
# Usage: bash run_electricity.sh <GPU_ID>

GPU_ID=${1:-3}

set -e
cd /home/azureuser/workspace-gzy/mts-world-model-grid/TSLib
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Private TMPDIR to avoid /tmp pressure (short path for AF_UNIX 108-char limit)
export TMPDIR=/tmp/gzy_ecl_gpu${GPU_ID}
mkdir -p $TMPDIR

echo "========================================="
echo "  Electricity Benchmark (GPU $GPU_ID)"
echo "  TMPDIR=$TMPDIR"
echo "  Start: $(date)"
echo "========================================="

# ECL: 321 variables, target=MT_320 (last column)
ECL_COMMON="--task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --data custom --features M --seq_len 96 --label_len 48 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --target MT_320 --itr 1"

# ===== Phase 1: PatchTST Baseline (TSLib official defaults) =====
# Official PatchTST ECL: e_layers=2, batch_size=16
# d_model=512, n_heads=8, d_ff=2048, dropout=0.1, lr=1e-4, patience=3, lradj=type1
echo ""
echo "===== Phase 1: PatchTST Baseline (official settings) ====="
for pred_len in 96 192 336 720; do
    echo "[$(date '+%H:%M:%S')] PatchTST ECL pred=$pred_len"
    python -u run.py $ECL_COMMON \
        --model_id ECL_baseline_${pred_len} --model PatchTST \
        --pred_len $pred_len --e_layers 2 --n_heads 8 \
        --d_model 512 --d_ff 2048 --dropout 0.1 \
        --learning_rate 0.0001 --lradj type1 --patience 3 \
        --batch_size 16 --train_epochs 100 \
        --des 'Baseline'
done

echo ""
echo "===== PatchTST Baseline: DONE ====="
echo ""

# ===== Phase 2: WorldTST_GRU (our settings) =====
# Our common: d_model=128, d_ff=256, dropout=0.2, lr=1e-4, lradj=type3, patience=20, clip_grad=1.0
# ECL unified: el=2, nh=4, bs=16 (bs=16 due to 321 vars memory)
echo "===== Phase 2: WorldTST_GRU (our settings) ====="
OUR_COMMON="$ECL_COMMON --d_model 128 --d_ff 256 --dropout 0.2 --learning_rate 0.0001 --lradj type3 --patience 20 --clip_grad 1.0 --train_epochs 100"

for pred_len in 96 192 336 720; do
    echo "[$(date '+%H:%M:%S')] WorldTST_GRU ECL pred=$pred_len"
    python -u run.py $OUR_COMMON \
        --model_id ECL_GRU_${pred_len} --model WorldTST_GRU \
        --pred_len $pred_len --e_layers 2 --n_heads 4 --batch_size 16 \
        --slow_interval 2 --des 'ECL'
done

echo ""
echo "===== WorldTST_GRU: DONE ====="
echo ""

# ===== Phase 3: WorldTST_Causal (our settings) =====
echo "===== Phase 3: WorldTST_Causal (our settings) ====="
for pred_len in 96 192 336 720; do
    echo "[$(date '+%H:%M:%S')] WorldTST_Causal ECL pred=$pred_len"
    python -u run.py $OUR_COMMON \
        --model_id ECL_Causal_${pred_len} --model WorldTST_Causal \
        --pred_len $pred_len --e_layers 2 --n_heads 4 --batch_size 16 \
        --des 'ECL'
done

echo ""
echo "========================================="
echo "  Electricity Benchmark: ALL DONE"
echo "  End: $(date)"
echo "========================================="

# Cleanup tmp
rm -rf $TMPDIR
