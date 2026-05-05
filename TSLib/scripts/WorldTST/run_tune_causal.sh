#!/bin/bash
# WorldTST-Causal: Targeted tuning for weak spots
# Focus on: ETTh2-96/192/720 (currently losing)
# Strategy: try e_layers, n_heads, dropout variations

set -e
cd /home/azureuser/workspace-gzy/mts-world-model-grid/TSLib
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "========================================="
echo "  WorldTST-Causal Tuning on GPU $GPU_ID"
echo "  Start: $(date)"
echo "========================================="

BASE="--task_name long_term_forecast --is_training 1 --features M --seq_len 96 --label_len 48 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --learning_rate 0.0001 --train_epochs 100 --itr 1 --clip_grad 1.0 --lradj type3 --patience 20 --model WorldTST_Causal"

ETTh2="--root_path ./dataset/ETT-small/ --data_path ETTh2.csv --data ETTh2"

# ===================================================================
# ETTh2: Causal loses at 96 (+3.6%), 192 (+2.8%), 720 (+1.2%)
# Try: e_layers=2, e_layers=1, n_heads=8, dropout=0.3
# ===================================================================

# Variant A: e_layers=2
for pred_len in 96 192 720; do
    echo "[$(date '+%H:%M:%S')] ETTh2 Causal el=2 pred=$pred_len"
    python -u run.py $BASE $ETTh2 \
        --model_id ETTh2_Causal_el2_${pred_len} \
        --pred_len $pred_len --e_layers 2 --n_heads 4 \
        --d_ff 256 --dropout 0.2 \
        --batch_size 32 --des 'Tune_el2'
done

# Variant B: e_layers=1
for pred_len in 96 192 720; do
    echo "[$(date '+%H:%M:%S')] ETTh2 Causal el=1 pred=$pred_len"
    python -u run.py $BASE $ETTh2 \
        --model_id ETTh2_Causal_el1_${pred_len} \
        --pred_len $pred_len --e_layers 1 --n_heads 4 \
        --d_ff 256 --dropout 0.2 \
        --batch_size 32 --des 'Tune_el1'
done

# Variant C: n_heads=8, dropout=0.3
for pred_len in 96 192 720; do
    echo "[$(date '+%H:%M:%S')] ETTh2 Causal nh8do3 pred=$pred_len"
    python -u run.py $BASE $ETTh2 \
        --model_id ETTh2_Causal_nh8do3_${pred_len} \
        --pred_len $pred_len --e_layers 3 --n_heads 8 \
        --d_ff 256 --dropout 0.3 \
        --batch_size 32 --des 'Tune_nh8do3'
done

echo ""
echo "========================================="
echo "  Causal Tuning: ALL DONE"
echo "  End: $(date)"
echo "========================================="
