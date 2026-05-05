#!/bin/bash
# GPU 1: ETTh2 — PatchTST baseline + Dreamer
# ETTh2 official: e_layers=3, n_heads=4 for ALL horizons
set -e
cd /home/azureuser/workspace-gzy/mts-world-model-grid/TSLib
export CUDA_VISIBLE_DEVICES=1

echo "========================================="
echo "  GPU 1: ETTh2 Final Experiments"
echo "  Start: $(date)"
echo "========================================="

COMMON="--task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --data ETTh2 --features M --seq_len 96 --label_len 48 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --dropout 0.2 --learning_rate 0.0001 --train_epochs 100 --itr 1"

# ETTh2 official: e_layers=3, n_heads=4, batch_size=32

# ===== PatchTST Baseline =====
echo "===== PatchTST Baseline ====="

for pred_len in 96 192 336 720; do
    echo "[$(date '+%H:%M:%S')] PatchTST ETTh2 pred=$pred_len e_layers=3 n_heads=4"
    python -u run.py $COMMON \
        --model_id ETTh2_final_${pred_len} --model PatchTST \
        --pred_len $pred_len --e_layers 3 --n_heads 4 \
        --batch_size 32 --patience 10 --des 'Final'
done

# ===== PatchTST-Dreamer =====
echo "===== PatchTST-Dreamer ====="

for pred_len in 96 192 336 720; do
    echo "[$(date '+%H:%M:%S')] Dreamer ETTh2 pred=$pred_len e_layers=3 n_heads=4"
    python -u run.py $COMMON \
        --model_id ETTh2_final_${pred_len} --model PatchTST_Dreamer \
        --pred_len $pred_len --e_layers 3 --n_heads 4 \
        --d_latent 256 --slow_interval 2 --head_variant dreamer \
        --clip_grad 1.0 --lradj type3 \
        --batch_size 32 --patience 20 --des 'Final'
done

echo ""
echo "========================================="
echo "  GPU 1: ETTh2 ALL DONE"
echo "  End: $(date)"
echo "========================================="
