#!/bin/bash
# GPU 2: ETTm1 — PatchTST baseline + Dreamer
# ETTm1 official: per-horizon hyperparams vary
set -e
cd /home/azureuser/workspace-gzy/mts-world-model-grid/TSLib
export CUDA_VISIBLE_DEVICES=2

echo "========================================="
echo "  GPU 2: ETTm1 Final Experiments"
echo "  Start: $(date)"
echo "========================================="

COMMON="--task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --data ETTm1 --features M --seq_len 96 --label_len 48 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --dropout 0.2 --learning_rate 0.0001 --train_epochs 100 --itr 1"

# ===== PatchTST Baseline (official per-horizon) =====
echo "===== PatchTST Baseline ====="

# ETTm1-96:  e=1, nh=2,  bs=32
echo "[$(date '+%H:%M:%S')] PatchTST ETTm1 pred=96"
python -u run.py $COMMON \
    --model_id ETTm1_final_96 --model PatchTST \
    --pred_len 96 --e_layers 1 --n_heads 2 \
    --batch_size 32 --patience 10 --des 'Final'

# ETTm1-192: e=3, nh=2,  bs=128
echo "[$(date '+%H:%M:%S')] PatchTST ETTm1 pred=192"
python -u run.py $COMMON \
    --model_id ETTm1_final_192 --model PatchTST \
    --pred_len 192 --e_layers 3 --n_heads 2 \
    --batch_size 128 --patience 10 --des 'Final'

# ETTm1-336: e=1, nh=4,  bs=128
echo "[$(date '+%H:%M:%S')] PatchTST ETTm1 pred=336"
python -u run.py $COMMON \
    --model_id ETTm1_final_336 --model PatchTST \
    --pred_len 336 --e_layers 1 --n_heads 4 \
    --batch_size 128 --patience 10 --des 'Final'

# ETTm1-720: e=3, nh=4,  bs=128
echo "[$(date '+%H:%M:%S')] PatchTST ETTm1 pred=720"
python -u run.py $COMMON \
    --model_id ETTm1_final_720 --model PatchTST \
    --pred_len 720 --e_layers 3 --n_heads 4 \
    --batch_size 128 --patience 10 --des 'Final'

# ===== PatchTST-Dreamer (same encoder, Dreamer head, tuned training) =====
echo "===== PatchTST-Dreamer ====="

# ETTm1-96: e=1, nh=2
echo "[$(date '+%H:%M:%S')] Dreamer ETTm1 pred=96"
python -u run.py $COMMON \
    --model_id ETTm1_final_96 --model PatchTST_Dreamer \
    --pred_len 96 --e_layers 1 --n_heads 2 \
    --d_latent 256 --slow_interval 2 --head_variant dreamer \
    --clip_grad 1.0 --lradj type3 \
    --batch_size 32 --patience 20 --des 'Final'

# ETTm1-192: e=3, nh=2
echo "[$(date '+%H:%M:%S')] Dreamer ETTm1 pred=192"
python -u run.py $COMMON \
    --model_id ETTm1_final_192 --model PatchTST_Dreamer \
    --pred_len 192 --e_layers 3 --n_heads 2 \
    --d_latent 256 --slow_interval 2 --head_variant dreamer \
    --clip_grad 1.0 --lradj type3 \
    --batch_size 128 --patience 20 --des 'Final'

# ETTm1-336: e=1, nh=4
echo "[$(date '+%H:%M:%S')] Dreamer ETTm1 pred=336"
python -u run.py $COMMON \
    --model_id ETTm1_final_336 --model PatchTST_Dreamer \
    --pred_len 336 --e_layers 1 --n_heads 4 \
    --d_latent 256 --slow_interval 2 --head_variant dreamer \
    --clip_grad 1.0 --lradj type3 \
    --batch_size 128 --patience 20 --des 'Final'

# ETTm1-720: e=3, nh=4
echo "[$(date '+%H:%M:%S')] Dreamer ETTm1 pred=720"
python -u run.py $COMMON \
    --model_id ETTm1_final_720 --model PatchTST_Dreamer \
    --pred_len 720 --e_layers 3 --n_heads 4 \
    --d_latent 256 --slow_interval 2 --head_variant dreamer \
    --clip_grad 1.0 --lradj type3 \
    --batch_size 128 --patience 20 --des 'Final'

echo ""
echo "========================================="
echo "  GPU 2: ETTm1 ALL DONE"
echo "  End: $(date)"
echo "========================================="
