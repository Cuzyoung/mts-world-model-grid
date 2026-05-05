#!/bin/bash
# WorldTST-GRU: Targeted tuning for weak spots
# Focus on: ETTh2 (all), ETTm1-96, ETTm2-96, ETTh1-720
# Strategy: try e_layers, slow_interval, dropout, d_ff variations

set -e
cd /home/azureuser/workspace-gzy/mts-world-model-grid/TSLib
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "========================================="
echo "  WorldTST-GRU Tuning on GPU $GPU_ID"
echo "  Start: $(date)"
echo "========================================="

BASE="--task_name long_term_forecast --is_training 1 --features M --seq_len 96 --label_len 48 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --learning_rate 0.0001 --train_epochs 100 --itr 1 --clip_grad 1.0 --lradj type3 --patience 20 --model WorldTST_GRU"

# ===================================================================
# ETTh2: Currently losing (el=3 too deep for GRU?)
# Try: el=2, el=1, slow_interval=3, dropout=0.3
# ===================================================================
echo "===== ETTh2 Tuning ====="
ETTh2="--root_path ./dataset/ETT-small/ --data_path ETTh2.csv --data ETTh2"

# Variant A: e_layers=2 (reduce depth, might help GRU)
for pred_len in 96 192 336 720; do
    echo "[$(date '+%H:%M:%S')] ETTh2 el=2 pred=$pred_len"
    python -u run.py $BASE $ETTh2 \
        --model_id ETTh2_GRU_el2_${pred_len} \
        --pred_len $pred_len --e_layers 2 --n_heads 4 \
        --d_ff 256 --dropout 0.2 --slow_interval 2 \
        --batch_size 32 --des 'Tune_el2'
done

# Variant B: e_layers=1 (minimal depth)
for pred_len in 96 192 336 720; do
    echo "[$(date '+%H:%M:%S')] ETTh2 el=1 pred=$pred_len"
    python -u run.py $BASE $ETTh2 \
        --model_id ETTh2_GRU_el1_${pred_len} \
        --pred_len $pred_len --e_layers 1 --n_heads 4 \
        --d_ff 256 --dropout 0.2 --slow_interval 2 \
        --batch_size 32 --des 'Tune_el1'
done

# Variant C: slow_interval=3 + dropout=0.3
for pred_len in 96 192 336 720; do
    echo "[$(date '+%H:%M:%S')] ETTh2 si=3,do=0.3 pred=$pred_len"
    python -u run.py $BASE $ETTh2 \
        --model_id ETTh2_GRU_si3do3_${pred_len} \
        --pred_len $pred_len --e_layers 3 --n_heads 4 \
        --d_ff 256 --dropout 0.3 --slow_interval 3 \
        --batch_size 32 --des 'Tune_si3do3'
done

# ===================================================================
# ETTh1-720: Currently +6.0%, GRU long-seq weakness
# Try: slow_interval=4, d_ff=512, dropout=0.3
# ===================================================================
echo "===== ETTh1-720 Tuning ====="
ETTh1="--root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1"

echo "[$(date '+%H:%M:%S')] ETTh1-720 si=4"
python -u run.py $BASE $ETTh1 \
    --model_id ETTh1_GRU_720_si4 \
    --pred_len 720 --e_layers 1 --n_heads 16 \
    --d_ff 256 --dropout 0.2 --slow_interval 4 \
    --batch_size 32 --des 'Tune_si4'

echo "[$(date '+%H:%M:%S')] ETTh1-720 do=0.3"
python -u run.py $BASE $ETTh1 \
    --model_id ETTh1_GRU_720_do3 \
    --pred_len 720 --e_layers 1 --n_heads 16 \
    --d_ff 256 --dropout 0.3 --slow_interval 2 \
    --batch_size 32 --des 'Tune_do3'

echo "[$(date '+%H:%M:%S')] ETTh1-720 dff512"
python -u run.py $BASE $ETTh1 \
    --model_id ETTh1_GRU_720_dff512 \
    --pred_len 720 --e_layers 1 --n_heads 16 \
    --d_ff 512 --dropout 0.2 --slow_interval 2 \
    --batch_size 32 --des 'Tune_dff512'

# ===================================================================
# ETTm1-96: Currently +1.3%, close to flipping
# Try: e_layers=2, slow_interval=3
# ===================================================================
echo "===== ETTm1-96 Tuning ====="
ETTm1="--root_path ./dataset/ETT-small/ --data_path ETTm1.csv --data ETTm1"

echo "[$(date '+%H:%M:%S')] ETTm1-96 el=2"
python -u run.py $BASE $ETTm1 \
    --model_id ETTm1_GRU_96_el2 \
    --pred_len 96 --e_layers 2 --n_heads 2 \
    --d_ff 256 --dropout 0.2 --slow_interval 2 \
    --batch_size 32 --des 'Tune_el2'

echo "[$(date '+%H:%M:%S')] ETTm1-96 si=3"
python -u run.py $BASE $ETTm1 \
    --model_id ETTm1_GRU_96_si3 \
    --pred_len 96 --e_layers 1 --n_heads 2 \
    --d_ff 256 --dropout 0.2 --slow_interval 3 \
    --batch_size 32 --des 'Tune_si3'

# ===================================================================
# ETTm2-96: Currently +0.6%, almost there
# Try: e_layers=2, slow_interval=3
# ===================================================================
echo "===== ETTm2-96 Tuning ====="
ETTm2="--root_path ./dataset/ETT-small/ --data_path ETTm2.csv --data ETTm2"

echo "[$(date '+%H:%M:%S')] ETTm2-96 el=2"
python -u run.py $BASE $ETTm2 \
    --model_id ETTm2_GRU_96_el2 \
    --pred_len 96 --e_layers 2 --n_heads 16 \
    --d_ff 256 --dropout 0.2 --slow_interval 2 \
    --batch_size 32 --des 'Tune_el2'

echo "[$(date '+%H:%M:%S')] ETTm2-96 si=3"
python -u run.py $BASE $ETTm2 \
    --model_id ETTm2_GRU_96_si3 \
    --pred_len 96 --e_layers 3 --n_heads 16 \
    --d_ff 256 --dropout 0.2 --slow_interval 3 \
    --batch_size 32 --des 'Tune_si3'

echo ""
echo "========================================="
echo "  GRU Tuning: ALL DONE"
echo "  End: $(date)"
echo "========================================="
