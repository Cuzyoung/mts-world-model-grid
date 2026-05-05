#!/bin/bash
# PatchTST-Dreamer with paper-aligned encoder settings
# Encoder: seq_len=336, e_layers=3, d_model=128, n_heads=16, d_ff=256
# Dreamer head: d_latent=256, slow_interval=2
# Datasets: ETTh1, ETTh2, ETTm1, ECL

set -e
cd /home/aiscuser/workspace-gzy/mts-world-model-grid/TSLib

export CUDA_VISIBLE_DEVICES=0

echo "========================================="
echo "  PatchTST-Dreamer (Paper-Aligned Encoder)"
echo "  Start: $(date)"
echo "========================================="

# Paper-aligned encoder settings
SEQ_LEN=336
E_LAYERS=3
D_MODEL=128
N_HEADS=16
D_FF=256
DROPOUT=0.2
LR=0.0001
BATCH_SIZE=128
EPOCHS=100
PATIENCE=10

# Dreamer head settings
D_LATENT=256
SLOW_INTERVAL=2

# ============================================
# ETTh1
# ============================================
for pred_len in 96 192 336 720; do
    echo ""
    echo "[$(date '+%H:%M:%S')] PatchTST_Dreamer ETTh1 seq=${SEQ_LEN} pred=${pred_len}"
    python -u run.py \
        --task_name long_term_forecast --is_training 1 \
        --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
        --model_id ETTh1_${SEQ_LEN}_${pred_len} --model PatchTST_Dreamer --data ETTh1 \
        --features M --seq_len $SEQ_LEN --label_len 48 --pred_len $pred_len \
        --e_layers $E_LAYERS --d_layers 1 --factor 3 \
        --enc_in 7 --dec_in 7 --c_out 7 \
        --d_model $D_MODEL --d_ff $D_FF --n_heads $N_HEADS \
        --d_latent $D_LATENT --slow_interval $SLOW_INTERVAL --head_variant dreamer \
        --dropout $DROPOUT --learning_rate $LR \
        --batch_size $BATCH_SIZE --des 'DreamerV2' --itr 1 \
        --train_epochs $EPOCHS --patience $PATIENCE
done

# ============================================
# ETTh2
# ============================================
for pred_len in 96 192 336 720; do
    echo ""
    echo "[$(date '+%H:%M:%S')] PatchTST_Dreamer ETTh2 seq=${SEQ_LEN} pred=${pred_len}"
    python -u run.py \
        --task_name long_term_forecast --is_training 1 \
        --root_path ./dataset/ETT-small/ --data_path ETTh2.csv \
        --model_id ETTh2_${SEQ_LEN}_${pred_len} --model PatchTST_Dreamer --data ETTh2 \
        --features M --seq_len $SEQ_LEN --label_len 48 --pred_len $pred_len \
        --e_layers $E_LAYERS --d_layers 1 --factor 3 \
        --enc_in 7 --dec_in 7 --c_out 7 \
        --d_model $D_MODEL --d_ff $D_FF --n_heads $N_HEADS \
        --d_latent $D_LATENT --slow_interval $SLOW_INTERVAL --head_variant dreamer \
        --dropout $DROPOUT --learning_rate $LR \
        --batch_size $BATCH_SIZE --des 'DreamerV2' --itr 1 \
        --train_epochs $EPOCHS --patience $PATIENCE
done

# ============================================
# ETTm1
# ============================================
for pred_len in 96 192 336 720; do
    echo ""
    echo "[$(date '+%H:%M:%S')] PatchTST_Dreamer ETTm1 seq=${SEQ_LEN} pred=${pred_len}"
    python -u run.py \
        --task_name long_term_forecast --is_training 1 \
        --root_path ./dataset/ETT-small/ --data_path ETTm1.csv \
        --model_id ETTm1_${SEQ_LEN}_${pred_len} --model PatchTST_Dreamer --data ETTm1 \
        --features M --seq_len $SEQ_LEN --label_len 48 --pred_len $pred_len \
        --e_layers $E_LAYERS --d_layers 1 --factor 3 \
        --enc_in 7 --dec_in 7 --c_out 7 \
        --d_model $D_MODEL --d_ff $D_FF --n_heads $N_HEADS \
        --d_latent $D_LATENT --slow_interval $SLOW_INTERVAL --head_variant dreamer \
        --dropout $DROPOUT --learning_rate $LR \
        --batch_size $BATCH_SIZE --des 'DreamerV2' --itr 1 \
        --train_epochs $EPOCHS --patience $PATIENCE
done

# ============================================
# ECL (smaller batch due to 321 features)
# ============================================
for pred_len in 96 192 336 720; do
    echo ""
    echo "[$(date '+%H:%M:%S')] PatchTST_Dreamer ECL seq=${SEQ_LEN} pred=${pred_len}"
    python -u run.py \
        --task_name long_term_forecast --is_training 1 \
        --root_path ./dataset/electricity/ --data_path electricity.csv \
        --model_id ECL_${SEQ_LEN}_${pred_len} --model PatchTST_Dreamer --data custom \
        --features M --target MT_320 --seq_len $SEQ_LEN --label_len 48 --pred_len $pred_len \
        --e_layers $E_LAYERS --d_layers 1 --factor 3 \
        --enc_in 321 --dec_in 321 --c_out 321 \
        --d_model $D_MODEL --d_ff $D_FF --n_heads $N_HEADS \
        --d_latent $D_LATENT --slow_interval $SLOW_INTERVAL --head_variant dreamer \
        --dropout $DROPOUT --learning_rate $LR \
        --batch_size 32 --des 'DreamerV2' --itr 1 \
        --train_epochs $EPOCHS --patience $PATIENCE
done

echo ""
echo "========================================="
echo "  PatchTST-Dreamer (paper setting) done!"
echo "  End: $(date)"
echo "========================================="
