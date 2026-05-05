#!/bin/bash
# Setting B: seq_len=96, e_layers=3
# Fair comparison: same encoder depth as paper, shorter lookback
# PatchTST baseline + Dreamer, both with e_layers=3
set -e
cd /home/aiscuser/workspace-gzy/mts-world-model-grid/TSLib
export CUDA_VISIBLE_DEVICES=2

SEQ_LEN=96
E_LAYERS=3
D_MODEL=128
N_HEADS=16
D_FF=256
DROPOUT=0.2
LR=0.0001
BATCH_SIZE=128
EPOCHS=100
PATIENCE=10
D_LATENT=256
SLOW_INTERVAL=2

echo "========================================="
echo "  Setting B: seq=96, e_layers=3 (GPU 2)"
echo "  Start: $(date)"
echo "========================================="

# --- PatchTST Baseline ---
echo "===== PatchTST Baseline (seq=96, e_layers=3) ====="
for dataset in ETTh1 ETTh2 ETTm1; do
    for pred_len in 96 192 336 720; do
        echo ""
        echo "[$(date '+%H:%M:%S')] PatchTST ${dataset} seq=${SEQ_LEN} e=${E_LAYERS} pred=${pred_len}"
        python -u run.py \
            --task_name long_term_forecast --is_training 1 \
            --root_path ./dataset/ETT-small/ --data_path ${dataset}.csv \
            --model_id ${dataset}_${SEQ_LEN}_${pred_len}_e3 --model PatchTST --data ${dataset} \
            --features M --seq_len $SEQ_LEN --label_len 48 --pred_len $pred_len \
            --e_layers $E_LAYERS --d_layers 1 --factor 3 \
            --enc_in 7 --dec_in 7 --c_out 7 \
            --d_model $D_MODEL --d_ff $D_FF --n_heads $N_HEADS \
            --dropout $DROPOUT --learning_rate $LR \
            --batch_size $BATCH_SIZE --des 'SettingB' --itr 1 \
            --train_epochs $EPOCHS --patience $PATIENCE
    done
done

# --- PatchTST_Dreamer ---
echo ""
echo "===== PatchTST_Dreamer (seq=96, e_layers=3) ====="
for dataset in ETTh1 ETTh2 ETTm1; do
    for pred_len in 96 192 336 720; do
        echo ""
        echo "[$(date '+%H:%M:%S')] Dreamer ${dataset} seq=${SEQ_LEN} e=${E_LAYERS} pred=${pred_len}"
        python -u run.py \
            --task_name long_term_forecast --is_training 1 \
            --root_path ./dataset/ETT-small/ --data_path ${dataset}.csv \
            --model_id ${dataset}_${SEQ_LEN}_${pred_len}_e3 --model PatchTST_Dreamer --data ${dataset} \
            --features M --seq_len $SEQ_LEN --label_len 48 --pred_len $pred_len \
            --e_layers $E_LAYERS --d_layers 1 --factor 3 \
            --enc_in 7 --dec_in 7 --c_out 7 \
            --d_model $D_MODEL --d_ff $D_FF --n_heads $N_HEADS \
            --d_latent $D_LATENT --slow_interval $SLOW_INTERVAL --head_variant dreamer \
            --dropout $DROPOUT --learning_rate $LR \
            --batch_size $BATCH_SIZE --des 'SettingB' --itr 1 \
            --train_epochs $EPOCHS --patience $PATIENCE
    done
done

echo ""
echo "========================================="
echo "  Setting B done!"
echo "  End: $(date)"
echo "========================================="
