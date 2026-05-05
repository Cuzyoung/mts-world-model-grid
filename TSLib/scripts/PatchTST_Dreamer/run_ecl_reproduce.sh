#!/bin/bash
# Reproduce PatchTST on ECL only (fix: --target MT_320)
set -e
cd /home/aiscuser/workspace-gzy/mts-world-model-grid/TSLib
export CUDA_VISIBLE_DEVICES=0

SEQ_LEN=336
E_LAYERS=3
D_MODEL=128
N_HEADS=16
D_FF=256
DROPOUT=0.2
LR=0.0001
EPOCHS=100
PATIENCE=10

echo "===== PatchTST ECL Reproduction ====="
for pred_len in 96 192 336 720; do
    echo ""
    echo "[$(date '+%H:%M:%S')] PatchTST ECL seq=${SEQ_LEN} pred=${pred_len}"
    python -u run.py \
        --task_name long_term_forecast --is_training 1 \
        --root_path ./dataset/electricity/ --data_path electricity.csv \
        --model_id ECL_${SEQ_LEN}_${pred_len} --model PatchTST --data custom \
        --features M --target MT_320 --seq_len $SEQ_LEN --label_len 48 --pred_len $pred_len \
        --e_layers $E_LAYERS --d_layers 1 --factor 3 \
        --enc_in 321 --dec_in 321 --c_out 321 \
        --d_model $D_MODEL --d_ff $D_FF --n_heads $N_HEADS \
        --dropout $DROPOUT --learning_rate $LR \
        --batch_size 32 --des 'Reproduce' --itr 1 \
        --train_epochs $EPOCHS --patience $PATIENCE
done
echo "ECL reproduction done!"
echo "Paper reproduction done"
