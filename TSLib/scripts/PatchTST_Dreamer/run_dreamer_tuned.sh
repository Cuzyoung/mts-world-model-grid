#!/bin/bash
# Dreamer Tuned: d_latent=128 (reduced from 256 to combat overfitting)
# Setting A encoder (seq=96, e=1) - best baseline setting
# Key change: d_latent 256→128, reduces head params from ~2.5M to ~870K
set -e
cd /home/aiscuser/workspace-gzy/mts-world-model-grid/TSLib
export CUDA_VISIBLE_DEVICES=3

SEQ_LEN=96
E_LAYERS=1
D_MODEL=128
N_HEADS=16
D_FF=256
DROPOUT=0.2
LR=0.0001
BATCH_SIZE=128
EPOCHS=100
PATIENCE=10
D_LATENT=128
SLOW_INTERVAL=2

echo "========================================="
echo "  Dreamer Tuned: d_latent=128 (GPU 3)"
echo "  Start: $(date)"
echo "========================================="

for dataset in ETTh1 ETTh2 ETTm1; do
    if [ "$dataset" = "ETTm1" ] || [ "$dataset" = "ETTm2" ]; then
        DATA_NAME=$dataset
    else
        DATA_NAME=$dataset
    fi
    for pred_len in 96 192 336 720; do
        echo ""
        echo "[$(date '+%H:%M:%S')] Dreamer_tuned ${dataset} seq=${SEQ_LEN} pred=${pred_len}"
        python -u run.py \
            --task_name long_term_forecast --is_training 1 \
            --root_path ./dataset/ETT-small/ --data_path ${dataset}.csv \
            --model_id ${dataset}_${SEQ_LEN}_${pred_len}_tuned --model PatchTST_Dreamer --data ${dataset} \
            --features M --seq_len $SEQ_LEN --label_len 48 --pred_len $pred_len \
            --e_layers $E_LAYERS --d_layers 1 --factor 3 \
            --enc_in 7 --dec_in 7 --c_out 7 \
            --d_model $D_MODEL --d_ff $D_FF --n_heads $N_HEADS \
            --d_latent $D_LATENT --slow_interval $SLOW_INTERVAL --head_variant dreamer \
            --dropout $DROPOUT --learning_rate $LR \
            --batch_size $BATCH_SIZE --des 'Tuned128' --itr 1 \
            --train_epochs $EPOCHS --patience $PATIENCE
    done
done

echo ""
echo "========================================="
echo "  Dreamer Tuned done!"
echo "  End: $(date)"
echo "========================================="
