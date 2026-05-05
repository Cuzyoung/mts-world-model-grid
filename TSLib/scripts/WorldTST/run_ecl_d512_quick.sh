#!/bin/bash
# Quick finish: d512 remaining horizons with patience=3 (same as PatchTST)
# GRU: 336, 720 on GPU 3
# Causal: 336, 720 on GPU 0
set -e
cd /home/azureuser/workspace-gzy/mts-world-model-grid/TSLib

ECL_COMMON="--task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --data custom --features M --seq_len 96 --label_len 48 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --target MT_320 --itr 1"
D512="$ECL_COMMON --d_model 512 --d_ff 2048 --dropout 0.2 --learning_rate 0.0001 --lradj type3 --patience 3 --clip_grad 1.0 --train_epochs 100 --e_layers 2 --n_heads 8 --batch_size 16"

echo "=== Quick d512 finish (patience=3) Start: $(date) ==="

# GPU 3: GRU 336+720
(
    export CUDA_VISIBLE_DEVICES=3
    export TMPDIR=/tmp/gzy_g512q
    mkdir -p $TMPDIR
    for pl in 336 720; do
        echo "[$(date '+%H:%M:%S')] GPU3: GRU d512 pred=$pl"
        python -u run.py $D512 \
            --model_id ECL_GRU_d512_${pl} --model WorldTST_GRU \
            --pred_len $pl --slow_interval 2 --des 'ECL_d512'
    done
    echo "[$(date '+%H:%M:%S')] GPU3: GRU d512 DONE"
    rm -rf $TMPDIR
) > logs/ecl_gru_d512_quick.log 2>&1 &
echo "GRU quick PID=$!"

# GPU 0: Causal 336+720
(
    export CUDA_VISIBLE_DEVICES=0
    export TMPDIR=/tmp/gzy_c512q
    mkdir -p $TMPDIR
    for pl in 336 720; do
        echo "[$(date '+%H:%M:%S')] GPU0: Causal d512 pred=$pl"
        python -u run.py $D512 \
            --model_id ECL_Causal_d512_${pl} --model WorldTST_Causal \
            --pred_len $pl --des 'ECL_d512'
    done
    echo "[$(date '+%H:%M:%S')] GPU0: Causal d512 DONE"
    rm -rf $TMPDIR
) > logs/ecl_causal_d512_quick.log 2>&1 &
echo "Causal quick PID=$!"

echo "=== Launched. patience=3 → each horizon ~15min ==="
