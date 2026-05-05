#!/bin/bash
# Electricity d_model=512 experiments — fair comparison with PatchTST
# Same d_model/d_ff as PatchTST baseline, only replace self-attention
#
# GPU 3: GRU d512 ×4 horizons (~7GB, 16GB free)
# GPU 0: Causal d512 ×4 horizons (~7GB, 10GB free)
#
# Usage: bash run_ecl_d512.sh

set -e
cd /home/azureuser/workspace-gzy/mts-world-model-grid/TSLib

ECL_COMMON="--task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --data custom --features M --seq_len 96 --label_len 48 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --target MT_320 --itr 1"

# Use same d_model=512, d_ff=2048 as PatchTST, but keep our training recipe
# e_layers=2, n_heads=8 to match PatchTST architecture capacity
# patience=20 + lradj=type3 (our recipe, gives models more time to converge)
D512_COMMON="$ECL_COMMON --d_model 512 --d_ff 2048 --dropout 0.2 --learning_rate 0.0001 --lradj type3 --patience 20 --clip_grad 1.0 --train_epochs 100 --e_layers 2 --n_heads 8 --batch_size 16"

echo "========================================="
echo "  Electricity d512 PARALLEL Launch"
echo "  Start: $(date)"
echo "========================================="

# ===== GPU 3: GRU d512 ×4 =====
(
    export CUDA_VISIBLE_DEVICES=3
    export TMPDIR=/tmp/gzy_ecl_gru512
    mkdir -p $TMPDIR
    for pred_len in 96 192 336 720; do
        echo "[$(date '+%H:%M:%S')] GPU3: WorldTST_GRU d512 ECL pred=$pred_len"
        python -u run.py $D512_COMMON \
            --model_id ECL_GRU_d512_${pred_len} --model WorldTST_GRU \
            --pred_len $pred_len --slow_interval 2 --des 'ECL_d512'
    done
    echo "[$(date '+%H:%M:%S')] GPU3: ALL GRU d512 DONE"
    rm -rf $TMPDIR
) > logs/ecl_gru_d512.log 2>&1 &
GRU_PID=$!
echo "GRU d512 on GPU3: PID=$GRU_PID"

# ===== GPU 0: Causal d512 ×4 =====
(
    export CUDA_VISIBLE_DEVICES=0
    export TMPDIR=/tmp/gzy_ecl_cau512
    mkdir -p $TMPDIR
    for pred_len in 96 192 336 720; do
        echo "[$(date '+%H:%M:%S')] GPU0: WorldTST_Causal d512 ECL pred=$pred_len"
        python -u run.py $D512_COMMON \
            --model_id ECL_Causal_d512_${pred_len} --model WorldTST_Causal \
            --pred_len $pred_len --des 'ECL_d512'
    done
    echo "[$(date '+%H:%M:%S')] GPU0: ALL Causal d512 DONE"
    rm -rf $TMPDIR
) > logs/ecl_causal_d512.log 2>&1 &
CAUSAL_PID=$!
echo "Causal d512 on GPU0: PID=$CAUSAL_PID"

echo ""
echo "========================================="
echo "  PIDs: GRU=$GRU_PID  Causal=$CAUSAL_PID"
echo "========================================="
echo "Monitor:"
echo "  tail -f logs/ecl_gru_d512.log"
echo "  tail -f logs/ecl_causal_d512.log"
