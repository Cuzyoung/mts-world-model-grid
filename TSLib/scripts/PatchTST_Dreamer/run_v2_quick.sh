#!/bin/bash
# PatchTST-Dreamer v2: Quick validation on ETTh1 (4 horizons)
# Compare v2 against PatchTST baseline using same per-horizon hyperparams as v1 final
set -e
cd /home/azureuser/workspace-gzy/mts-world-model-grid/TSLib
export CUDA_VISIBLE_DEVICES=0

echo "========================================="
echo "  Dreamer v2: ETTh1 Quick Validation"
echo "  Start: $(date)"
echo "========================================="

COMMON="--task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 --features M --seq_len 96 --label_len 48 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --dropout 0.2 --learning_rate 0.0001 --train_epochs 100 --itr 1"

# ===== PatchTST-Dreamer v2 (Gated World Model) =====
echo "===== PatchTST-Dreamer v2 ====="

for pred_len in 96 192 336 720; do
    case $pred_len in
        96)  NH=2  ;;
        192) NH=8  ;;
        336) NH=8  ;;
        720) NH=16 ;;
    esac
    echo "[$(date '+%H:%M:%S')] Dreamer_v2 ETTh1 pred=$pred_len n_heads=$NH e_layers=1"
    python -u run.py $COMMON \
        --model_id ETTh1_v2_${pred_len} --model PatchTST_Dreamer_v2 \
        --pred_len $pred_len --e_layers 1 --n_heads $NH \
        --d_latent 256 --slow_interval 2 \
        --clip_grad 1.0 --lradj type3 \
        --batch_size 32 --patience 20 --des 'V2'
done

echo ""
echo "========================================="
echo "  Dreamer v2 ETTh1: ALL DONE"
echo "  End: $(date)"
echo "========================================="
