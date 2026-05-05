#!/bin/bash
# WorldTST Ablation Study on ETTh1 (all 4 horizons)
# GPU 1 (free since ETTh2 finished)
# Ablation variants for both GRU and Causal models

set -e
cd /home/azureuser/workspace-gzy/mts-world-model-grid/TSLib
export CUDA_VISIBLE_DEVICES=1

echo "========================================="
echo "  WorldTST Ablation Study: ETTh1"
echo "  Start: $(date)"
echo "========================================="

COMMON="--task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 --features M --seq_len 96 --label_len 48 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --dropout 0.2 --learning_rate 0.0001 --train_epochs 100 --itr 1 --clip_grad 1.0 --lradj type3 --batch_size 32 --patience 20"

# ===== GRU Ablation: NoCross (remove CrossAttention) =====
echo ""
echo "===== GRU-NoCross (no observation correction) ====="
for pred_len in 96 192 336 720; do
    case $pred_len in
        96) NH=2;; 192) NH=8;; 336) NH=8;; 720) NH=16;;
    esac
    echo "[$(date '+%H:%M:%S')] GRU_NoCross ETTh1 pred=$pred_len"
    python -u run.py $COMMON \
        --model_id ETTh1_GRU_NoCross_${pred_len} --model WorldTST_GRU_NoCross \
        --pred_len $pred_len --e_layers 1 --n_heads $NH \
        --slow_interval 2 --des 'Ablation'
done

# ===== GRU Ablation: SingleScale (remove slow GRU + gating) =====
echo ""
echo "===== GRU-SingleScale (no multi-scale dynamics) ====="
for pred_len in 96 192 336 720; do
    case $pred_len in
        96) NH=2;; 192) NH=8;; 336) NH=8;; 720) NH=16;;
    esac
    echo "[$(date '+%H:%M:%S')] GRU_SingleScale ETTh1 pred=$pred_len"
    python -u run.py $COMMON \
        --model_id ETTh1_GRU_SingleScale_${pred_len} --model WorldTST_GRU_SingleScale \
        --pred_len $pred_len --e_layers 1 --n_heads $NH \
        --slow_interval 2 --des 'Ablation'
done

# ===== Causal Ablation: NoCross (remove CrossAttention) =====
echo ""
echo "===== Causal-NoCross (no observation correction) ====="
for pred_len in 96 192 336 720; do
    case $pred_len in
        96) NH=2;; 192) NH=8;; 336) NH=8;; 720) NH=16;;
    esac
    echo "[$(date '+%H:%M:%S')] Causal_NoCross ETTh1 pred=$pred_len"
    python -u run.py $COMMON \
        --model_id ETTh1_Causal_NoCross_${pred_len} --model WorldTST_Causal_NoCross \
        --pred_len $pred_len --e_layers 1 --n_heads $NH \
        --des 'Ablation'
done

echo ""
echo "========================================="
echo "  Ablation Study: ALL DONE"
echo "  End: $(date)"
echo "========================================="
