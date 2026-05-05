#!/bin/bash
# GPU 0: ETTh1 — PatchTST baseline + Dreamer + Ablation
# Uses TSLib official hyperparameters per pred_len
set -e
cd /home/azureuser/workspace-gzy/mts-world-model-grid/TSLib
export CUDA_VISIBLE_DEVICES=0

echo "========================================="
echo "  GPU 0: ETTh1 Final Experiments"
echo "  Start: $(date)"
echo "========================================="

COMMON="--task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 --features M --seq_len 96 --label_len 48 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --dropout 0.2 --learning_rate 0.0001 --train_epochs 100 --itr 1"

# ===== PatchTST Baseline (official per-horizon hyperparams) =====
echo "===== PatchTST Baseline ====="

for pred_len in 96 192 336 720; do
    case $pred_len in
        96)  NH=2  ;;
        192) NH=8  ;;
        336) NH=8  ;;
        720) NH=16 ;;
    esac
    echo "[$(date '+%H:%M:%S')] PatchTST ETTh1 pred=$pred_len n_heads=$NH e_layers=1"
    python -u run.py $COMMON \
        --model_id ETTh1_final_${pred_len} --model PatchTST \
        --pred_len $pred_len --e_layers 1 --n_heads $NH \
        --batch_size 32 --patience 10 --des 'Final'
done

# ===== PatchTST-Dreamer (same encoder params, Dreamer head, tuned training) =====
echo "===== PatchTST-Dreamer ====="

for pred_len in 96 192 336 720; do
    case $pred_len in
        96)  NH=2  ;;
        192) NH=8  ;;
        336) NH=8  ;;
        720) NH=16 ;;
    esac
    echo "[$(date '+%H:%M:%S')] Dreamer ETTh1 pred=$pred_len n_heads=$NH e_layers=1"
    python -u run.py $COMMON \
        --model_id ETTh1_final_${pred_len} --model PatchTST_Dreamer \
        --pred_len $pred_len --e_layers 1 --n_heads $NH \
        --d_latent 256 --slow_interval 2 --head_variant dreamer \
        --clip_grad 1.0 --lradj type3 \
        --batch_size 32 --patience 20 --des 'Final'
done

# ===== Ablation: Single-Scale =====
echo "===== Ablation: Single-Scale ====="

for pred_len in 96 192 336 720; do
    case $pred_len in
        96)  NH=2  ;;
        192) NH=8  ;;
        336) NH=8  ;;
        720) NH=16 ;;
    esac
    echo "[$(date '+%H:%M:%S')] SingleScale ETTh1 pred=$pred_len"
    python -u run.py $COMMON \
        --model_id ETTh1_final_${pred_len}_single --model PatchTST_Dreamer \
        --pred_len $pred_len --e_layers 1 --n_heads $NH \
        --d_latent 256 --slow_interval 2 --head_variant single_scale \
        --clip_grad 1.0 --lradj type3 \
        --batch_size 32 --patience 20 --des 'Final'
done

# ===== Ablation: Hybrid =====
echo "===== Ablation: Hybrid ====="

for pred_len in 96 192 336 720; do
    case $pred_len in
        96)  NH=2  ;;
        192) NH=8  ;;
        336) NH=8  ;;
        720) NH=16 ;;
    esac
    echo "[$(date '+%H:%M:%S')] Hybrid ETTh1 pred=$pred_len"
    python -u run.py $COMMON \
        --model_id ETTh1_final_${pred_len}_hybrid --model PatchTST_Dreamer \
        --pred_len $pred_len --e_layers 1 --n_heads $NH \
        --d_latent 256 --slow_interval 2 --head_variant hybrid \
        --clip_grad 1.0 --lradj type3 \
        --batch_size 32 --patience 20 --des 'Final'
done

echo ""
echo "========================================="
echo "  GPU 0: ETTh1 ALL DONE"
echo "  End: $(date)"
echo "========================================="
