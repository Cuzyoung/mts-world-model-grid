#!/bin/bash
# Run ALL remaining experiments (9 total)
# GPU 0: ETTm1 Causal Unified (4 horizons)
# GPU 1: CausalNoCross ablation 336+720 (2 remaining)
# GPU 2: ETTm2 Causal Unified (4 horizons, restart from scratch)
#
# Usage: bash run_remaining_all.sh
# Or run each GPU separately in tmux

set -e
cd /home/azureuser/workspace-gzy/mts-world-model-grid/TSLib

echo "========================================="
echo "  Remaining Experiments Launcher"
echo "  Start: $(date)"
echo "========================================="

# ============ GPU 0: ETTm1 Causal Unified ============
echo "Launching ETTm1 Causal Unified on GPU 0..."
nohup bash scripts/WorldTST/run_unified.sh 0 ETTm1 WorldTST_Causal \
    > logs/uni_m1_causal.log 2>&1 &
PID0=$!
echo "  PID=$PID0, log=logs/uni_m1_causal.log"

# ============ GPU 1: CausalNoCross ablation 336+720 ============
echo "Launching CausalNoCross ablation (336,720) on GPU 1..."
nohup bash -c '
cd /home/azureuser/workspace-gzy/mts-world-model-grid/TSLib
export CUDA_VISIBLE_DEVICES=1

COMMON="--task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 --features M --seq_len 96 --label_len 48 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --dropout 0.2 --learning_rate 0.0001 --train_epochs 100 --itr 1 --clip_grad 1.0 --lradj type3 --batch_size 32 --patience 20"

echo "===== Causal-NoCross pred=336 ====="
python -u run.py $COMMON \
    --model_id ETTh1_Causal_NoCross_336 --model WorldTST_Causal_NoCross \
    --pred_len 336 --e_layers 1 --n_heads 8 --des Ablation

echo "===== Causal-NoCross pred=720 ====="
python -u run.py $COMMON \
    --model_id ETTh1_Causal_NoCross_720 --model WorldTST_Causal_NoCross \
    --pred_len 720 --e_layers 1 --n_heads 16 --des Ablation

echo "CausalNoCross ablation: ALL DONE ($(date))"
' > logs/ablation_remaining.log 2>&1 &
PID1=$!
echo "  PID=$PID1, log=logs/ablation_remaining.log"

# ============ GPU 2: ETTm2 Causal Unified ============
echo "Launching ETTm2 Causal Unified on GPU 2..."
nohup bash scripts/WorldTST/run_unified.sh 2 ETTm2 WorldTST_Causal \
    > logs/uni_m2_causal_restart.log 2>&1 &
PID2=$!
echo "  PID=$PID2, log=logs/uni_m2_causal_restart.log"

echo ""
echo "========================================="
echo "  All 3 GPU jobs launched!"
echo "  GPU 0: ETTm1 Causal x4   (PID $PID0)"
echo "  GPU 1: CausalNoCross x2   (PID $PID1)"
echo "  GPU 2: ETTm2 Causal x4   (PID $PID2)"
echo "  GPU 3: free"
echo "========================================="
echo "  Monitor: tail -f logs/uni_m1_causal.log"
echo "           tail -f logs/ablation_remaining.log"
echo "           tail -f logs/uni_m2_causal_restart.log"
