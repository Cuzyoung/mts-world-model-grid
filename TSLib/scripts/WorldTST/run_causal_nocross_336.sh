#!/bin/bash
# Rerun CausalNoCross pred=336 (crashed twice during training)
# Usage: bash run_causal_nocross_336.sh <GPU_ID>

GPU_ID=${1:-1}

set -e
cd /home/azureuser/workspace-gzy/mts-world-model-grid/TSLib
export CUDA_VISIBLE_DEVICES=$GPU_ID
export TMPDIR=/tmp/gzy_abl_gpu${GPU_ID}
mkdir -p $TMPDIR

echo "===== Causal-NoCross pred=336 (retry) ====="
echo "GPU=$GPU_ID  TMPDIR=$TMPDIR  Start: $(date)"

COMMON="--task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 --features M --seq_len 96 --label_len 48 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --dropout 0.2 --learning_rate 0.0001 --train_epochs 100 --itr 1 --clip_grad 1.0 --lradj type3 --batch_size 32 --patience 20"

python -u run.py $COMMON \
    --model_id ETTh1_Causal_NoCross_336 --model WorldTST_Causal_NoCross \
    --pred_len 336 --e_layers 1 --n_heads 8 --des Ablation

echo "===== DONE: $(date) ====="
rm -rf $TMPDIR
