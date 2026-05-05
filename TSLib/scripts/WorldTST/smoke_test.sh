#!/bin/bash
# WorldTST: Smoke test — 3 epochs on ETTh1 pred=96 for both models
set -e
cd /home/azureuser/workspace-gzy/mts-world-model-grid/TSLib
export CUDA_VISIBLE_DEVICES=0

COMMON="--task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 --features M --seq_len 96 --label_len 48 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --dropout 0.2 --learning_rate 0.0001 --train_epochs 3 --itr 1"

echo "===== Smoke Test: WorldTST-GRU ====="
python -u run.py $COMMON \
    --model_id ETTh1_smoke_GRU --model WorldTST_GRU \
    --pred_len 96 --e_layers 1 --n_heads 2 \
    --slow_interval 2 \
    --clip_grad 1.0 --lradj type3 \
    --batch_size 32 --patience 20 --des 'Smoke'

echo ""
echo "===== Smoke Test: WorldTST-Causal ====="
python -u run.py $COMMON \
    --model_id ETTh1_smoke_Causal --model WorldTST_Causal \
    --pred_len 96 --e_layers 1 --n_heads 2 \
    --clip_grad 1.0 --lradj type3 \
    --batch_size 32 --patience 20 --des 'Smoke'

echo ""
echo "===== Smoke Test Complete ====="
