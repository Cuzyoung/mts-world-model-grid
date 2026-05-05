#!/bin/bash
# PatchTST Baseline: Unified per-dataset hyperparameters (matching WorldTST experiments)
# This ensures FAIR comparison — baseline uses same structural hyperparams

GPU_ID=${1:-0}
DATASET=${2:-ETTh1}

set -e
cd /home/azureuser/workspace-gzy/mts-world-model-grid/TSLib
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "========================================="
echo "  PatchTST Baseline (Unified): $DATASET on GPU $GPU_ID"
echo "  Start: $(date)"
echo "========================================="

# Same unified per-dataset hyperparameters as WorldTST
case $DATASET in
    ETTh1) EL=1; NH=4; BS=32; DO=0.2; DFF=256;;
    ETTh2) EL=3; NH=4; BS=32; DO=0.2; DFF=256;;
    ETTm1) EL=2; NH=4; BS=128; DO=0.2; DFF=256;;
    ETTm2) EL=3; NH=4; BS=128; DO=0.2; DFF=256;;
esac

COMMON="--task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ${DATASET}.csv --data ${DATASET} --features M --seq_len 96 --label_len 48 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff $DFF --dropout $DO --learning_rate 0.0001 --train_epochs 100 --itr 1 --clip_grad 1.0 --lradj type3 --patience 20 --e_layers $EL --n_heads $NH --batch_size $BS"

echo "Config: el=$EL nh=$NH bs=$BS do=$DO dff=$DFF"

for pred_len in 96 192 336 720; do
    echo ""
    echo "[$(date '+%H:%M:%S')] PatchTST $DATASET pred=$pred_len (unified: el=$EL nh=$NH)"
    python -u run.py $COMMON \
        --model_id ${DATASET}_baseline_unified_${pred_len} --model PatchTST \
        --pred_len $pred_len --des 'BaseUnified'
done

echo ""
echo "========================================="
echo "  PatchTST Baseline $DATASET: ALL DONE"
echo "  End: $(date)"
echo "========================================="
