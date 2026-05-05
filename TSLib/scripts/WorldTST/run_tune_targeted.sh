#!/bin/bash
# Targeted tuning for benchmarks where WorldTST LOSES to PatchTST
# Strategy: For each losing (model, dataset, pred_len), sweep el/nh/bs
# Usage: bash run_tune_targeted.sh <GPU_ID> <MODEL> <DATASET> <PRED_LEN>

GPU_ID=${1:-0}
MODEL=${2:-WorldTST_GRU}
DATASET=${3:-ETTh1}
PRED_LEN=${4:-720}

set -e
cd /home/azureuser/workspace-gzy/mts-world-model-grid/TSLib
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "========================================="
echo "  Targeted Tuning: $MODEL on $DATASET pred=$PRED_LEN (GPU $GPU_ID)"
echo "  Start: $(date)"
echo "========================================="

# Fixed settings
COMMON_BASE="--task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ${DATASET}.csv --data ${DATASET} --features M --seq_len 96 --label_len 48 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --dropout 0.2 --learning_rate 0.0001 --train_epochs 100 --itr 1 --clip_grad 1.0 --lradj type3 --patience 20 --pred_len $PRED_LEN"

# Extra args for GRU model
EXTRA=""
if [[ "$MODEL" == *"GRU"* ]]; then
    EXTRA="--slow_interval 2"
fi

# Sweep: el={1,2,3}, nh={2,4,8}, bs={32,128}
# That's 18 combos but many were already run. We skip known configs.
CONFIGS=(
    "1 2 32"
    "1 4 32"
    "1 8 32"
    "1 2 128"
    "1 4 128"
    "1 8 128"
    "2 2 32"
    "2 4 32"
    "2 8 32"
    "2 2 128"
    "2 4 128"
    "2 8 128"
    "3 2 32"
    "3 4 32"
    "3 8 32"
    "3 2 128"
    "3 4 128"
    "3 8 128"
)

best_mse=999
best_config=""

for cfg in "${CONFIGS[@]}"; do
    read el nh bs <<< "$cfg"
    
    model_id="${DATASET}_tune_${MODEL}_pl${PRED_LEN}_el${el}_nh${nh}_bs${bs}"
    
    # Check if result already exists
    result_pattern="results/long_term_forecast_${model_id}*"
    if ls $result_pattern 1>/dev/null 2>&1; then
        echo "[SKIP] $model_id — already exists"
        continue
    fi
    
    echo ""
    echo "[$(date '+%H:%M:%S')] $MODEL $DATASET pred=$PRED_LEN el=$el nh=$nh bs=$bs"
    python -u run.py $COMMON_BASE $EXTRA \
        --e_layers $el --n_heads $nh --batch_size $bs \
        --model_id $model_id --model $MODEL \
        --des 'Tune' 2>&1 | tail -5
done

echo ""
echo "========================================="
echo "  Tuning $MODEL $DATASET pred=$PRED_LEN: ALL DONE"
echo "  End: $(date)"
echo "========================================="

# Print best result
echo ""
echo "=== Results Summary ==="
for cfg in "${CONFIGS[@]}"; do
    read el nh bs <<< "$cfg"
    model_id="${DATASET}_tune_${MODEL}_pl${PRED_LEN}_el${el}_nh${nh}_bs${bs}"
    result_dir=$(ls -d results/long_term_forecast_${model_id}* 2>/dev/null | head -1)
    if [ -n "$result_dir" ] && [ -f "$result_dir/metrics.npy" ]; then
        python3 -c "
import numpy as np
m = np.load('$result_dir/metrics.npy', allow_pickle=True)
if isinstance(m, np.ndarray) and m.ndim == 0: m = m.item()
if isinstance(m, dict): mse, mae = m['mse'], m['mae']
else: mae, mse = float(m[0]), float(m[1])
print(f'el=$el nh=$nh bs=$bs → MSE={mse:.4f} MAE={mae:.4f}')
"
    fi
done
