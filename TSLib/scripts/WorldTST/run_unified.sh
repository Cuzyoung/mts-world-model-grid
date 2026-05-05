#!/bin/bash
# WorldTST Unified: Per-dataset ONE set of hyperparameters, shared across all horizons
# This is the FINAL paper version — no per-horizon tuning
#
# Usage: bash run_unified.sh <GPU_ID> <DATASET> <MODEL>
# Example: bash run_unified.sh 0 ETTh1 WorldTST_GRU
#          bash run_unified.sh 1 ETTh2 WorldTST_Causal

GPU_ID=${1:-0}
DATASET=${2:-ETTh1}
MODEL=${3:-WorldTST_GRU}

set -e
cd /home/azureuser/workspace-gzy/mts-world-model-grid/TSLib
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Private TMPDIR to avoid /tmp pressure (short path for AF_UNIX 108-char limit)
export TMPDIR=/tmp/gzy_uni_${GPU_ID}
mkdir -p $TMPDIR

echo "========================================="
echo "  Unified: $MODEL on $DATASET (GPU $GPU_ID)"
echo "  TMPDIR=$TMPDIR"
echo "  Start: $(date)"
echo "========================================="

# ============================================================
# Unified per-dataset hyperparameters
# Rule: ONE config per dataset, ALL horizons share it
# ============================================================
case $DATASET in
    ETTh1)
        # ETTh1: shallow encoder, moderate heads
        EL=1; NH=4; BS=32; DO=0.2; DFF=256; SI=2
        ;;
    ETTh2)
        # ETTh2: deeper encoder (matches baseline), fixed heads
        EL=3; NH=4; BS=32; DO=0.2; DFF=256; SI=2
        ;;
    ETTm1)
        # ETTm1: 2-layer balanced
        EL=2; NH=4; BS=128; DO=0.2; DFF=256; SI=2
        ;;
    ETTm2)
        # ETTm2: deeper encoder, moderate heads
        EL=3; NH=4; BS=128; DO=0.2; DFF=256; SI=2
        ;;
esac

COMMON="--task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ${DATASET}.csv --data ${DATASET} --features M --seq_len 96 --label_len 48 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff $DFF --dropout $DO --learning_rate 0.0001 --train_epochs 100 --itr 1 --clip_grad 1.0 --lradj type3 --patience 20 --e_layers $EL --n_heads $NH --batch_size $BS"

# Extra args for GRU model
EXTRA=""
if [[ "$MODEL" == *"GRU"* ]]; then
    EXTRA="--slow_interval $SI"
fi

echo "Config: el=$EL nh=$NH bs=$BS do=$DO dff=$DFF si=$SI"
echo ""

for pred_len in 96 192 336 720; do
    echo "[$(date '+%H:%M:%S')] $MODEL $DATASET pred=$pred_len (unified: el=$EL nh=$NH)"
    python -u run.py $COMMON $EXTRA \
        --model_id ${DATASET}_unified_${pred_len} --model $MODEL \
        --pred_len $pred_len --des 'Unified'
done

echo ""
echo "========================================="
echo "  Unified $MODEL $DATASET: ALL DONE"
echo "  End: $(date)"
echo "========================================="

# Cleanup tmp
rm -rf $TMPDIR
