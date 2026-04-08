#!/bin/bash
# Run all experiments aligned with PatchTST (ICLR 2023) settings
# All logs saved to logs/<dataset>/<model>/pred<H>.log
# All results saved to results/<dataset>/<model>/pred<H>_{val,test}.json
# Predictions saved to results/<dataset>/<model>/pred<H>_{preds,trues}.npy

set -e
cd /home/aiscuser/workspace-gzy/mts-world-model-grid

PRED_LENS="96 192 336 720"
ETT_DATASETS="etth1 etth2 ettm1"
MODELS="mts_wm lstm dlinear informer"
ABLATION_MODELS="single_scale no_slow no_fast no_direct"

echo "========================================="
echo "  MTS-WM Full Benchmark"
echo "  Settings: seq_len=336, aligned with PatchTST"
echo "  Start time: $(date)"
echo "========================================="

# --- Main experiments: ETT datasets ---
for dataset in $ETT_DATASETS; do
    for model in $MODELS; do
        for pred_len in $PRED_LENS; do
            echo ""
            echo "[$(date '+%H:%M:%S')] --- $model on $dataset, pred_len=$pred_len ---"
            python src/train.py --config "configs/${dataset}.yaml" --model "$model" --pred_len "$pred_len"
        done
    done
done

# --- Main experiments: ECL ---
for model in $MODELS; do
    for pred_len in $PRED_LENS; do
        echo ""
        echo "[$(date '+%H:%M:%S')] --- $model on ECL, pred_len=$pred_len ---"
        python src/train.py --config configs/ecl.yaml --model "$model" --pred_len "$pred_len"
    done
done

# --- Ablation studies on ETTh1 ---
echo ""
echo "========================================="
echo "  Ablation Studies (ETTh1)"
echo "========================================="
for model in $ABLATION_MODELS; do
    for pred_len in $PRED_LENS; do
        echo ""
        echo "[$(date '+%H:%M:%S')] --- Ablation: $model, pred_len=$pred_len ---"
        python src/train.py --config configs/etth1.yaml --model "$model" --pred_len "$pred_len"
    done
done

echo ""
echo "========================================="
echo "  All experiments done!"
echo "  End time: $(date)"
echo "  Logs: logs/"
echo "  Results: results/"
echo "========================================="
