#!/bin/bash
# Run all experiments: all models x all datasets x all prediction horizons

set -e

PRED_LENS="96 192 336 720"
DATASETS="etth1 etth2 ettm1"
MODELS="mts_wm lstm dlinear informer"

# Ablation models (only run on ETTh1)
ABLATION_MODELS="single_scale no_slow no_fast"

echo "========================================="
echo "  MTS-WM Full Benchmark"
echo "========================================="

# Main experiments
for dataset in $DATASETS; do
    for model in $MODELS; do
        for pred_len in $PRED_LENS; do
            echo ""
            echo "--- $model on $dataset, pred_len=$pred_len ---"
            python src/train.py --config "configs/${dataset}.yaml" --model "$model" --pred_len "$pred_len"
        done
    done
done

# ECL (only key horizons due to size)
for model in $MODELS; do
    for pred_len in 96 336; do
        echo ""
        echo "--- $model on ECL, pred_len=$pred_len ---"
        python src/train.py --config configs/ecl.yaml --model "$model" --pred_len "$pred_len"
    done
done

# Ablation studies on ETTh1
echo ""
echo "========================================="
echo "  Ablation Studies (ETTh1)"
echo "========================================="
for model in $ABLATION_MODELS; do
    for pred_len in 96 336; do
        echo ""
        echo "--- Ablation: $model, pred_len=$pred_len ---"
        python src/train.py --config configs/etth1.yaml --model "$model" --pred_len "$pred_len"
    done
done

echo ""
echo "========================================="
echo "  All experiments done!"
echo "  Results saved in results/"
echo "========================================="
