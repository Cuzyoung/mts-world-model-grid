#!/bin/bash
# PatchTST-Dreamer v2: Full experiments on all 4 ETT datasets
# 4 GPUs in parallel: GPU0=ETTh1, GPU1=ETTh2, GPU2=ETTm1, GPU3=ETTm2
# This script runs on a SINGLE GPU — launch 4 instances with different GPU_ID and DATASET

GPU_ID=${1:-0}
DATASET=${2:-ETTh1}

set -e
cd /home/azureuser/workspace-gzy/mts-world-model-grid/TSLib
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "========================================="
echo "  Dreamer v2 Full: $DATASET on GPU $GPU_ID"
echo "  Start: $(date)"
echo "========================================="

# Per-dataset hyperparameters (matching TSLib official + v1 final)
case $DATASET in
    ETTh1)
        NH_96=2;  NH_192=8;  NH_336=8;  NH_720=16
        EL_96=1;  EL_192=1;  EL_336=1;  EL_720=1
        BS=32
        ;;
    ETTh2)
        NH_96=4;  NH_192=4;  NH_336=4;  NH_720=4
        EL_96=3;  EL_192=3;  EL_336=3;  EL_720=3
        BS=32
        ;;
    ETTm1)
        NH_96=2;  NH_192=2;  NH_336=4;  NH_720=4
        EL_96=1;  EL_192=3;  EL_336=1;  EL_720=3
        BS_96=32; BS_192=128; BS_336=128; BS_720=128
        ;;
    ETTm2)
        NH_96=16; NH_192=2;  NH_336=4;  NH_720=4
        EL_96=3;  EL_192=3;  EL_336=1;  EL_720=3
        BS_96=32; BS_192=128; BS_336=32;  BS_720=128
        ;;
esac

COMMON="--task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ${DATASET}.csv --data ${DATASET} --features M --seq_len 96 --label_len 48 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --dropout 0.2 --learning_rate 0.0001 --train_epochs 100 --itr 1"

for pred_len in 96 192 336 720; do
    # Get per-horizon hyperparams
    eval NH=\$NH_${pred_len}
    eval EL=\$EL_${pred_len}

    # Batch size: per-horizon if defined, else default
    eval THIS_BS=\$BS_${pred_len}
    if [ -z "$THIS_BS" ]; then
        THIS_BS=${BS:-32}
    fi

    echo ""
    echo "[$(date '+%H:%M:%S')] v2 ${DATASET} pred=$pred_len nh=$NH el=$EL bs=$THIS_BS"
    python -u run.py $COMMON \
        --model_id ${DATASET}_v2_${pred_len} --model PatchTST_Dreamer_v2 \
        --pred_len $pred_len --e_layers $EL --n_heads $NH \
        --d_latent 256 --slow_interval 2 \
        --clip_grad 1.0 --lradj type3 \
        --batch_size $THIS_BS --patience 20 --des 'V2'
done

echo ""
echo "========================================="
echo "  Dreamer v2 $DATASET: ALL DONE"
echo "  End: $(date)"
echo "========================================="
