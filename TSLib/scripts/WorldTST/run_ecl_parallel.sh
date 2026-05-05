#!/bin/bash
# Electricity ALL remaining experiments in parallel
# Current status: PatchTST pred=96,192 done; pred=336 training on GPU3
# Remaining: PatchTST pred=720 + GRU×4 + Causal×4 = 9 experiments
#
# GPU allocation (sharing with gaoxin VLLM):
#   GPU 0: ~5.7GB free → our models (d_model=128) fit easily (~1GB)
#   GPU 2: ~3.6GB free → our models fit
#   GPU 3: PatchTST 336 still running (6.7GB), wait for it then run 720
#
# Strategy:
#   GPU 0: GRU pred=96,192,336,720 (serial on same GPU, each ~20min)
#   GPU 2: Causal pred=96,192,336,720 (serial on same GPU)
#   GPU 3: After PatchTST 336 finishes → PatchTST 720
#
# Usage: bash run_ecl_parallel.sh

set -e
cd /home/azureuser/workspace-gzy/mts-world-model-grid/TSLib

ECL_COMMON="--task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --data custom --features M --seq_len 96 --label_len 48 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --target MT_320 --itr 1"
OUR_COMMON="$ECL_COMMON --d_model 128 --d_ff 256 --dropout 0.2 --learning_rate 0.0001 --lradj type3 --patience 20 --clip_grad 1.0 --train_epochs 100"

echo "========================================="
echo "  Electricity PARALLEL Launch"
echo "  Start: $(date)"
echo "========================================="

# ===== GPU 0: GRU ×4 (serial within GPU) =====
(
    export CUDA_VISIBLE_DEVICES=0
    export TMPDIR=/tmp/gzy_ecl_gru
    mkdir -p $TMPDIR
    for pred_len in 96 192 336 720; do
        echo "[$(date '+%H:%M:%S')] GPU0: WorldTST_GRU ECL pred=$pred_len"
        python -u run.py $OUR_COMMON \
            --model_id ECL_GRU_${pred_len} --model WorldTST_GRU \
            --pred_len $pred_len --e_layers 2 --n_heads 4 --batch_size 16 \
            --slow_interval 2 --des 'ECL'
    done
    echo "[$(date '+%H:%M:%S')] GPU0: ALL GRU DONE"
    rm -rf $TMPDIR
) > logs/ecl_gru_parallel.log 2>&1 &
GRU_PID=$!
echo "GRU on GPU0: PID=$GRU_PID"

# ===== GPU 2: Causal ×4 (serial within GPU) =====
(
    export CUDA_VISIBLE_DEVICES=2
    export TMPDIR=/tmp/gzy_ecl_causal
    mkdir -p $TMPDIR
    for pred_len in 96 192 336 720; do
        echo "[$(date '+%H:%M:%S')] GPU2: WorldTST_Causal ECL pred=$pred_len"
        python -u run.py $OUR_COMMON \
            --model_id ECL_Causal_${pred_len} --model WorldTST_Causal \
            --pred_len $pred_len --e_layers 2 --n_heads 4 --batch_size 16 \
            --des 'ECL'
    done
    echo "[$(date '+%H:%M:%S')] GPU2: ALL Causal DONE"
    rm -rf $TMPDIR
) > logs/ecl_causal_parallel.log 2>&1 &
CAUSAL_PID=$!
echo "Causal on GPU2: PID=$CAUSAL_PID"

# ===== GPU 3: PatchTST pred=720 (wait for 336 to finish) =====
(
    export CUDA_VISIBLE_DEVICES=3
    export TMPDIR=/tmp/gzy_ecl_pt720
    mkdir -p $TMPDIR

    # Wait for PatchTST 336 to finish (check every 30s)
    echo "[$(date '+%H:%M:%S')] GPU3: Waiting for PatchTST 336 (PID 3423656) to finish..."
    while kill -0 3423656 2>/dev/null; do
        sleep 30
    done
    echo "[$(date '+%H:%M:%S')] GPU3: PatchTST 336 done, starting pred=720"

    python -u run.py $ECL_COMMON \
        --model_id ECL_baseline_720 --model PatchTST \
        --pred_len 720 --e_layers 2 --n_heads 8 \
        --d_model 512 --d_ff 2048 --dropout 0.1 \
        --learning_rate 0.0001 --lradj type1 --patience 3 \
        --batch_size 16 --train_epochs 100 \
        --des 'Baseline'

    echo "[$(date '+%H:%M:%S')] GPU3: PatchTST 720 DONE"
    rm -rf $TMPDIR
) > logs/ecl_patchtst720_parallel.log 2>&1 &
PT720_PID=$!
echo "PatchTST-720 on GPU3: PID=$PT720_PID"

echo ""
echo "========================================="
echo "  All launched! PIDs:"
echo "    GRU (GPU0):        $GRU_PID"
echo "    Causal (GPU2):     $CAUSAL_PID"
echo "    PatchTST720 (GPU3): $PT720_PID"
echo "========================================="
echo ""
echo "Monitor with:"
echo "  tail -f logs/ecl_gru_parallel.log"
echo "  tail -f logs/ecl_causal_parallel.log"
echo "  tail -f logs/ecl_patchtst720_parallel.log"
