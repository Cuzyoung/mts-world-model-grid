#!/bin/bash
# Run remaining experiments (Step 1 ECL + Steps 2-4)
# Step 1 ETT baselines already completed

set -e
cd /home/aiscuser/workspace-gzy/mts-world-model-grid/TSLib

export CUDA_VISIBLE_DEVICES=0

echo "========================================="
echo "  PatchTST-Dreamer Remaining Experiments"
echo "  Start: $(date)"
echo "========================================="

# ============================================
# Step 1 remaining: PatchTST on ECL
# ============================================
echo ""
echo "===== Step 1 (remaining): PatchTST ECL ====="

for pred_len in 96 192 336 720; do
    echo "[$(date '+%H:%M:%S')] PatchTST ECL pred=$pred_len"
    python -u run.py \
        --task_name long_term_forecast --is_training 1 \
        --root_path ./dataset/electricity/ --data_path electricity.csv \
        --model_id ECL_96_${pred_len} --model PatchTST --data custom \
        --features M --seq_len 96 --label_len 48 --pred_len $pred_len \
        --e_layers 1 --d_layers 1 --factor 3 \
        --enc_in 321 --dec_in 321 --c_out 321 \
        --d_model 128 --d_ff 256 --n_heads 16 \
        --dropout 0.2 --batch_size 16 --des 'Exp' --itr 1 \
        --train_epochs 100 --patience 5
done


# ============================================
# Step 2: PatchTST-Dreamer (ENHANCED: d_latent=256, cross-attention, deep decoder)
# ============================================
echo ""
echo "===== Step 2: Training PatchTST-Dreamer (Enhanced) ====="

for pred_len in 96 192 336 720; do
    echo "[$(date '+%H:%M:%S')] PatchTST_Dreamer ETTh1 pred=$pred_len"
    python -u run.py \
        --task_name long_term_forecast --is_training 1 \
        --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
        --model_id ETTh1_96_${pred_len} --model PatchTST_Dreamer --data ETTh1 \
        --features M --seq_len 96 --label_len 48 --pred_len $pred_len \
        --e_layers 1 --d_layers 1 --factor 3 \
        --enc_in 7 --dec_in 7 --c_out 7 \
        --d_model 128 --d_ff 256 --n_heads 16 \
        --d_latent 256 --slow_interval 2 --head_variant dreamer \
        --dropout 0.2 --des 'Exp' --itr 1 \
        --train_epochs 100 --patience 5

    echo "[$(date '+%H:%M:%S')] PatchTST_Dreamer ETTh2 pred=$pred_len"
    python -u run.py \
        --task_name long_term_forecast --is_training 1 \
        --root_path ./dataset/ETT-small/ --data_path ETTh2.csv \
        --model_id ETTh2_96_${pred_len} --model PatchTST_Dreamer --data ETTh2 \
        --features M --seq_len 96 --label_len 48 --pred_len $pred_len \
        --e_layers 1 --d_layers 1 --factor 3 \
        --enc_in 7 --dec_in 7 --c_out 7 \
        --d_model 128 --d_ff 256 --n_heads 16 \
        --d_latent 256 --slow_interval 2 --head_variant dreamer \
        --dropout 0.2 --des 'Exp' --itr 1 \
        --train_epochs 100 --patience 5

    echo "[$(date '+%H:%M:%S')] PatchTST_Dreamer ETTm1 pred=$pred_len"
    python -u run.py \
        --task_name long_term_forecast --is_training 1 \
        --root_path ./dataset/ETT-small/ --data_path ETTm1.csv \
        --model_id ETTm1_96_${pred_len} --model PatchTST_Dreamer --data ETTm1 \
        --features M --seq_len 96 --label_len 48 --pred_len $pred_len \
        --e_layers 1 --d_layers 1 --factor 3 \
        --enc_in 7 --dec_in 7 --c_out 7 \
        --d_model 128 --d_ff 256 --n_heads 16 \
        --d_latent 256 --slow_interval 2 --head_variant dreamer \
        --dropout 0.2 --des 'Exp' --itr 1 \
        --train_epochs 100 --patience 5
done

for pred_len in 96 192 336 720; do
    echo "[$(date '+%H:%M:%S')] PatchTST_Dreamer ECL pred=$pred_len"
    python -u run.py \
        --task_name long_term_forecast --is_training 1 \
        --root_path ./dataset/electricity/ --data_path electricity.csv \
        --model_id ECL_96_${pred_len} --model PatchTST_Dreamer --data custom \
        --features M --seq_len 96 --label_len 48 --pred_len $pred_len \
        --e_layers 1 --d_layers 1 --factor 3 \
        --enc_in 321 --dec_in 321 --c_out 321 \
        --d_model 128 --d_ff 256 --n_heads 16 \
        --d_latent 256 --slow_interval 2 --head_variant dreamer \
        --dropout 0.2 --batch_size 16 --des 'Exp' --itr 1 \
        --train_epochs 100 --patience 5
done


# ============================================
# Step 3: Ablation studies on ETTh1
# ============================================
echo ""
echo "===== Step 3: Ablation Studies (ETTh1) ====="

for head_variant in single_scale flatten; do
    for pred_len in 96 192 336 720; do
        echo "[$(date '+%H:%M:%S')] Ablation: $head_variant ETTh1 pred=$pred_len"
        python -u run.py \
            --task_name long_term_forecast --is_training 1 \
            --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
            --model_id ETTh1_96_${pred_len}_${head_variant} --model PatchTST_Dreamer --data ETTh1 \
            --features M --seq_len 96 --label_len 48 --pred_len $pred_len \
            --e_layers 1 --d_layers 1 --factor 3 \
            --enc_in 7 --dec_in 7 --c_out 7 \
            --d_model 128 --d_ff 256 --n_heads 16 \
            --d_latent 256 --slow_interval 2 --head_variant $head_variant \
            --dropout 0.2 --des 'Exp' --itr 1 \
            --train_epochs 100 --patience 5
    done
done


# ============================================
# Step 4: Other baselines (DLinear, Informer)
# ============================================
echo ""
echo "===== Step 4: Baselines ====="

for model in DLinear Informer; do
    for pred_len in 96 192 336 720; do
        echo "[$(date '+%H:%M:%S')] $model ETTh1 pred=$pred_len"
        python -u run.py \
            --task_name long_term_forecast --is_training 1 \
            --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
            --model_id ETTh1_96_${pred_len} --model $model --data ETTh1 \
            --features M --seq_len 96 --label_len 48 --pred_len $pred_len \
            --e_layers 2 --d_layers 1 --factor 3 \
            --enc_in 7 --dec_in 7 --c_out 7 \
            --des 'Exp' --itr 1 \
            --train_epochs 100 --patience 5

        echo "[$(date '+%H:%M:%S')] $model ETTh2 pred=$pred_len"
        python -u run.py \
            --task_name long_term_forecast --is_training 1 \
            --root_path ./dataset/ETT-small/ --data_path ETTh2.csv \
            --model_id ETTh2_96_${pred_len} --model $model --data ETTh2 \
            --features M --seq_len 96 --label_len 48 --pred_len $pred_len \
            --e_layers 2 --d_layers 1 --factor 3 \
            --enc_in 7 --dec_in 7 --c_out 7 \
            --des 'Exp' --itr 1 \
            --train_epochs 100 --patience 5

        echo "[$(date '+%H:%M:%S')] $model ETTm1 pred=$pred_len"
        python -u run.py \
            --task_name long_term_forecast --is_training 1 \
            --root_path ./dataset/ETT-small/ --data_path ETTm1.csv \
            --model_id ETTm1_96_${pred_len} --model $model --data ETTm1 \
            --features M --seq_len 96 --label_len 48 --pred_len $pred_len \
            --e_layers 2 --d_layers 1 --factor 3 \
            --enc_in 7 --dec_in 7 --c_out 7 \
            --des 'Exp' --itr 1 \
            --train_epochs 100 --patience 5
    done
done


echo ""
echo "========================================="
echo "  All experiments done!"
echo "  End: $(date)"
echo "  Results in: ./results/"
echo "  Checkpoints in: ./checkpoints/"
echo "========================================="
