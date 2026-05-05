#!/bin/bash
# Launch ALL unified experiments across 4 GPUs
# Each GPU handles one dataset, runs GRU + Causal sequentially
# This produces the FINAL paper numbers

# Kill old experiments first if needed
# tmux kill-session -t world_gpu0 2>/dev/null; tmux kill-session -t world_gpu2 2>/dev/null; tmux kill-session -t world_gpu3 2>/dev/null

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Launching unified experiments..."

# GPU 0: ETTh1 (GRU + Causal)
tmux new-session -d -s uni_gpu0 "bash -c 'source activate wan_refactor 2>/dev/null; bash $SCRIPT_DIR/run_unified.sh 0 ETTh1 WorldTST_GRU 2>&1 | tee logs/uni_h1_gru.log; bash $SCRIPT_DIR/run_unified.sh 0 ETTh1 WorldTST_Causal 2>&1 | tee logs/uni_h1_causal.log'"

# GPU 1: ETTh2 — SKIP (already unified: el=3 nh=4, results in world_gpu1.log)
echo "GPU 1: ETTh2 SKIP (already unified in world_gpu1.log)"

# GPU 2: ETTm1 (GRU + Causal)
tmux new-session -d -s uni_gpu2 "bash -c 'source activate wan_refactor 2>/dev/null; bash $SCRIPT_DIR/run_unified.sh 2 ETTm1 WorldTST_GRU 2>&1 | tee logs/uni_m1_gru.log; bash $SCRIPT_DIR/run_unified.sh 2 ETTm1 WorldTST_Causal 2>&1 | tee logs/uni_m1_causal.log'"

# GPU 3: ETTm2 (GRU + Causal)
tmux new-session -d -s uni_gpu3 "bash -c 'source activate wan_refactor 2>/dev/null; bash $SCRIPT_DIR/run_unified.sh 3 ETTm2 WorldTST_GRU 2>&1 | tee logs/uni_m2_gru.log; bash $SCRIPT_DIR/run_unified.sh 3 ETTm2 WorldTST_Causal 2>&1 | tee logs/uni_m2_causal.log'"

echo "Launched: uni_gpu0 (ETTh1), uni_gpu2 (ETTm1), uni_gpu3 (ETTm2)"
echo "ETTh2 reuses existing results (already unified)"
echo "Check: tmux ls | grep uni"
