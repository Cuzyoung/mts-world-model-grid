#!/bin/bash
# Wait for reproduction to finish, then run Dreamer
LOG=/home/aiscuser/workspace-gzy/mts-world-model-grid/TSLib/logs/reproduce_paper.log

echo "Waiting for PatchTST reproduction to finish..."
while true; do
    if grep -q "Paper reproduction done" "$LOG" 2>/dev/null; then
        echo "Reproduction finished! Starting Dreamer..."
        break
    fi
    sleep 30
done

bash /home/aiscuser/workspace-gzy/mts-world-model-grid/TSLib/scripts/PatchTST_Dreamer/run_dreamer_paper_setting.sh 2>&1 | tee /home/aiscuser/workspace-gzy/mts-world-model-grid/TSLib/logs/dreamer_paper_setting.log
