#!/bin/bash
# Directory containing the scripts
SCRIPTS_DIR="exp_scripts/zero-shot/CoT"

# Loop through each .sh file in the directory.
for script in "$SCRIPTS_DIR"/*.sh; do
    # Check that the file exists (in case there are no .sh files)
    if [ -f "$script" ]; then
        echo "Submitting job: $script with parameter 0 16"
        sbatch --time 10:0:0 --gres gpu:1 "$script" 0 16
    fi
done