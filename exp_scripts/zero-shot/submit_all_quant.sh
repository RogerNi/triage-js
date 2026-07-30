#!/bin/bash
# Determine the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for batch_size in 32 64 128;
do
    for script in "$SCRIPT_DIR"/quant/*.sh; 
    do
        echo "Submitting $script with batch size $batch_size"
        sbatch --time 3:00:00 --gres gpu:1 "$script" 0 $batch_size
    done
done