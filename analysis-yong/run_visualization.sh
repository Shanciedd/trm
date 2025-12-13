#!/bin/bash

# Script to run ARC-AGI-1 visualization
# Run this from the base directory: /home/yonghyun/trm/
# Usage: ./analysis-yong/run_visualization.sh <checkpoint_path>

if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_path>"
    echo "Example: $0 ckpt/arc_v1_public/step_518071"
    exit 1
fi

CHECKPOINT_PATH=$1

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Change to base directory
cd "$BASE_DIR"

echo "Running ARC-AGI-1 visualization..."
echo "Base directory: $BASE_DIR"
echo "Checkpoint: $CHECKPOINT_PATH"
echo ""

# Run visualization script from analysis-yong directory
python analysis-yong/visualize_arc.py \
    load_checkpoint=$CHECKPOINT_PATH \
    checkpoint_path=$(dirname $CHECKPOINT_PATH) \
    global_batch_size=1 \
    split=test

echo ""
echo "Visualization complete!"
echo "Results saved to: $(dirname $CHECKPOINT_PATH)/results/visualizations/"

