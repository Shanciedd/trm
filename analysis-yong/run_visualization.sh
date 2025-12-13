if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_path>"
    echo "Example: $0 ckpt/best/model.pt"
    exit 1
fi

CHECKPOINT_PATH=$1

echo "Running ARC-AGI-1 visualization..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo ""

# Run visualization script
python visualize_arc.py \
    load_checkpoint=$CHECKPOINT_PATH \
    checkpoint_path=$(dirname $CHECKPOINT_PATH) \
    global_batch_size=1 \
    split=test

echo ""
echo "Visualization complete!"
echo "Results saved to: $(dirname $CHECKPOINT_PATH)/results/visualizations/"

