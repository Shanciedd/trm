"""
Visualization script for ARC-AGI-1 data and TRM model predictions
"""
from typing import Optional
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import hydra
from omegaconf import DictConfig

from utils_sae import (
    EvalConfig, EvalState, 
    create_dataloader, init_eval_state
)

# Global dtype configuration
DTYPE = torch.bfloat16

# ARC color palette (standard 10 colors used in ARC-AGI)
ARC_COLORS = [
    '#000000',  # 0: black
    '#0074D9',  # 1: blue
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#AAAAAA',  # 5: grey
    '#F012BE',  # 6: magenta/fuchsia
    '#FF851B',  # 7: orange
    '#7FDBFF',  # 8: light blue/sky
    '#870C25',  # 9: maroon/dark red
]

def decode_arc_grid(tokens: torch.Tensor, vocab_size: int, grid_height: int = 30, grid_width: int = 30) -> np.ndarray:
    """
    Decode tokenized ARC grid back to 2D grid
    
    Args:
        tokens: [seq_len] tensor of token IDs
        vocab_size: vocabulary size
        grid_height: height of ARC grid
        grid_width: width of ARC grid
    
    Returns:
        grid: [H, W] numpy array of color IDs (0-9)
    """
    # The tokenization scheme likely encodes each cell as a token
    # For ARC, we have 10 colors (0-9)
    # Assuming tokens are direct color values or need simple decoding
    
    # Convert to numpy
    tokens_np = tokens.cpu().numpy()
    
    # Filter out padding and special tokens
    # Assuming color tokens are in range [0, 9]
    valid_tokens = tokens_np[(tokens_np >= 0) & (tokens_np < 10)]
    
    # Reshape to grid (if we have enough tokens)
    expected_size = grid_height * grid_width
    if len(valid_tokens) < expected_size:
        # Pad with zeros if needed
        valid_tokens = np.pad(valid_tokens, (0, expected_size - len(valid_tokens)), constant_values=0)
    elif len(valid_tokens) > expected_size:
        # Truncate if too many
        valid_tokens = valid_tokens[:expected_size]
    
    grid = valid_tokens.reshape(grid_height, grid_width)
    return grid


def visualize_arc_grid(grid: np.ndarray, ax: plt.Axes, title: str = ""):
    """
    Visualize an ARC grid
    
    Args:
        grid: [H, W] numpy array of color IDs (0-9)
        ax: matplotlib axis to plot on
        title: title for the plot
    """
    H, W = grid.shape
    
    # Create colormap
    cmap = ListedColormap(ARC_COLORS)
    
    # Plot grid
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9, aspect='equal')
    
    # Add grid lines
    for i in range(H + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
    for j in range(W + 1):
        ax.axvline(j - 0.5, color='white', linewidth=0.5, alpha=0.3)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)  # Flip y-axis


def visualize_batch(
    batch: dict,
    predictions: Optional[torch.Tensor],
    metadata,
    save_path: str,
    num_samples: int = 4,
):
    """
    Visualize a batch of ARC examples
    
    Args:
        batch: dictionary with 'inputs' and 'labels'
        predictions: [B, seq_len] tensor of predicted tokens (optional)
        metadata: dataset metadata
        save_path: path to save the visualization
        num_samples: number of samples to visualize
    """
    inputs = batch['inputs']  # [B, seq_len]
    labels = batch['labels']  # [B, seq_len]
    B = inputs.shape[0]
    
    # Determine number of samples to visualize
    num_samples = min(num_samples, B)
    
    # Determine grid dimensions (assume square grids for now)
    # You may need to adjust this based on actual data format
    seq_len = inputs.shape[1]
    grid_size = int(np.sqrt(seq_len))
    grid_size = min(grid_size, 30)  # Cap at 30x30
    
    # Create figure
    if predictions is not None:
        # Show input, target, prediction
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        if num_samples == 1:
            axes = axes[np.newaxis, :]
    else:
        # Show input, target only
        fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
        if num_samples == 1:
            axes = axes[np.newaxis, :]
    
    for i in range(num_samples):
        # Decode input
        input_grid = decode_arc_grid(inputs[i], metadata.vocab_size, grid_size, grid_size)
        visualize_arc_grid(input_grid, axes[i, 0], f"Input {i+1}")
        
        # Decode target
        target_grid = decode_arc_grid(labels[i], metadata.vocab_size, grid_size, grid_size)
        visualize_arc_grid(target_grid, axes[i, 1], f"Target {i+1}")
        
        # Decode prediction if available
        if predictions is not None:
            pred_grid = decode_arc_grid(predictions[i], metadata.vocab_size, grid_size, grid_size)
            visualize_arc_grid(pred_grid, axes[i, 2], f"Prediction {i+1}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()


def run_visualization(
    config: EvalConfig,
    eval_state: EvalState,
    eval_loader: torch.utils.data.DataLoader,
    metadata,
    output_dir: str,
    num_batches: int = 5,
):
    """
    Run visualization on evaluation dataset
    
    Args:
        config: evaluation config
        eval_state: evaluation state with model
        eval_loader: dataloader
        metadata: dataset metadata
        output_dir: directory to save visualizations
        num_batches: number of batches to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    
    eval_state.model.eval()
    
    with torch.no_grad():
        for batch_idx, (set_name, batch, global_batch_size) in enumerate(eval_loader):
            if batch_idx >= num_batches:
                break
            
            print(f"\n{'='*60}")
            print(f"Processing Batch {batch_idx + 1}/{num_batches}: {set_name}")
            print(f"{'='*60}")
            
            # Move to device
            batch = {k: v.to(device='cuda', dtype=DTYPE if v.dtype.is_floating_point else v.dtype) 
                    for k, v in batch.items()}
            
            # Initialize carry
            with torch.device("cuda"):
                carry = eval_state.model.initial_carry(batch)
            
            # Run model inference
            max_steps = config.arch.halt_max_steps
            print(f"Running inference for up to {max_steps} steps...")
            
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = eval_state.model(
                    carry=carry, batch=batch, return_keys={'preds'}
                )
                inference_steps += 1
                
                if all_finish:
                    print(f"  Inference completed in {inference_steps} steps")
                    break
                
                if inference_steps >= max_steps:
                    print(f"  Reached max steps ({max_steps})")
                    break
            
            # Get predictions
            predictions = preds.get('preds') if preds is not None else None
            
            # Visualize (show first 4 samples in batch)
            save_path = os.path.join(output_dir, f"batch_{batch_idx+1:03d}_{set_name}.png")
            visualize_batch(
                batch={k: v.cpu() for k, v in batch.items()},
                predictions=predictions.cpu() if predictions is not None else None,
                metadata=metadata,
                save_path=save_path,
                num_samples=min(4, global_batch_size),
            )
            
            print(f"  Loss: {loss.item():.4f}")
            if metrics:
                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: {v.item():.4f}")
                    else:
                        print(f"  {k}: {v}")


@hydra.main(config_path="config", config_name="cfg_eval", version_base=None)
def launch(hydra_config: DictConfig):
    """Main visualization launch function"""
    # Load config
    config = EvalConfig(**hydra_config)  # type: ignore

    # Override to use test split with batch_size=1 for visualization
    config.split = "test"
    config.global_batch_size = 1  # Visualize one at a time
    
    # Seed RNGs
    torch.random.manual_seed(config.seed)

    # Dataset - use test split
    print("Loading evaluation dataset (test split)...")
    eval_loader, eval_metadata = create_dataloader(
        config, 
        split="test",
        test_set_mode=True,  # Important: test mode for sequential iteration
        epochs_per_iter=1, 
        global_batch_size=config.global_batch_size
    )
    
    print(f"Dataset metadata:")
    print(f"  Vocab size: {eval_metadata.vocab_size}")
    print(f"  Sequence length: {eval_metadata.seq_len}")
    print(f"  Total puzzles: {eval_metadata.total_puzzles}")
    print(f"  Sets: {eval_metadata.sets}")

    # Evaluation state (load model)
    print("\nInitializing model...")
    eval_state = init_eval_state(config, eval_metadata)

    # Create output directory
    output_dir = os.path.join(
        config.checkpoint_path.replace('ckpt/', 'results/') if config.checkpoint_path else 'results/',
        'visualizations'
    )
    
    # Run visualization
    print(f"\nStarting visualization...")
    print(f"Output directory: {output_dir}")
    run_visualization(
        config=config,
        eval_state=eval_state,
        eval_loader=eval_loader,
        metadata=eval_metadata,
        output_dir=output_dir,
        num_batches=10,  # Visualize 10 batches
    )
    
    print(f"\n{'='*60}")
    print(f"Visualization complete! Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    launch()

