
# Local device 

Env (Follow TRM)
```bash
git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git
conda create -n trm python=3.10 -y
conda activate trm
pip install uv
uv pip install -r requirments.txt
```

Dataset (ARC-AGI-1)
```bash 
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets evaluation \
  --test-set-name evaluation
```

Model (ARC-AGI-1)
```bash
mkdir ckpt
mkdir ckpt/arc_v1_public 

cd ckpt/arc_v1_public 
wget https://huggingface.co/arcprize/trm_arc_prize_verification/resolve/main/arc_v1_public/all_config.yaml?download=true    # config
wget https://huggingface.co/arcprize/trm_arc_prize_verification/resolve/main/arc_v1_public/losses.py?download=true          # loss
wget https://huggingface.co/arcprize/trm_arc_prize_verification/resolve/main/arc_v1_public/step_518071?download=true        # model 
wget https://huggingface.co/arcprize/trm_arc_prize_verification/resolve/main/arc_v1_public/trm.py?download=true             # trm 

mv all_config.yaml* all_config.yaml
mv losses.py* losses.py
mv step_518071* step_518071
mv trm.py* trm.py

cd ../..
```

Run
```bash 
python eval.py \
  --config-path=ckpt/arc_v1_public \
  --config-name=all_config \
  load_checkpoint=ckpt/arc_v1_public/step_518071 \
  data_paths="[data/arc1concept-aug-1000]" \
  data_paths_test="[data/arc1concept-aug-1000]" \
  global_batch_size=1 \
  checkpoint_path=ckpt/arc_v1_public
```

# Visualization

Visualize ARC-AGI-1 data and TRM model predictions:

```bash
python visualize_arc.py \
  --config-path=ckpt/arc_v1_public \
  --config-name=all_config \
  load_checkpoint=ckpt/arc_v1_public/step_518071 \
  data_paths="[data/arc1concept-aug-1000]" \
  global_batch_size=1 \
  checkpoint_path=ckpt/arc_v1_public
```

Or use the convenience script:
```bash
./run_visualization.sh ckpt/arc_v1_public/step_518071
```

The visualization will:
1. Load the evaluation dataset (test split)
2. Run the TRM model to generate predictions
3. Visualize input grids, target grids, and predicted grids side-by-side
4. Save visualizations to `{checkpoint_path}/results/visualizations/`

Each batch will generate a PNG file showing up to 4 examples with:
- **Input**: The puzzle input grid
- **Target**: The ground truth solution
- **Prediction**: The model's predicted solution