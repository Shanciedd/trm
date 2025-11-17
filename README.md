
# Env (Follow TRM)
```
git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git

conda create -n trm python=3.10 -y
conda activate trm
pip install uv
uv pip install -r requirments.txt
```

# ARC-AGI-1, dataset
```bash 
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation
```

# ARC-AGI-1, model 
```bash
mkdir ckpt
mkdir ckpt/arc_v1_public 

cd ckpt/arc_v1_public 
wget https://huggingface.co/arcprize/trm_arc_prize_verification/resolve/main/arc_v1_public/all_config.yaml?download=true    # config
wget https://huggingface.co/arcprize/trm_arc_prize_verification/resolve/main/arc_v1_public/losses.py?download=true          # loss
wget https://huggingface.co/arcprize/trm_arc_prize_verification/resolve/main/arc_v1_public/step_518071?download=true        # model 
wget https://huggingface.co/arcprize/trm_arc_prize_verification/resolve/main/arc_v1_public/trm.py?download=true             # trm 
```

# 