import json

notebook_path = "02_Temporal_Modeling_and_Prediction.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_cells = [nb["cells"][0]] if len(nb["cells"]) > 0 and nb["cells"][0]["cell_type"] == "markdown" else []

new_code_cells = [
    """import os
import torch
from torch.utils.data import DataLoader
from dataset import TrueTimeSeriesSARDataset, get_train_val_test_splits
from models import TemporalTransformerPredictor
from train import train_temporal_model
from utils import set_seed, load_config

# 1. Setup & Configuration
config = load_config('config.json')
set_seed(config['training']['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
""",
    """# 2. Data Loading (Temporal Sequences)
s1_dir = config['data']['s1_dir']
label_dir = config['data']['label_dir']

# In a true temporal dataset, we expect sequences organized by location.
# As a fallback for Sen1Floods11 single snapshots, we will simulate the grouping 
# just to test the pipeline, but the TrueTimeSeriesSARDataset class is built for real sequences.

s1_files = sorted([f for f in os.listdir(s1_dir) if f.endswith('.tif')]) if os.path.exists(s1_dir) else []
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.tif')]) if os.path.exists(label_dir) else []

# Creating dummy sequences of length N for the single files available to test the pipeline
seq_length = config['data']['seq_length']
sequences = []
if len(s1_files) >= seq_length:
    for i in range(len(s1_files) - seq_length + 1):
        s1_seq = [os.path.join(s1_dir, s1_files[i+j]) for j in range(seq_length)]
        label_path = os.path.join(label_dir, label_files[i+seq_length-1])
        sequences.append((s1_seq, label_path))

if len(sequences) > 0:
    train_idx, val_idx, test_idx = get_train_val_test_splits(len(sequences), config['data']['split_ratio'])
    
    train_seqs = [sequences[i] for i in train_idx]
    val_seqs = [sequences[i] for i in val_idx]
    
    train_dataset = TrueTimeSeriesSARDataset(train_seqs, seq_length=seq_length, target_size=tuple(config['data']['target_size']))
    val_dataset = TrueTimeSeriesSARDataset(val_seqs, seq_length=seq_length, target_size=tuple(config['data']['target_size']))
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    
    print(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation sequences.")
else:
    print("Not enough data to create temporal sequences.")
""",
    """# 3. Model Definition & Training
if len(sequences) > 0:
    model = TemporalTransformerPredictor(
        seq_length=config['data']['seq_length'],
        embed_dim=config['model']['transformer_embed_dim'],
        num_heads=config['model']['transformer_heads'],
        hidden_dim=config['model']['transformer_hidden_dim'],
        num_layers=config['model']['transformer_layers'],
        dropout=config['model']['dropout_rate']
    ).to(device)
    
    # Load pretrained spatial encoder weights if available
    encoder_path = 'models/encoder.pth'
    if os.path.exists(encoder_path):
        print("Loading pretrained spatial encoder weights...")
        try:
            model.spatial_encoder.encoder.load_state_dict(torch.load(encoder_path, weights_only=True))
        except Exception as e:
            print(f"Could not load encoder weights: {e}")
    else:
        print("Pretrained encoder not found. Training from scratch.")
        
    train_temporal_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['temporal_epochs'],
        lr=config['training']['temporal_lr'],
        device=device
    )
"""
]

for src in new_code_cells:
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in src.split('\n')][:-1]
    })

nb["cells"] = new_cells

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Successfully refactored {notebook_path}")
