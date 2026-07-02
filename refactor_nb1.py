import json

notebook_path = "01_Data_and_Feature_Extraction.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Keep the first markdown cell (Introduction)
new_cells = [nb["cells"][0]] if len(nb["cells"]) > 0 and nb["cells"][0]["cell_type"] == "markdown" else []

new_code_cells = [
    """import os
import torch
from torch.utils.data import DataLoader
from dataset import Sen1FloodsDataset, get_train_val_test_splits
from models import ConvAutoencoder
from train import train_autoencoder
from utils import set_seed, load_config

# 1. Setup & Configuration
config = load_config('config.json')
set_seed(config['training']['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
""",
    """# 2. Data Loading
import rasterio

s1_dir = config['data']['s1_dir']
label_dir = config['data']['label_dir']

# Filter only valid TIF files
if os.path.exists(s1_dir) and os.path.exists(label_dir):
    s1_files = sorted([f for f in os.listdir(s1_dir) if f.endswith('.tif')])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.tif')])
    file_pairs = list(zip([os.path.join(s1_dir, f) for f in s1_files], 
                          [os.path.join(label_dir, f) for f in label_files]))
else:
    file_pairs = []
    print("Warning: Data directories not found.")

if len(file_pairs) > 0:
    # 70/15/15 Split
    train_idx, val_idx, test_idx = get_train_val_test_splits(len(file_pairs), config['data']['split_ratio'])
    
    train_pairs = [file_pairs[i] for i in train_idx]
    val_pairs = [file_pairs[i] for i in val_idx]
    
    train_dataset = Sen1FloodsDataset(train_pairs, target_size=tuple(config['data']['target_size']))
    val_dataset = Sen1FloodsDataset(val_pairs, target_size=tuple(config['data']['target_size']))
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    
    print(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation samples.")
else:
    print("No data pairs available. Please run data download.")
""",
    """# 3. Model Definition & Training
if len(file_pairs) > 0:
    model = ConvAutoencoder(dropout_rate=config['model']['dropout_rate']).to(device)
    
    # Execute full training loop with validation and early stopping
    train_autoencoder(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        num_epochs=config['training']['autoencoder_epochs'], 
        lr=config['training']['autoencoder_lr'],
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
