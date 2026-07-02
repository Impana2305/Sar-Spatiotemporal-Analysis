import json

notebook_path = "03_SpatioTemporal_Analysis_Outputs.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Keep the introduction markdown cell
new_cells = [nb["cells"][0]] if len(nb["cells"]) > 0 and nb["cells"][0]["cell_type"] == "markdown" else []

new_code_cells = [
    """import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from dataset import TrueTimeSeriesSARDataset
from models import TemporalTransformerPredictor
from utils import set_seed, load_config

# 1. Setup
config = load_config('config.json')
set_seed(config['training']['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Load Model
model = TemporalTransformerPredictor(
    seq_length=config['data']['seq_length'],
    embed_dim=config['model']['transformer_embed_dim'],
    num_heads=config['model']['transformer_heads'],
    hidden_dim=config['model']['transformer_hidden_dim'],
    num_layers=config['model']['transformer_layers'],
    dropout=config['model']['dropout_rate']
).to(device)

model_path = 'models/best_temporal_model.pth'
if os.path.exists(model_path):
    print("Loading best trained temporal model...")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
else:
    print("Model weights not found. Please train in Stage 2 first.")
""",
    """# 3. Generate Predictions on True Sequences
s1_dir = config['data']['s1_dir']
label_dir = config['data']['label_dir']
seq_length = config['data']['seq_length']

s1_files = sorted([f for f in os.listdir(s1_dir) if f.endswith('.tif')]) if os.path.exists(s1_dir) else []
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.tif')]) if os.path.exists(label_dir) else []

sequences = []
if len(s1_files) >= seq_length:
    for i in range(len(s1_files) - seq_length + 1):
        s1_seq = [os.path.join(s1_dir, s1_files[i+j]) for j in range(seq_length)]
        label_path = os.path.join(label_dir, label_files[i+seq_length-1])
        sequences.append((s1_seq, label_path))

if len(sequences) > 0:
    dataset = TrueTimeSeriesSARDataset(sequences, seq_length=seq_length, target_size=tuple(config['data']['target_size']))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Grab the first sequence for analysis
    seq_data, target = next(iter(loader))
    seq_data = seq_data.to(device)
    
    with torch.no_grad():
        logits = model(seq_data)
        predicted_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    vv = seq_data[0, -1, 0].cpu().numpy()
    vh = seq_data[0, -1, 1].cpu().numpy()
    rgb = np.stack([vv, vh, vv-vh], axis=-1)
    rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb) + 1e-8)
    plt.imshow(rgb)
    plt.title("Input SAR (t_n)")
    
    plt.subplot(1, 2, 2)
    cmap = ListedColormap(['gray', 'blue'])
    plt.imshow(predicted_mask, cmap=cmap)
    plt.title("Real Predicted Water Mask")
    plt.show()
else:
    print("No data available for prediction.")
""",
    """# 4. Spatio-Temporal Analysis & Change Detection
# Here we compare the predicted mask at t_n with the input SAR from t_{n-1} 
# In a real temporal sequence, we would run the model for t_{n-1} as well to get its mask.
if len(sequences) > 1:
    # Get sequence 1 and sequence 2 predictions
    seq1, _ = dataset[0]
    seq2, _ = dataset[1]
    
    with torch.no_grad():
        out1 = model(seq1.unsqueeze(0).to(device))
        mask1 = torch.argmax(out1, dim=1).squeeze(0).cpu().numpy()
        
        out2 = model(seq2.unsqueeze(0).to(device))
        mask2 = torch.argmax(out2, dim=1).squeeze(0).cpu().numpy()
        
    change_map = np.zeros_like(mask2, dtype=int)
    change_map[(mask1 == 1) & (mask2 == 1)] = 1  # Permanent
    change_map[(mask1 == 0) & (mask2 == 1)] = 2  # New Flood
    change_map[(mask1 == 1) & (mask2 == 0)] = -1 # Receded
    
    plt.figure(figsize=(8, 8))
    cmap_change = ListedColormap(['lightgreen', 'lightgray', 'blue', 'red'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap_change.N)
    
    im = plt.imshow(change_map, cmap=cmap_change, norm=norm)
    cbar = plt.colorbar(im, ticks=[-1, 0, 1, 2])
    cbar.ax.set_yticklabels(['Receded', 'No Water', 'Permanent Water', 'New Flood'])
    plt.title("Temporal Water Change Detection (Real Sequences)")
    plt.show()
    
    # Calculate real flood area over time
    pixel_area_sqm = 100
    area1 = np.sum(mask1) * pixel_area_sqm / 1e6
    area2 = np.sum(mask2) * pixel_area_sqm / 1e6
    
    print(f"Time Step 1 Flood Area: {area1:.2f} km^2")
    print(f"Time Step 2 Flood Area: {area2:.2f} km^2")
    print(f"Flood Spread Rate: {area2 - area1:.2f} km^2")
else:
    print("Need at least 2 sequences to perform change detection.")
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
