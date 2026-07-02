import json

notebook_path = "02_Temporal_Modeling_and_Prediction.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "class TemporalTransformerPredictor(nn.Module):" in source and "embed_dim=128*16*16" in source:
            new_source = """class SpatialEncoder(nn.Module):
    \"\"\"The Encoder part of the Autoencoder from Stage 1.\"\"\"
    def __init__(self):
        super(SpatialEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.encoder(x) # Output shape: [B, 128, 16, 16]

class TemporalTransformerPredictor(nn.Module):
    def __init__(self, seq_length=3, spatial_dim=128*16*16, embed_dim=512, num_heads=8, hidden_dim=512, num_layers=2):
        super(TemporalTransformerPredictor, self).__init__()
        
        # 1. Spatial Feature Extractor (Encoder)
        self.spatial_encoder = SpatialEncoder()
        
        # Linear projection to reduce dimensionality
        self.feature_proj = nn.Linear(spatial_dim, embed_dim)
        
        # Positional Encoding for time steps
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, embed_dim))
        
        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Linear projection back to spatial dimensions
        self.feature_unproj = nn.Linear(embed_dim, spatial_dim)
        
        # 3. Prediction Head (Decoder for Segmentation)
        # We take the transformer output for the LAST time step and decode it back to 256x256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 32x32x64
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x64x32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x128x16
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1),   # 256x256x2 (2 classes: non-water, water)
        )

    def forward(self, x):
        # x shape: [B, SeqLen, C, H, W]
        B, S, C, H, W = x.shape
        
        # Reshape to process all spatial images through encoder
        x_reshaped = x.view(B * S, C, H, W)
        
        # Extract Spatial Features
        spatial_features = self.spatial_encoder(x_reshaped) # Shape: [B*S, 128, 16, 16]
        
        # Flatten spatial dimensions for Transformer
        sf_flat = spatial_features.view(B, S, -1) # Shape: [B, S, 128*16*16]
        
        # Project down
        sf_proj = self.feature_proj(sf_flat) # Shape: [B, S, embed_dim]
        
        # Add positional embedding
        sf_proj = sf_proj + self.pos_embedding
        
        # Temporal Modeling
        temporal_features = self.transformer(sf_proj) # Shape: [B, S, embed_dim]
        
        # Take the features of the last time step for prediction
        last_step_features = temporal_features[:, -1, :] # Shape: [B, embed_dim]
        
        # Project back to original spatial feature dimensions
        last_step_unproj = self.feature_unproj(last_step_features) # Shape: [B, 128*16*16]
        
        # Reshape back to spatial map for decoder
        last_step_spatial = last_step_unproj.view(B, 128, 16, 16)
        
        # Multi-output Prediction Head (Segmentation)
        logits = self.decoder(last_step_spatial) # Shape: [B, 2, 256, 256]
        return logits

model = TemporalTransformerPredictor(seq_length=3).to(device)

# Try loading pretrained encoder weights if available
encoder_path = 'models/encoder.pth'
if os.path.exists(encoder_path):
    print("Loading pretrained spatial encoder weights...")
    model.spatial_encoder.encoder.load_state_dict(torch.load(encoder_path, weights_only=True))
else:
    print("Pretrained encoder not found. Training from scratch.")
"""
            # Split new source by newlines and append \n except for the last line
            new_source_lines = [line + "\\n" for line in new_source.split("\\n")]
            new_source_lines[-1] = new_source_lines[-1].rstrip("\\n")
            cell["source"] = new_source_lines
            break

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully!")
