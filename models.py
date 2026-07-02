import torch
import torch.nn as nn
import math

class SpatialEncoder(nn.Module):
    """Enhanced Encoder with BatchNorm and slightly increased depth."""
    def __init__(self, in_channels=2, dropout_rate=0.1):
        super(SpatialEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # Layer 1: 256x256 -> 128x128
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            # Layer 2: 128x128 -> 64x64
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # Layer 3: 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Layer 4: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout2d(dropout_rate),
            
            # Layer 5 (New for depth): 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.encoder(x) # Output shape: [B, 256, 8, 8]


class ConvAutoencoder(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(ConvAutoencoder, self).__init__()
        self.encoder = SpatialEncoder(dropout_rate=dropout_rate)
        
        self.decoder = nn.Sequential(
            # Layer 1: 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Layer 2: 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Layer 3: 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # Layer 4: 64x64 -> 128x128
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            # Layer 5: 128x128 -> 256x256
            nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x shape: [batch_size, seq_len, d_model]"""
        return x + self.pe[:, :x.size(1), :]


class TemporalTransformerPredictor(nn.Module):
    def __init__(self, seq_length=3, spatial_dim=256*8*8, embed_dim=512, num_heads=8, hidden_dim=512, num_layers=2, dropout=0.1):
        super(TemporalTransformerPredictor, self).__init__()
        
        self.spatial_encoder = SpatialEncoder(dropout_rate=dropout)
        
        self.feature_proj = nn.Linear(spatial_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_length)
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.feature_unproj = nn.Linear(embed_dim, spatial_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        # x shape: [B, SeqLen, C, H, W]
        B, S, C, H, W = x.shape
        x_reshaped = x.view(B * S, C, H, W)
        
        spatial_features = self.spatial_encoder(x_reshaped) # Shape: [B*S, 256, 8, 8]
        sf_flat = spatial_features.view(B, S, -1) # Shape: [B, S, 256*8*8]
        
        sf_proj = self.feature_proj(sf_flat)
        sf_proj = self.layer_norm(sf_proj) # Normalize temporal features consistently
        sf_proj = self.pos_encoder(sf_proj) # Proper positional encoding
        
        temporal_features = self.transformer(sf_proj)
        
        last_step_features = temporal_features[:, -1, :] # Predict using last time step
        last_step_unproj = self.feature_unproj(last_step_features)
        
        last_step_spatial = last_step_unproj.view(B, 256, 8, 8)
        
        logits = self.decoder(last_step_spatial) # Shape: [B, 2, 256, 256]
        return logits
