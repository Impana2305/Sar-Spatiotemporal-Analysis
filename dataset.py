import os
import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset
from scipy.ndimage import uniform_filter

def apply_speckle_filter(image, size=3):
    """Applies a simple box filter to reduce speckle noise."""
    filtered_image = np.zeros_like(image)
    for band in range(image.shape[0]):
        filtered_image[band] = uniform_filter(image[band], size=size)
    return filtered_image

def preprocess_s1(data):
    """Fill NaNs, apply speckle filter, clip range, and normalize."""
    vv = np.nan_to_num(data[0], nan=-25.0)
    vh = np.nan_to_num(data[1], nan=-30.0)
    
    stacked = np.stack([vv, vh], axis=0)
    filtered = apply_speckle_filter(stacked, size=3)
    
    vv_clipped = np.clip(filtered[0], -30.0, 0.0)
    vh_clipped = np.clip(filtered[1], -35.0, -5.0)
    
    vv_norm = (vv_clipped - (-30.0)) / 30.0
    vh_norm = (vh_clipped - (-35.0)) / 30.0
    
    return np.stack([vv_norm, vh_norm], axis=0)

class Sen1FloodsDataset(Dataset):
    def __init__(self, file_pairs, target_size=(256, 256)):
        self.file_pairs = file_pairs
        self.target_size = target_size
        
    def __len__(self):
        return len(self.file_pairs)
        
    def __getitem__(self, idx):
        s1_path, label_path = self.file_pairs[idx]
        
        with rasterio.open(s1_path) as src:
            s1_data = src.read()
        
        with rasterio.open(label_path) as src:
            label_data = src.read(1)
            
        s1_preprocessed = preprocess_s1(s1_data)
        
        s1_tensor = torch.tensor(s1_preprocessed, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        s1_resized = torch.nn.functional.interpolate(
            s1_tensor, size=self.target_size, mode='bilinear', align_corners=False
        ).squeeze(0)
        
        label_resized = torch.nn.functional.interpolate(
            label_tensor, size=self.target_size, mode='nearest'
        ).squeeze(0).squeeze(0).long()
        
        return s1_resized, label_resized


class TrueTimeSeriesSARDataset(Dataset):
    """
    Expects true time series sequences. 
    `sequences` is a list where each element is a tuple: (list_of_s1_paths, label_path)
    Where list_of_s1_paths has length `seq_length` (e.g. 3 time steps for the exact same location).
    """
    def __init__(self, sequences, seq_length=3, target_size=(256, 256)):
        self.sequences = sequences
        self.seq_length = seq_length
        self.target_size = target_size
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        s1_paths, label_path = self.sequences[idx]
        seq_s1 = []
        
        for s1_path in s1_paths:
            with rasterio.open(s1_path) as src:
                s1_data = src.read()
            s1_preprocessed = preprocess_s1(s1_data)
            s1_tensor = torch.tensor(s1_preprocessed, dtype=torch.float32).unsqueeze(0)
            s1_resized = torch.nn.functional.interpolate(
                s1_tensor, size=self.target_size, mode='bilinear', align_corners=False
            ).squeeze(0)
            seq_s1.append(s1_resized)
            
        with rasterio.open(label_path) as src:
            label_data = src.read(1)
        label_tensor = torch.tensor(label_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        label_resized = torch.nn.functional.interpolate(
            label_tensor, size=self.target_size, mode='nearest'
        ).squeeze(0).squeeze(0).long()
        
        seq_s1_tensor = torch.stack(seq_s1) # Shape: [seq_length, C, H, W]
        return seq_s1_tensor, label_resized

def get_train_val_test_splits(dataset_length, split_ratios=(0.7, 0.15, 0.15)):
    """Return indices for train, validation, and test splits."""
    indices = np.random.permutation(dataset_length)
    train_end = int(split_ratios[0] * dataset_length)
    val_end = train_end + int(split_ratios[1] * dataset_length)
    
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]
