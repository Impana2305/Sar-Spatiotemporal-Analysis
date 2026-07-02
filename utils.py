import torch
import random
import numpy as np
import json
import os

def set_seed(seed=42):
    """Set all seeds to make results reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path="config.json"):
    """Load configuration from a JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def calculate_metrics(preds, targets, num_classes=2):
    """
    Calculate IoU, Pixel Accuracy, Precision, Recall, and F1-score.
    Targets containing -1 are ignored.
    """
    valid_mask = (targets != -1)
    preds = preds[valid_mask]
    targets = targets[valid_mask]
    
    if len(preds) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    tp = torch.sum((preds == 1) & (targets == 1)).float()
    tn = torch.sum((preds == 0) & (targets == 0)).float()
    fp = torch.sum((preds == 1) & (targets == 0)).float()
    fn = torch.sum((preds == 0) & (targets == 1)).float()
    
    # Pixel Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    # Precision, Recall, F1
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # IoU for water class (1)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return iou.item(), accuracy.item(), precision.item(), recall.item(), f1_score.item()
