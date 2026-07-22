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
    Calculate IoU, Pixel Accuracy, Precision, Recall, F1-score, and Dice Coefficient.
    Targets containing -1 are ignored.
    """
    valid_mask = (targets != -1)
    preds = preds[valid_mask]
    targets = targets[valid_mask]
    
    if len(preds) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    if num_classes == 2:
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
        
        # IoU (Jaccard Index) & Dice Coefficient
        iou = tp / (tp + fp + fn + 1e-8)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        
        return iou.item(), accuracy.item(), precision.item(), recall.item(), f1_score.item(), dice.item()

def calculate_metrics_dict(preds, targets):
    """Return dictionary of all metrics including IoU and Dice Coefficient."""
    iou, acc, prec, rec, f1, dice = calculate_metrics(preds, targets)
    return {
        "IoU": iou,
        "Dice": dice,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    }

def calculate_multiclass_metrics(preds, targets, num_classes=4):
    """Calculate mean IoU, overall accuracy, and class-wise IoU for multi-class segmentation (Land Use)."""
    valid_mask = (targets != -1)
    preds = preds[valid_mask]
    targets = targets[valid_mask]
    
    if len(preds) == 0:
        return 0.0, 0.0, {}
    
    correct = torch.sum(preds == targets).float()
    overall_acc = correct / (len(targets) + 1e-8)
    
    class_ious = {}
    for c in range(num_classes):
        tp = torch.sum((preds == c) & (targets == c)).float()
        fp = torch.sum((preds == c) & (targets != c)).float()
        fn = torch.sum((preds != c) & (targets == c)).float()
        iou_c = tp / (tp + fp + fn + 1e-8)
        class_ious[c] = iou_c.item()
        
    mean_iou = sum(class_ious.values()) / num_classes
    return mean_iou, overall_acc.item(), class_ious

