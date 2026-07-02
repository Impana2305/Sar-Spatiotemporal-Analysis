import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import calculate_metrics

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

def train_autoencoder(model, train_loader, val_loader, num_epochs=15, lr=1e-3, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    os.makedirs('models', exist_ok=True)
    early_stopping = EarlyStopping(patience=5, path='models/best_autoencoder.pth')
    
    print("Starting Autoencoder Training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                reconstructed, _ = model(data)
                loss = criterion(reconstructed, data)
                val_loss += loss.item() * data.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")
        
        scheduler.step(val_loss)
        # For early stopping we want to minimize loss, so we pass -val_loss to maximize
        early_stopping(-val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
            
    print("Autoencoder Training Complete. Best model saved to models/best_autoencoder.pth")


def train_temporal_model(model, train_loader, val_loader, num_epochs=30, lr=1e-4, device='cpu'):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    os.makedirs('models', exist_ok=True)
    early_stopping = EarlyStopping(patience=5, path='models/best_temporal_model.pth')
    
    print("Starting Temporal Transformer Training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for seq_data, target in train_loader:
            seq_data, target = seq_data.to(device), target.to(device)
            optimizer.zero_grad()
            
            outputs = model(seq_data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * seq_data.size(0)
        train_loss /= len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for seq_data, target in val_loader:
                seq_data, target = seq_data.to(device), target.to(device)
                outputs = model(seq_data)
                loss = criterion(outputs, target)
                val_loss += loss.item() * seq_data.size(0)
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(target.cpu())
                
        val_loss /= len(val_loader.dataset)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        iou, acc, precision, recall, f1 = calculate_metrics(all_preds, all_targets)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val Metrics - IoU: {iou:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, P: {precision:.4f}, R: {recall:.4f}")
        
        # Maximize IoU
        scheduler.step(iou)
        early_stopping(iou, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered based on IoU.")
            break
            
    print("Temporal Training Complete. Best model saved to models/best_temporal_model.pth")
