import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

def calculate_accuracy(y_pred, y_true):
    """
    Args:
    y_pred: Predicted probabilities or logits (shape: [batch_size, num_classes])
    y_true: Ground truth labels (shape: [batch_size])
    
    Returns:
    Accuracy
    """
    # Chọn lớp có xác suất cao nhất
    _, y_pred_classes = torch.max(y_pred, dim=1)
    
    # So sánh với nhãn thực tế
    correct = (y_pred_classes == y_true).sum().item()
    
    return correct / y_true.size(0)

import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        """
        Args:
            patience (int): Số epochs không cải thiện trước khi dừng huấn luyện.
            verbose (bool): In thông báo khi cải thiện.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.best_acc = 0.0
        self.early_stop = False

    def __call__(self, val_loss, val_acc):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Loss improved to {val_loss:.6f}.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Loss did not improve. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered!")
        
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter = 0
            if self.verbose:
                print(f"Accuracy improved to {val_acc:.6f}.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Accuracy did not improve. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered!")
