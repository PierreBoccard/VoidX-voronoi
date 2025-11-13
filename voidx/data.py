"""
Data loading and preprocessing module for VoidX.

This module handles:
- Loading galaxy data with positions, void labels, and neighbor distances
- Preprocessing and normalization
- Creating PyTorch datasets and dataloaders
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.model_selection import StratifiedShuffleSplit


class GalaxyDataset(Dataset):
    """PyTorch Dataset for galaxy data with positions and void labels.
    
    Args:
        X: Feature array (positions or other features)
        y: Label array (void membership flags)
        y_dtype: PyTorch dtype for labels (default: torch.float32)
    """
    def __init__(self, X, y, *, y_dtype=torch.float32):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=y_dtype)
        assert len(self.X) == len(self.y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def split_indices(n, train=0.7, val=0.15, seed=42):
    """Split indices into train, validation, and test sets.
    
    Args:
        n: Total number of samples
        train: Fraction for training set (default: 0.7)
        val: Fraction for validation set (default: 0.15)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (train_idx, val_idx, test_idx) as numpy arrays
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(n * train)
    n_val = int(n * val)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def split_indices_stratified(y, train=0.7, val=0.15, seed=42):
    """Stratified split of indices into train/validation/test to preserve class ratios.
    
    Args:
        y: array-like of shape (n,) with binary labels {0,1} (or multiclass)
        train: Fraction for training set
        val: Fraction for validation set
        seed: Random seed
    Returns:
        (train_idx, val_idx, test_idx) as numpy arrays
    """
    import numpy as np
    y = np.asarray(y)
    n = len(y)
    val_test = 1.0 - train
    # First split: train vs (val+test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val_test, random_state=seed)
    (train_idx, vt_idx) = next(sss1.split(np.zeros(n), y))
    # Second split: val vs test within vt_idx
    vt_y = y[vt_idx]
    val_frac_within_vt = val / max(val_test, 1e-12)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_frac_within_vt), random_state=seed)
    (val_rel, test_rel) = next(sss2.split(np.zeros(len(vt_idx)), vt_y))
    val_idx = vt_idx[val_rel]
    test_idx = vt_idx[test_rel]
    return train_idx, val_idx, test_idx


def normalize_features(X_train, X_val=None, X_test=None, epsilon=1e-6):
    """Normalize features using training set statistics.
    
    Args:
        X_train: Training feature array
        X_val: Validation feature array (optional)
        X_test: Test feature array (optional)
        epsilon: Small value to avoid division by zero
    
    Returns:
        Tuple of normalized arrays and (mean, std) statistics
    """
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + epsilon
    std[std == 0] = 1.0  # avoid division by zero
    
    X_train_norm = (X_train - mean) / std
    
    result = [X_train_norm]
    if X_val is not None:
        result.append((X_val - mean) / std)
    if X_test is not None:
        result.append((X_test - mean) / std)
    
    result.append((mean, std))
    return tuple(result)
