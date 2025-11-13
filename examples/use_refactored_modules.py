#!/usr/bin/env python3
"""
Example script demonstrating how to use the refactored voidx modules.

This shows how the components extracted from the notebooks can be used
in a standalone Python script.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

# Import all the refactored components
from voidx import (
    GalaxyDataset,
    split_indices,
    normalize_features,
    MLP,
    evaluate_model,
    setup_paths,
    setup_device_and_seed,
    TrainingConfig,
    DataConfig,
)


def main():
    """Main example workflow."""
    
    print("VoidX Refactored Modules Example")
    print("=" * 60)
    
    # 1. Setup environment
    print("\n1. Setting up environment...")
    device = setup_device_and_seed(device='cpu', seed=42)
    print(f"   Device: {device}")
    
    # 2. Setup paths
    print("\n2. Setting up paths...")
    paths = setup_paths('example_experiment', local=True)
    print(f"   Data directory: {paths['data_dir']}")
    
    # 3. Generate or load sample data
    print("\n3. Generating sample data...")
    n_samples = 1000
    n_features = 3
    
    # Simulate galaxy positions
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Simulate void membership labels (0 or 1)
    y = np.random.randint(0, 2, n_samples).astype(np.int32)
    print(f"   Generated {n_samples} samples with {n_features} features")
    
    # 4. Split data
    print("\n4. Splitting data...")
    train_idx, val_idx, test_idx = split_indices(
        n_samples, 
        train=0.7, 
        val=0.15, 
        seed=42
    )
    print(f"   Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # 5. Normalize features
    print("\n5. Normalizing features...")
    X_train_norm, X_val_norm, X_test_norm, (mean, std) = normalize_features(
        X_train, X_val, X_test
    )
    print(f"   Mean shape: {mean.shape}, Std shape: {std.shape}")
    
    # 6. Create datasets and dataloaders
    print("\n6. Creating datasets...")
    config = TrainingConfig(batch_size=32, learning_rate=0.001)
    
    ds_train = GalaxyDataset(X_train_norm, y_train, y_dtype=torch.float32)
    ds_val = GalaxyDataset(X_val_norm, y_val, y_dtype=torch.float32)
    ds_test = GalaxyDataset(X_test_norm, y_test, y_dtype=torch.float32)
    
    dl_train = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=config.batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=config.batch_size, shuffle=False)
    
    print(f"   Created dataloaders with batch_size={config.batch_size}")
    
    # 7. Create model
    print("\n7. Creating MLP model...")
    model = MLP(
        in_dim=n_features,
        hidden=(128, 64, 32),
        dropout=0.3
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {n_params:,}")
    
    # 8. Training (simplified for example)
    print("\n8. Training model (dummy training)...")
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Just one epoch for demonstration
    model.train()
    for batch_idx, (x, y_batch) in enumerate(dl_train):
        if batch_idx > 5:  # Only train on a few batches for demo
            break
        x, y_batch = x.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    
    print("   Model trained (demo only)")
    
    # 9. Evaluate model
    print("\n9. Evaluating model...")
    metrics = evaluate_model(model, dl_test, device=device)
    
    print(f"   Accuracy: {metrics['accuracy']:.3f}")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall: {metrics['recall']:.3f}")
    print(f"   F1 Score: {metrics['f1']:.3f}")
    print(f"   ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"   Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"   MCC: {metrics['mcc']:.3f}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("\nAll these components were extracted from the notebooks:")
    print("  - GalaxyDataset, split_indices, normalize_features from voidx.data")
    print("  - MLP from voidx.models")
    print("  - evaluate_model from voidx.utils")
    print("  - setup_paths, setup_device_and_seed from voidx.config")
    print("\nThe notebooks now import these instead of defining them locally.")


if __name__ == '__main__':
    main()
