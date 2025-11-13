# Code Refactoring Summary

This document summarizes the reorganization of code from Jupyter notebooks into reusable Python modules.

## Overview

The following reusable components have been extracted from notebooks in the `notebooks/` folder and organized into the `voidx/` module:

## Extracted Components

### 1. `voidx/data.py`

**GalaxyDataset Class**
- PyTorch Dataset for galaxy data with positions and void labels
- Originally defined in: `model_training.ipynb` (Cell 18), `void_finder.ipynb` (Cell 15)
- Now imported from: `voidx.data.GalaxyDataset`

**split_indices Function**
- Splits indices into train, validation, and test sets with reproducibility
- Originally defined in: `model_training.ipynb` (Cell 14), `void_finder.ipynb` (Cell 15)
- Now imported from: `voidx.data.split_indices`

**normalize_features Function**
- Normalizes features using training set statistics (mean/std normalization)
- Originally scattered across: `model_training.ipynb`, `void_finder.ipynb`
- Now imported from: `voidx.data.normalize_features`

### 2. `voidx/models.py`

**MLP Class**
- Multi-Layer Perceptron for binary classification of void membership
- Originally defined in: `model_training.ipynb` (Cell 20)
- Now imported from: `voidx.models.MLP`

### 3. `voidx/utils.py`

**convert_to_Cartesian Function**
- Converts RA, Dec, z coordinates to Cartesian coordinates
- Originally defined in: `data_preparation.ipynb`
- Now imported from: `voidx.utils.convert_to_Cartesian`

**evaluate_model Function**
- Comprehensive model evaluation with multiple metrics (accuracy, F1, ROC-AUC, Brier, etc.)
- Originally defined in: `model_training.ipynb` (Cell 22)
- Now imported from: `voidx.utils.evaluate_model`

### 4. `voidx/config.py`

**setup_paths Function**
- Configures directory paths for data, checkpoints, plots, and results
- Originally scattered across: `data_preparation.ipynb` (Cells 7, 9), `model_training.ipynb` (Cells 7, 9), `void_finder.ipynb` (Cells 8, 10)
- Now imported from: `voidx.config.setup_paths`

**setup_device_and_seed Function**
- Configures computation device and random seed for reproducibility
- Originally defined in: `model_training.ipynb` (Cell 11), `void_finder.ipynb` (Cell 14)
- Now imported from: `voidx.config.setup_device_and_seed`

**TrainingConfig Class**
- Dataclass for training configuration parameters
- Now imported from: `voidx.config.TrainingConfig`

**DataConfig Class**
- Dataclass for data loading and preprocessing configuration
- Now imported from: `voidx.config.DataConfig`

## Notebook Updates

All notebooks have been updated to import from the `voidx` module instead of defining these components locally:

### data_preparation.ipynb
- Added import: `from voidx import convert_to_Cartesian, setup_paths`
- Removed: Local path setup code (Cells 7, 9)
- Now uses: `setup_paths()` function

### void_finder.ipynb
- Added import: `from voidx import GalaxyDataset, split_indices, normalize_features, setup_paths, setup_device_and_seed`
- Removed: GalaxyDataset class definition, split_indices function, path setup code
- Now uses: Imported functions and classes

### model_training.ipynb
- Added import: `from voidx import GalaxyDataset, split_indices, normalize_features, MLP, evaluate_model, setup_paths, setup_device_and_seed`
- Removed: GalaxyDataset class, split_indices function, MLP class, evaluate function, path setup code
- Now uses: Imported functions and classes

## Benefits

1. **Code Reusability**: Common functionality is now defined once and can be imported across all notebooks
2. **Maintainability**: Updates to shared code only need to be made in one place
3. **Testing**: Extracted code can be easily unit tested
4. **Documentation**: Centralized docstrings for all reusable components
5. **Clean Notebooks**: Notebooks now focus on analysis and experimentation rather than infrastructure code

## Usage Example

```python
# Import reusable components
from voidx import (
    GalaxyDataset, 
    split_indices, 
    normalize_features,
    MLP,
    evaluate_model,
    setup_paths,
    setup_device_and_seed
)

# Setup environment
device = setup_device_and_seed(device='cpu', seed=42)
paths = setup_paths('my_experiment', local=True)

# Prepare data
train_idx, val_idx, test_idx = split_indices(n_samples, train=0.7, val=0.15)
X_train_norm, X_val_norm, X_test_norm, stats = normalize_features(
    X_train, X_val, X_test
)

# Create datasets
ds_train = GalaxyDataset(X_train_norm, y_train)
ds_val = GalaxyDataset(X_val_norm, y_val)

# Train model
model = MLP(in_dim=3, hidden=(256, 128, 64), dropout=0.3)
# ... training loop ...

# Evaluate
metrics = evaluate_model(model, test_loader, device=device)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
```

## Files Modified

- `voidx/__init__.py` - Updated exports
- `voidx/data.py` - Added GalaxyDataset, split_indices, normalize_features
- `voidx/models.py` - Added MLP class
- `voidx/utils.py` - Added evaluate_model function
- `voidx/config.py` - Added setup_paths, setup_device_and_seed, config classes
- `notebooks/data_preparation.ipynb` - Updated to use voidx imports
- `notebooks/void_finder.ipynb` - Updated to use voidx imports
- `notebooks/model_training.ipynb` - Updated to use voidx imports
