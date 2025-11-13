# VoidX Notebooks

This directory contains Jupyter notebooks for interactive exploration and experimentation with VoidX.

## 🎉 Recent Update: Code Reorganization

The notebooks have been refactored to use reusable components from the `voidx` module instead of defining them locally. This makes the notebooks cleaner and focuses them on analysis rather than implementation details.

**What changed:**
- Common classes and functions are now imported from `voidx` module
- Notebooks are shorter and easier to read
- Code is reusable across notebooks and scripts
- See `REFACTORING_SUMMARY.md` in the root directory for details

## Available Notebooks

### data_preparation.ipynb

**Purpose:** Prepare galaxy data and classify void membership

**Contents:**
- Load galaxy and void catalog data
- Convert coordinates to Cartesian system (using `voidx.convert_to_Cartesian`)
- Classify galaxies as inside or outside voids
- Save processed data for training

**Now imports from voidx:**
- `convert_to_Cartesian` - Coordinate conversion utility
- `setup_paths` - Path configuration utility

### void_finder.ipynb

**Purpose:** Train neural networks to identify voids

**Contents:**
- Load prepared galaxy data
- Split data into train/val/test sets (using `voidx.split_indices`)
- Create PyTorch datasets (using `voidx.GalaxyDataset`)
- Train void detection models
- Make predictions on new data

**Now imports from voidx:**
- `GalaxyDataset` - PyTorch dataset for galaxy data
- `split_indices` - Data splitting utility
- `normalize_features` - Feature normalization
- `setup_paths`, `setup_device_and_seed` - Configuration utilities

### model_training.ipynb

**Purpose:** Train and evaluate void detection models

**Contents:**
- Load preprocessed data
- Create and train MLP models (using `voidx.MLP`)
- Evaluate model performance (using `voidx.evaluate_model`)
- Visualize results and metrics

**Now imports from voidx:**
- `GalaxyDataset` - PyTorch dataset
- `split_indices` - Data splitting
- `normalize_features` - Normalization
- `MLP` - Neural network model
- `evaluate_model` - Comprehensive evaluation
- `setup_paths`, `setup_device_and_seed` - Configuration

## Getting Started

1. Install Jupyter and VoidX:
```bash
pip install jupyter
pip install -e ..
```

2. Launch Jupyter:
```bash
jupyter notebook
```

3. Open a notebook and run the cells sequentially

## Tips

- **Run cells in order**: Notebooks are designed to be run sequentially from top to bottom
- **Experiment**: Feel free to modify parameters and try different configurations
- **Save your work**: Use "File > Save and Checkpoint" to save your progress
- **Restart if needed**: If things get messy, use "Kernel > Restart & Clear Output"
- **Use the modules**: All common functionality is now in `voidx` - no need to copy/paste code between notebooks!

## Requirements

- Jupyter Notebook or JupyterLab
- All VoidX dependencies (see requirements.txt)
- matplotlib for inline plotting

## Data

The notebooks work with galaxy survey data and void catalogs. Place your data files according to the path setup in each notebook, or modify the configuration to point to your data location.

## Outputs

Notebook outputs (plots, trained models, etc.) are saved to:
- `checkpoints/` - Model weights
- `plot/` - Visualizations
- `result/` - Processed results

Add these directories to `.gitignore` if you don't want to commit them.
