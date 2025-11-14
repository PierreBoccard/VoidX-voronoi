# VoidX

A machine learning package for detecting cosmic voids in galaxy distributions using PyTorch.

## Overview

VoidX leverages deep learning techniques to identify galaxies located in cosmic voids based on their 3D positions and distances to neighboring galaxies. The package provides tools for data preprocessing, model training, evaluation, and visualization.

**New**: Voronoi tessellation-based features and Graph Neural Networks (GNN) to address spatial data leakage concerns!

## Features

- **Multiple Model Architectures**: MLP, CNN, Attention-based, and GNN models for void detection
- **Voronoi Tessellation**: Extract topological features from galaxy distributions to avoid spatial data leakage
- **Graph Neural Networks**: VoronoiGCN, VoronoiGAT, and VoronoiSAGE for leveraging spatial structure
- **Data Processing**: Utilities for loading, preprocessing, and normalizing galaxy data
- **Training Framework**: Comprehensive training pipeline with early stopping and checkpointing
- **Visualization Tools**: 3D visualization of galaxy distributions and model predictions
- **Jupyter Notebooks**: Interactive notebooks for data exploration and model training
- **Easy-to-use API**: Simple interface for training and making predictions

## What's New: Voronoi + GNN

### The Problem: Spatial Data Leakage

When training ML models on galaxy positions, there's a risk of spatial data leakage where the model memorizes positions rather than learning void characteristics. This causes poor generalization when tested on different spatial regions.

### The Solution: Voronoi Features + Graph Neural Networks

1. **Voronoi Tessellation**: Extract topological features (cell volumes, neighbor counts, adjacency) that capture local density without position memorization
2. **MLP with Voronoi features**: Use only topological features for classification
3. **GNN with Voronoi graph**: Leverage graph structure to capture multi-scale patterns while using positions in graph context

See [VORONOI_GNN_GUIDE.md](VORONOI_GNN_GUIDE.md) for detailed explanation and usage examples.

## Installation

### From Source

```bash
git clone https://github.com/LucasSauniere/VoidX.git
cd VoidX
pip install -e .
```

### Dependencies

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Matplotlib >= 3.7.0
- scikit-learn >= 1.3.0

See `requirements.txt` for a complete list of dependencies.

## Quick Start

### Generate Sample Data and Train a Model

```python
from voidx.data import generate_sample_data, preprocess_data, create_dataloaders
from voidx.models import VoidDetectorMLP
from voidx.train import Trainer

# Generate sample galaxy data
positions, labels, neighbor_distances = generate_sample_data(
    n_galaxies=10000,
    n_neighbors=100,
    void_fraction=0.3,
)

# Preprocess and split data
datasets = preprocess_data(
    positions=positions,
    labels=labels,
    neighbor_distances=neighbor_distances,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
)

# Create data loaders
dataloaders = create_dataloaders(datasets, batch_size=64)

# Create and train model
model = VoidDetectorMLP(input_dim=103, hidden_dims=(256, 128, 64))
trainer = Trainer(
    model=model,
    train_loader=dataloaders['train'],
    val_loader=dataloaders['val'],
)

# Train
history = trainer.train(num_epochs=50, early_stopping_patience=10)

# Evaluate
results = trainer.evaluate(dataloaders['test'])
print(f"Test Accuracy: {results['accuracy']:.4f}")
```

### Using Voronoi Features to Avoid Data Leakage

```python
from voidx.voronoi import VoronoiFeatureExtractor
from voidx.models import VoidMLP, VoronoiGCN
import numpy as np

# Load your galaxy positions and labels
positions = np.load('your_positions.npy')  # Shape: (N, 3)
labels = np.load('your_labels.npy')        # Shape: (N,)

# Extract Voronoi tessellation features
extractor = VoronoiFeatureExtractor(
    box_size=250.0,           # Your simulation box size
    use_periodic=True,        # Use periodic boundaries
)

features = extractor.extract_features(positions)

# Option 1: Train MLP with topological features (no positions!)
mlp_features = extractor.create_mlp_features(
    positions, 
    include_positions=False  # Avoids spatial leakage
)
model = VoidMLP(in_features=2)  # volume + neighbor_count

# Option 2: Train GNN with graph structure
gnn_features = np.hstack([
    features['normalized_volumes'][:, np.newaxis],
    features['neighbor_count'][:, np.newaxis],
    positions,  # OK in GNN due to graph context
])
model = VoronoiGCN(in_features=5)

# See examples/compare_mlp_gnn_voronoi.py for complete example
```

## Data Format

VoidX expects galaxy data with the following structure:

- **Positions**: 3D coordinates (x, y, z) for each galaxy
- **Labels**: Binary labels (0 = not in void, 1 = in void)
- **Neighbor Distances**: Distances to the N nearest neighbors (default N=100)

The data can be loaded from CSV, NumPy arrays, or pickle files:

```python
from voidx.data import load_galaxy_data

positions, labels, neighbor_distances = load_galaxy_data(
    filepath='data/galaxy_data.csv',
    format='csv'
)
```

## Model Architectures

### Multi-Layer Perceptron (MLP)

A fully connected neural network for void classification:

```python
from voidx.models import VoidDetectorMLP

model = VoidDetectorMLP(
    input_dim=103,  # 3 position + 100 neighbor distances
    hidden_dims=(256, 128, 64),
    dropout_rate=0.3,
)
```

### 1D Convolutional Network (CNN)

Uses 1D convolutions to process neighbor distance patterns:

```python
from voidx.models import VoidDetectorCNN

model = VoidDetectorCNN(
    n_neighbors=100,
    n_conv_layers=3,
    n_filters=64,
    kernel_size=5,
)
```

### Attention-based Model

Leverages self-attention to focus on relevant neighbor distances:

```python
from voidx.models import VoidDetectorAttention

model = VoidDetectorAttention(
    n_neighbors=100,
    embedding_dim=64,
    n_heads=4,
    n_layers=2,
)
```

## Notebooks

The `notebooks/` directory contains Jupyter notebooks for interactive exploration:

- **01_data_exploration.ipynb**: Data loading, visualization, and preprocessing
- **02_model_training.ipynb**: Model training, evaluation, and comparison

To use the notebooks:

```bash
jupyter notebook notebooks/
```

## Examples

The `examples/` directory contains standalone scripts:

- **train_model.py**: Complete training pipeline example
- **predict.py**: Load a trained model and make predictions

Run an example:

```bash
python examples/train_model.py
```

## Visualization

VoidX provides various visualization utilities:

```python
from voidx.utils import (
    plot_galaxy_distribution_3d,
    plot_neighbor_distances,
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
)

# Visualize galaxy distribution
plot_galaxy_distribution_3d(positions, labels)

# Plot training history
plot_training_history(history)

# Show confusion matrix
plot_confusion_matrix(true_labels, predictions)
```

## Project Structure

```
VoidX/
├── voidx/                 # Main package
│   ├── __init__.py
│   ├── config.py          # Configuration classes
│   ├── data.py            # Data loading and preprocessing
│   ├── models.py          # PyTorch model architectures
│   ├── train.py           # Training utilities
│   └── utils.py           # Visualization and utilities
├── notebooks/             # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── examples/              # Example scripts
│   ├── train_model.py
│   └── predict.py
├── tests/                 # Unit tests
├── data/                  # Data directory
├── setup.py               # Package setup
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Citation

If you use VoidX in your research, please cite:

```bibtex
@software{voidx2024,
  author = {Sauniere, Lucas},
  title = {VoidX: Machine Learning for Cosmic Void Detection},
  year = {2024},
  url = {https://github.com/LucasSauniere/VoidX}
}
```

## Contact

For questions or feedback, please open an issue on GitHub.