# Quick Start Guide for VoidX

Welcome to VoidX! This guide will help you get up and running quickly.

## Installation

```bash
# Clone the repository
git clone https://github.com/LucasSauniere/VoidX.git
cd VoidX

# Install dependencies and the package
pip install -r requirements.txt
pip install -e .
```

## 5-Minute Tutorial

### 1. Basic Usage

```python
# Import VoidX
from voidx.data import generate_sample_data, preprocess_data, create_dataloaders
from voidx.models import VoidDetectorMLP
from voidx.train import Trainer

# Generate sample data
positions, labels, neighbor_distances = generate_sample_data(
    n_galaxies=5000,
    n_neighbors=100,
    void_fraction=0.3,
)

# Preprocess and split
datasets = preprocess_data(
    positions, labels, neighbor_distances,
    train_split=0.7, val_split=0.15, test_split=0.15,
)

# Create data loaders
dataloaders = create_dataloaders(datasets, batch_size=64)

# Create model
model = VoidDetectorMLP(input_dim=103, hidden_dims=(256, 128, 64))

# Train
trainer = Trainer(
    model=model,
    train_loader=dataloaders['train'],
    val_loader=dataloaders['val'],
)
history = trainer.train(num_epochs=50, early_stopping_patience=10)

# Evaluate
results = trainer.evaluate(dataloaders['test'])
print(f"Test Accuracy: {results['accuracy']:.4f}")
```

### 2. Using Example Scripts

Run the complete training example:
```bash
python examples/train_model.py
```

This will:
- Generate sample data
- Train a model
- Evaluate performance
- Create visualizations in `outputs/`

Then make predictions with the trained model:
```bash
python examples/predict.py
```

### 3. Using Jupyter Notebooks

Start Jupyter and explore interactively:
```bash
jupyter notebook notebooks/
```

- **01_data_exploration.ipynb**: Learn about the data
- **02_model_training.ipynb**: Train and compare models

## Key Features

### Multiple Model Architectures

```python
from voidx.models import VoidDetectorMLP, VoidDetectorCNN, VoidDetectorAttention

# Multi-Layer Perceptron
mlp = VoidDetectorMLP(input_dim=103, hidden_dims=(256, 128, 64))

# Convolutional Neural Network
cnn = VoidDetectorCNN(n_neighbors=100, n_conv_layers=3)

# Attention-based Model
attention = VoidDetectorAttention(n_neighbors=100, n_heads=4)
```

### Visualization Tools

```python
from voidx.utils import (
    plot_galaxy_distribution_3d,
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
)

# Visualize galaxy distribution
plot_galaxy_distribution_3d(positions, labels)

# Plot training progress
plot_training_history(history)

# Evaluate predictions
plot_confusion_matrix(true_labels, predictions)
plot_roc_curve(true_labels, probabilities)
```

### Loading Your Own Data

```python
from voidx.data import load_galaxy_data

# Load from CSV (columns: x, y, z, label, dist1, dist2, ...)
positions, labels, neighbor_distances = load_galaxy_data(
    'data/my_galaxies.csv',
    format='csv'
)

# Or from NumPy array
positions, labels, neighbor_distances = load_galaxy_data(
    'data/my_galaxies.npy',
    format='npy'
)
```

## Data Format

Your data should contain:
- **Positions**: (x, y, z) coordinates for each galaxy
- **Labels**: Binary (0 = not in void, 1 = in void)
- **Neighbor Distances**: Distances to N nearest neighbors (e.g., 100)

## Next Steps

1. **Explore the notebooks**: Run the Jupyter notebooks to understand the data and models
2. **Try different models**: Compare MLP, CNN, and Attention architectures
3. **Use your own data**: Load real galaxy survey data
4. **Tune hyperparameters**: Adjust learning rate, batch size, model architecture
5. **Visualize results**: Create 3D plots and performance charts

## Getting Help

- Read the full README.md for detailed documentation
- Check the examples/ directory for usage patterns
- Look at the notebooks/ for interactive tutorials
- Examine the code in voidx/ for implementation details

## Common Issues

**Import Error**: Make sure you installed the package:
```bash
pip install -e .
```

**CUDA not available**: The package works on CPU too, but will be slower:
```python
trainer = Trainer(..., device='cpu')
```

**Out of memory**: Reduce batch size:
```python
dataloaders = create_dataloaders(datasets, batch_size=32)
```

## Project Structure

```
VoidX/
├── voidx/              # Main package code
├── notebooks/          # Jupyter notebooks
├── examples/           # Example scripts
├── tests/              # Unit tests
├── data/               # Data directory
└── README.md          # Full documentation
```

Happy void hunting! 🌌
