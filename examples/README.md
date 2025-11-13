# VoidX Examples

This directory contains example scripts demonstrating how to use VoidX.

## Available Examples

### 1. train_model.py

Complete training pipeline example that:
- Generates sample galaxy data
- Preprocesses and splits the data
- Creates and trains a void detection model
- Evaluates the model on test data
- Creates visualizations

**Usage:**
```bash
python train_model.py
```

**Output:**
- Trained model checkpoint in `checkpoints/`
- Training history JSON file
- Visualizations in `outputs/` directory

### 2. predict.py

Demonstrates how to load a trained model and make predictions on new data.

**Usage:**
```bash
# First train a model
python train_model.py

# Then make predictions
python predict.py
```

**Output:**
- Prediction accuracy metrics
- 3D visualization comparing predictions to ground truth

### 3. config_example.py

Demonstrates how to use the VoidX configuration system for notebooks.
Shows how to centralize common parameters and path setup to avoid repetition.

**Usage:**
```bash
python config_example.py
```

**Features:**
- Create configurations with default or custom parameters
- Automatic path setup for data, checkpoints, plots, and results
- Easy access to all configuration values
- Seamless integration with notebooks

## Running the Examples

1. Make sure VoidX is installed:
```bash
pip install -e ..
```

2. Run any example script:
```bash
python train_model.py
```

3. Check the output directories for results:
- `checkpoints/` - Saved model weights
- `outputs/` - Visualizations and plots

## Customization

You can modify the examples to:
- Use your own galaxy data instead of synthetic data
- Adjust model hyperparameters
- Try different model architectures (MLP, CNN, Attention)
- Change training parameters (learning rate, epochs, etc.)

See the VoidX documentation for more details on available options.
