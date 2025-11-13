"""
Basic tests for VoidX package.
"""

import numpy as np
import torch

from voidx.data import generate_sample_data, preprocess_data, GalaxyDataset
from voidx.models import VoidDetectorMLP, VoidDetectorCNN, VoidDetectorAttention


def test_generate_sample_data():
    """Test sample data generation."""
    n_galaxies = 100
    n_neighbors = 50

    positions, labels, neighbor_distances = generate_sample_data(
        n_galaxies=n_galaxies,
        n_neighbors=n_neighbors,
        void_fraction=0.3,
        random_seed=42,
    )

    assert positions.shape == (n_galaxies, 3)
    assert labels.shape == (n_galaxies,)
    assert neighbor_distances.shape == (n_galaxies, n_neighbors)
    assert set(np.unique(labels)) <= {0, 1}


def test_galaxy_dataset():
    """Test GalaxyDataset class."""
    n_galaxies = 50
    n_neighbors = 20

    positions, labels, neighbor_distances = generate_sample_data(
        n_galaxies=n_galaxies,
        n_neighbors=n_neighbors,
    )

    dataset = GalaxyDataset(
        positions=positions,
        labels=labels,
        neighbor_distances=neighbor_distances,
        normalize=True,
    )

    assert len(dataset) == n_galaxies

    features, label = dataset[0]
    assert features.shape == (3 + n_neighbors,)
    assert isinstance(label, torch.Tensor)


def test_preprocess_data():
    """Test data preprocessing and splitting."""
    n_galaxies = 100
    n_neighbors = 50

    positions, labels, neighbor_distances = generate_sample_data(
        n_galaxies=n_galaxies,
        n_neighbors=n_neighbors,
    )

    datasets = preprocess_data(
        positions=positions,
        labels=labels,
        neighbor_distances=neighbor_distances,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
    )

    assert 'train' in datasets
    assert 'val' in datasets
    assert 'test' in datasets

    total = len(datasets['train']) + len(datasets['val']) + len(datasets['test'])
    assert total == n_galaxies


def test_mlp_model():
    """Test MLP model creation and forward pass."""
    model = VoidDetectorMLP(
        input_dim=103,
        hidden_dims=(128, 64),
        dropout_rate=0.3,
    )

    # Test forward pass
    batch_size = 10
    x = torch.randn(batch_size, 103)
    output = model(x)

    assert output.shape == (batch_size, 1)
    assert torch.all((output >= 0) & (output <= 1))  # Sigmoid output


def test_cnn_model():
    """Test CNN model creation and forward pass."""
    model = VoidDetectorCNN(
        n_neighbors=100,
        n_conv_layers=2,
        n_filters=32,
    )

    # Test forward pass
    batch_size = 10
    x = torch.randn(batch_size, 103)
    output = model(x)

    assert output.shape == (batch_size, 1)
    assert torch.all((output >= 0) & (output <= 1))


def test_attention_model():
    """Test Attention model creation and forward pass."""
    model = VoidDetectorAttention(
        n_neighbors=100,
        embedding_dim=32,
        n_heads=2,
        n_layers=1,
    )

    # Test forward pass
    batch_size = 10
    x = torch.randn(batch_size, 103)
    output = model(x)

    assert output.shape == (batch_size, 1)
    assert torch.all((output >= 0) & (output <= 1))


if __name__ == "__main__":
    # Run tests manually
    print("Running tests...")

    test_generate_sample_data()
    print("✓ test_generate_sample_data passed")

    test_galaxy_dataset()
    print("✓ test_galaxy_dataset passed")

    test_preprocess_data()
    print("✓ test_preprocess_data passed")

    test_mlp_model()
    print("✓ test_mlp_model passed")

    test_cnn_model()
    print("✓ test_cnn_model passed")

    test_attention_model()
    print("✓ test_attention_model passed")

    print("\nAll tests passed!")
