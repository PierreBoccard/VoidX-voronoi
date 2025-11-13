"""
Example script for training a void detection model.

This script demonstrates how to:
1. Generate or load galaxy data
2. Preprocess and split the data
3. Create and train a model
4. Evaluate performance
"""

import torch
import numpy as np
from pathlib import Path

from voidx.data import generate_sample_data, preprocess_data, create_dataloaders
from voidx.models import VoidDetectorMLP
from voidx.train import Trainer
from voidx.utils import (
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
)


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration
    n_galaxies = 10000
    n_neighbors = 100
    void_fraction = 0.3

    print("=" * 60)
    print("VoidX: Void Detection Training Example")
    print("=" * 60)

    # Step 1: Generate sample data
    print("\n1. Generating sample galaxy data...")
    positions, labels, neighbor_distances = generate_sample_data(
        n_galaxies=n_galaxies,
        n_neighbors=n_neighbors,
        void_fraction=void_fraction,
        random_seed=42,
    )
    print(f"   Generated {n_galaxies} galaxies")
    print(f"   Void fraction: {labels.mean():.3f}")

    # Step 2: Preprocess and split data
    print("\n2. Preprocessing and splitting data...")
    datasets = preprocess_data(
        positions=positions,
        labels=labels,
        neighbor_distances=neighbor_distances,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        normalize=True,
        random_seed=42,
    )
    print(f"   Train set: {len(datasets['train'])} samples")
    print(f"   Val set: {len(datasets['val'])} samples")
    print(f"   Test set: {len(datasets['test'])} samples")

    # Step 3: Create dataloaders
    print("\n3. Creating data loaders...")
    dataloaders = create_dataloaders(
        datasets,
        batch_size=64,
        num_workers=0,
    )

    # Step 4: Create model
    print("\n4. Creating model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")

    model = VoidDetectorMLP(
        input_dim=103,  # 3 position + 100 neighbor distances
        hidden_dims=(256, 128, 64),
        dropout_rate=0.3,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {n_params:,}")

    # Step 5: Create trainer and train
    print("\n5. Training model...")
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        learning_rate=0.001,
        weight_decay=1e-5,
        device=device,
        checkpoint_dir='checkpoints',
    )

    history = trainer.train(
        num_epochs=50,
        early_stopping_patience=10,
        verbose=True,
    )

    # Save training history
    trainer.save_history()
    print("\n   Training history saved!")

    # Step 6: Evaluate on test set
    print("\n6. Evaluating on test set...")
    trainer.load_checkpoint('best_model.pth')
    results = trainer.evaluate(dataloaders['test'])

    print("\nTest Results:")
    print(f"  Accuracy:    {results['accuracy']:.4f}")
    print(f"  Precision:   {results['precision']:.4f}")
    print(f"  Recall:      {results['recall']:.4f}")
    print(f"  F1 Score:    {results['f1']:.4f}")
    print(f"  Specificity: {results['specificity']:.4f}")

    # Step 7: Create visualizations
    print("\n7. Creating visualizations...")
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    plot_training_history(
        history,
        save_path=output_dir / 'training_history.png',
    )

    plot_confusion_matrix(
        results['labels'],
        results['predictions'],
        save_path=output_dir / 'confusion_matrix.png',
    )

    plot_roc_curve(
        results['labels'],
        results['probabilities'],
        save_path=output_dir / 'roc_curve.png',
    )

    print(f"   Visualizations saved to {output_dir}/")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
