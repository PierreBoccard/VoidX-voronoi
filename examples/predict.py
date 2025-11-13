"""
Example script for using a trained model to make predictions.

This script demonstrates how to load a trained model and use it
to predict void membership for new galaxies.
"""

import torch
import numpy as np
from pathlib import Path

from voidx.data import generate_sample_data, GalaxyDataset
from voidx.models import VoidDetectorMLP
from voidx.utils import plot_predictions_3d


def main():
    print("=" * 60)
    print("VoidX: Void Detection Prediction Example")
    print("=" * 60)

    # Check if model exists
    checkpoint_path = Path('checkpoints/best_model.pth')
    if not checkpoint_path.exists():
        print("\nError: No trained model found!")
        print("Please run 'train_model.py' first to train a model.")
        return

    # Step 1: Load model
    print("\n1. Loading trained model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = VoidDetectorMLP(
        input_dim=103,
        hidden_dims=(256, 128, 64),
        dropout_rate=0.3,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"   Model loaded from {checkpoint_path}")
    print(f"   Using device: {device}")

    # Step 2: Generate new test data
    print("\n2. Generating new test galaxies...")
    positions, true_labels, neighbor_distances = generate_sample_data(
        n_galaxies=1000,
        n_neighbors=100,
        void_fraction=0.3,
        random_seed=123,  # Different seed for new data
    )
    print(f"   Generated {len(true_labels)} galaxies")

    # Step 3: Prepare data
    print("\n3. Preparing data for prediction...")
    # Note: In a real scenario, you'd need to use the same scaler from training
    dataset = GalaxyDataset(
        positions=positions,
        labels=true_labels,
        neighbor_distances=neighbor_distances,
        normalize=True,
    )

    # Step 4: Make predictions
    print("\n4. Making predictions...")
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for i in range(len(dataset)):
            features, _ = dataset[i]
            features = features.unsqueeze(0).to(device)

            output = model(features)
            prob = output.item()
            pred = 1 if prob > 0.5 else 0

            all_predictions.append(pred)
            all_probabilities.append(prob)

    predictions = np.array(all_predictions)
    probabilities = np.array(all_probabilities)

    # Step 5: Calculate accuracy
    print("\n5. Evaluating predictions...")
    accuracy = (predictions == true_labels).mean()
    void_predictions = predictions.sum()
    true_voids = true_labels.sum()

    print(f"\n   Accuracy: {accuracy:.4f}")
    print(f"   Predicted voids: {void_predictions} ({void_predictions/len(predictions)*100:.1f}%)")
    print(f"   True voids: {true_voids} ({true_voids/len(true_labels)*100:.1f}%)")

    # Step 6: Visualize predictions
    print("\n6. Creating visualization...")
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    plot_predictions_3d(
        positions,
        true_labels,
        predictions,
        save_path=output_dir / 'predictions_3d.png',
    )

    print(f"   Visualization saved to {output_dir}/predictions_3d.png")

    # Step 7: Show some example predictions
    print("\n7. Example predictions:")
    print("\n   Galaxy | True | Predicted | Probability")
    print("   " + "-" * 45)
    for i in range(min(10, len(predictions))):
        true_label = "Void" if true_labels[i] == 1 else "Not void"
        pred_label = "Void" if predictions[i] == 1 else "Not void"
        print(f"   {i:4d}   | {true_label:8s} | {pred_label:9s} | {probabilities[i]:.4f}")

    print("\n" + "=" * 60)
    print("Prediction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
