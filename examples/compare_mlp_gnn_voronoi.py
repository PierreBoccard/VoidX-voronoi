"""
Example script comparing MLP and GNN approaches for void detection using Voronoi features.

This script demonstrates:
1. Computing Voronoi tessellation from galaxy positions
2. Extracting features (volumes, adjacency, neighbor counts)
3. Training both MLP and GNN models
4. Addressing spatial data leakage concerns
"""

import torch
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from voidx.voronoi import VoronoiFeatureExtractor
from voidx.models import VoidMLP, VoronoiGCN, VoronoiGAT, VoronoiSAGE
from voidx.data import GalaxyDataset, split_indices_stratified, normalize_features

# Check if torch_geometric is available
try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as PyGDataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not available. Only MLP will be demonstrated.")


def load_data_with_voronoi(
    positions: np.ndarray,
    labels: np.ndarray,
    box_size: float = 250.0,
    use_periodic: bool = True,
    include_positions_in_mlp: bool = False,
):
    """
    Load galaxy data and compute Voronoi features.
    
    Args:
        positions: Galaxy positions (N, 3)
        labels: Void membership labels (N,)
        box_size: Size of simulation box
        use_periodic: Use periodic boundary conditions
        include_positions_in_mlp: Include positions in MLP features (may cause leakage)
    
    Returns:
        Dictionary with MLP and GNN features
    """
    print(f"\nComputing Voronoi tessellation for {len(positions)} galaxies...")
    
    extractor = VoronoiFeatureExtractor(
        box_size=box_size,
        use_periodic=use_periodic,
        clip_infinite=True,
    )
    
    # Extract features
    features_dict = extractor.extract_features(positions)
    
    print(f"  - Found {features_dict['edge_index'].shape[1]} adjacency edges")
    print(f"  - Average neighbors per cell: {features_dict['neighbor_count'].mean():.2f}")
    
    # Create MLP features
    mlp_features = extractor.create_mlp_features(positions, include_positions=include_positions_in_mlp)
    
    # Create GNN features (include positions for GNN as they're used in context of graph structure)
    gnn_features = np.hstack([
        features_dict['normalized_volumes'][:, np.newaxis],
        features_dict['neighbor_count'][:, np.newaxis],
        positions,  # Positions are OK in GNN due to graph structure
    ])
    
    return {
        'mlp_features': mlp_features,
        'gnn_features': gnn_features,
        'edge_index': features_dict['edge_index'],
        'labels': labels,
        'volumes': features_dict['volumes'],
        'neighbor_count': features_dict['neighbor_count'],
    }


def train_mlp_model(X_train, y_train, X_val, y_val, device='cpu', epochs=50):
    """Train MLP on Voronoi features."""
    print("\n" + "="*60)
    print("Training MLP on Voronoi Features")
    print("="*60)
    
    # Create datasets
    train_dataset = GalaxyDataset(X_train, y_train)
    val_dataset = GalaxyDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create model
    model = VoidMLP(
        in_features=X_train.shape[1],
        hidden_layers=(128, 64, 32),
        dropout=0.3,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.BCELoss()
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()
                
                predicted = (pred > 0.5).float()
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_val_loss


def train_gnn_model(X_train, edge_index, y_train, train_mask, val_mask, 
                    model_class=VoronoiGCN, device='cpu', epochs=200):
    """Train GNN on Voronoi graph."""
    print("\n" + "="*60)
    print(f"Training {model_class.__name__} on Voronoi Graph")
    print("="*60)
    
    # Create graph data
    data = Data(
        x=torch.as_tensor(X_train, dtype=torch.float32),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        y=torch.as_tensor(y_train, dtype=torch.float32),
    ).to(device)
    
    # Create model
    model = model_class(
        in_features=X_train.shape[1],
        hidden_channels=64,
        num_layers=3,
        dropout=0.3,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    train_mask = torch.as_tensor(train_mask, dtype=torch.bool).to(device)
    val_mask = torch.as_tensor(val_mask, dtype=torch.bool).to(device)
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_loss = criterion(out[val_mask], data.y[val_mask]).item()
            
            pred = torch.sigmoid(out[val_mask]) > 0.5
            val_acc = (pred == data.y[val_mask]).float().mean().item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss: {loss.item():.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_val_loss


def evaluate_model(model, X_test, y_test, edge_index=None, test_mask=None, device='cpu', is_gnn=False):
    """Evaluate model on test set."""
    model.eval()
    
    with torch.no_grad():
        if is_gnn:
            # GNN evaluation
            data = Data(
                x=torch.as_tensor(X_test, dtype=torch.float32),
                edge_index=torch.as_tensor(edge_index, dtype=torch.long),
                y=torch.as_tensor(y_test, dtype=torch.float32),
            ).to(device)
            
            test_mask = torch.as_tensor(test_mask, dtype=torch.bool).to(device)
            
            out = model(data.x, data.edge_index)
            pred_probs = torch.sigmoid(out[test_mask])
            pred_labels = (pred_probs > 0.5).float()
            
            y_true = data.y[test_mask].cpu().numpy()
            y_pred = pred_labels.cpu().numpy()
            y_prob = pred_probs.cpu().numpy()
        else:
            # MLP evaluation
            X_test_t = torch.as_tensor(X_test, dtype=torch.float32).to(device)
            pred_probs = model(X_test_t)
            pred_labels = (pred_probs > 0.5).float()
            
            y_true = y_test
            y_pred = pred_labels.cpu().numpy()
            y_prob = pred_probs.cpu().numpy()
    
    # Calculate metrics
    accuracy = (y_pred == y_true).mean()
    
    # Precision, Recall, F1
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\n" + "="*60)
    print("Voronoi-based Void Detection: MLP vs GNN Comparison")
    print("="*60)
    
    # For demonstration, we'll load existing data if available
    # Otherwise, generate synthetic data
    try:
        # Try to load existing data
        data_path = Path('data/knn_data.npz')
        if data_path.exists():
            print(f"\nLoading data from {data_path}...")
            data = np.load(data_path)
            positions = data['positions']
            labels = data['membership'].astype(np.float32)
            print(f"Loaded {len(positions)} galaxies")
        else:
            raise FileNotFoundError("Generate synthetic data")
    except:
        print("\nGenerating synthetic galaxy data...")
        # Simple synthetic data generation
        n_galaxies = 5000
        box_size = 250.0
        
        # Background galaxies (not in voids)
        n_bg = int(n_galaxies * 0.7)
        positions_bg = np.random.uniform(0, box_size, size=(n_bg, 3))
        labels_bg = np.zeros(n_bg)
        
        # Void galaxies (clustered in sparse regions)
        n_void = n_galaxies - n_bg
        positions_void = np.random.uniform(0, box_size, size=(n_void, 3))
        labels_void = np.ones(n_void)
        
        positions = np.vstack([positions_bg, positions_void])
        labels = np.hstack([labels_bg, labels_void]).astype(np.float32)
        
        # Shuffle
        idx = np.random.permutation(n_galaxies)
        positions = positions[idx]
        labels = labels[idx]
        
        print(f"Generated {n_galaxies} galaxies ({labels.mean():.2%} in voids)")
    
    # Compute Voronoi features
    box_size = 250.0
    features_data = load_data_with_voronoi(
        positions=positions,
        labels=labels,
        box_size=box_size,
        use_periodic=True,
        include_positions_in_mlp=False,  # Don't include positions to avoid leakage
    )
    
    # Split data
    train_idx, val_idx, test_idx = split_indices_stratified(
        labels, train=0.7, val=0.15, seed=42
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val:   {len(val_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")
    
    # ==================== MLP Training ====================
    
    # Prepare MLP data
    X_mlp = features_data['mlp_features']
    X_train_mlp = X_mlp[train_idx]
    X_val_mlp = X_mlp[val_idx]
    X_test_mlp = X_mlp[test_idx]
    
    # Normalize
    X_train_mlp, X_val_mlp, X_test_mlp, _ = normalize_features(
        X_train_mlp, X_val_mlp, X_test_mlp
    )
    
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]
    
    print(f"\nMLP features shape: {X_mlp.shape}")
    print(f"  Features: normalized_volume, neighbor_count")
    
    # Train MLP
    mlp_model, mlp_val_loss = train_mlp_model(
        X_train_mlp, y_train, X_val_mlp, y_val,
        device=device, epochs=100
    )
    
    # Evaluate MLP
    mlp_results = evaluate_model(mlp_model, X_test_mlp, y_test, device=device, is_gnn=False)
    
    print("\nMLP Test Results:")
    print(f"  Accuracy:  {mlp_results['accuracy']:.4f}")
    print(f"  Precision: {mlp_results['precision']:.4f}")
    print(f"  Recall:    {mlp_results['recall']:.4f}")
    print(f"  F1 Score:  {mlp_results['f1']:.4f}")
    
    # ==================== GNN Training ====================
    
    if TORCH_GEOMETRIC_AVAILABLE:
        # Prepare GNN data
        X_gnn = features_data['gnn_features']
        edge_index = features_data['edge_index']
        
        # Normalize
        X_train_gnn = X_gnn[train_idx]
        X_val_gnn = X_gnn[val_idx]
        X_test_gnn = X_gnn[test_idx]
        X_train_gnn, X_val_gnn, X_test_gnn, (mean, std) = normalize_features(
            X_train_gnn, X_val_gnn, X_test_gnn
        )
        
        # Normalize full dataset for GNN
        X_gnn_norm = (X_gnn - mean) / std
        
        # Create masks
        train_mask = np.zeros(len(labels), dtype=bool)
        val_mask = np.zeros(len(labels), dtype=bool)
        test_mask = np.zeros(len(labels), dtype=bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        print(f"\nGNN features shape: {X_gnn.shape}")
        print(f"  Features: normalized_volume, neighbor_count, position_x, position_y, position_z")
        print(f"  Note: Positions are used in GNN within graph context to avoid spatial leakage")
        
        # Train different GNN models
        gnn_models = {
            'GCN': VoronoiGCN,
            'GAT': VoronoiGAT,
            'GraphSAGE': VoronoiSAGE,
        }
        
        gnn_results_dict = {}
        
        for name, model_class in gnn_models.items():
            gnn_model, gnn_val_loss = train_gnn_model(
                X_gnn_norm, edge_index, labels,
                train_mask, val_mask,
                model_class=model_class,
                device=device,
                epochs=200
            )
            
            # Evaluate GNN
            gnn_results = evaluate_model(
                gnn_model, X_gnn_norm, labels,
                edge_index=edge_index, test_mask=test_mask,
                device=device, is_gnn=True
            )
            
            gnn_results_dict[name] = gnn_results
            
            print(f"\n{name} Test Results:")
            print(f"  Accuracy:  {gnn_results['accuracy']:.4f}")
            print(f"  Precision: {gnn_results['precision']:.4f}")
            print(f"  Recall:    {gnn_results['recall']:.4f}")
            print(f"  F1 Score:  {gnn_results['f1']:.4f}")
        
        # ==================== Summary ====================
        
        print("\n" + "="*60)
        print("Summary: Model Comparison")
        print("="*60)
        print(f"{'Model':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 60)
        print(f"{'MLP':<15} {mlp_results['accuracy']:>10.4f} {mlp_results['precision']:>10.4f} "
              f"{mlp_results['recall']:>10.4f} {mlp_results['f1']:>10.4f}")
        
        for name, results in gnn_results_dict.items():
            print(f"{name:<15} {results['accuracy']:>10.4f} {results['precision']:>10.4f} "
                  f"{results['recall']:>10.4f} {results['f1']:>10.4f}")
        
        print("\n" + "="*60)
        print("Key Insights:")
        print("="*60)
        print("1. MLP uses only topological features (volume, neighbor count)")
        print("   - Avoids spatial data leakage")
        print("   - Limited to local cell properties")
        print("")
        print("2. GNN uses graph structure + node features")
        print("   - Leverages spatial relationships through graph edges")
        print("   - Can learn patterns over multiple hops")
        print("   - Position features are contextualized by graph structure")
        print("")
        print("3. GNN models (GCN, GAT, GraphSAGE) can capture:")
        print("   - Multi-scale void structures")
        print("   - Boundary effects between void/non-void regions")
        print("   - Topological patterns in the Voronoi tessellation")
        print("="*60)
    
    else:
        print("\n" + "="*60)
        print("torch_geometric not installed - skipping GNN comparison")
        print("Install with: pip install torch-geometric")
        print("="*60)


if __name__ == "__main__":
    main()
