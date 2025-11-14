# Voronoi-based Void Detection: MLP vs GNN

## Overview

This document explains the implementation of Voronoi tessellation-based features for void detection in galaxy distributions, addressing spatial data leakage concerns and comparing MLP vs GNN approaches.

## The Problem: Spatial Data Leakage

When using raw galaxy positions as features for machine learning models, there's a risk of **spatial data leakage**:

- The model can "memorize" specific galaxy positions from the training set
- If train/validation/test splits are done randomly (not spatially), the model performs well
- But if splits are done spatially (e.g., different regions of the simulation box), performance degrades significantly
- This means the model hasn't learned true void characteristics, just position patterns

## The Solution: Voronoi Tessellation Features

Instead of using raw positions, we use features derived from **Voronoi tessellation**:

### What is Voronoi Tessellation?

Voronoi tessellation divides space into cells, where each cell contains all points closer to one galaxy than to any other. This captures the **local density structure** around each galaxy.

### Key Features Extracted

1. **Cell Volume**: Large volumes indicate sparse regions (potential voids)
2. **Neighbor Count**: Number of adjacent Voronoi cells (topological property)
3. **Adjacency Graph**: Which cells share boundaries (for GNN)

### Why This Helps

- Features are **topological** rather than positional
- They capture **relative** structure, not absolute positions
- More invariant to spatial transformations
- Better generalization across different regions

## Implementation

### 1. Voronoi Feature Extraction

```python
from voidx.voronoi import VoronoiFeatureExtractor

# Create extractor
extractor = VoronoiFeatureExtractor(
    box_size=250.0,           # Simulation box size
    use_periodic=True,        # Periodic boundary conditions
    clip_infinite=True,       # Handle infinite cells
)

# Extract features from galaxy positions
features = extractor.extract_features(positions)

# Available features:
# - features['volumes']: Cell volumes
# - features['normalized_volumes']: Log-normalized volumes
# - features['neighbor_count']: Number of neighbors per cell
# - features['edge_index']: Adjacency graph (for GNN)
```

### 2. MLP Approach

Use only topological features (no positions):

```python
from voidx.models import VoidMLP

# Create MLP features (volume + neighbor_count only)
mlp_features = extractor.create_mlp_features(
    positions, 
    include_positions=False  # Avoid spatial leakage!
)

# Train MLP
model = VoidMLP(
    in_features=2,  # volume + neighbor_count
    hidden_layers=(128, 64, 32),
    dropout=0.3,
)
```

**Pros:**
- Simple architecture
- Fast training
- Good baseline performance
- No spatial leakage with topological features

**Cons:**
- Limited to local properties
- Can't capture multi-scale structure
- No information about neighbor relationships

### 3. GNN Approach

Use graph structure to capture relationships:

```python
from voidx.models import VoronoiGCN, VoronoiGAT, VoronoiSAGE

# Create GNN features (volume + neighbor_count + positions)
gnn_features = np.hstack([
    features['normalized_volumes'][:, np.newaxis],
    features['neighbor_count'][:, np.newaxis],
    positions,  # OK in GNN due to graph context!
])

# Train GNN
model = VoronoiGCN(
    in_features=5,  # volume + neighbor_count + xyz
    hidden_channels=64,
    num_layers=3,
    dropout=0.3,
)

# Forward pass
logits = model(
    x=node_features,          # Node features
    edge_index=edge_index,    # Adjacency graph
)
```

**Why positions are OK in GNN:**
- Positions are used **in the context of the graph structure**
- The model learns from **relationships** between neighbors
- Not just memorizing absolute positions
- The graph provides local coordinate frame

**Pros:**
- Captures multi-hop relationships
- Can learn void boundaries (rapid changes in properties)
- Leverages spatial structure without leakage
- Better at complex patterns

**Cons:**
- More complex architecture
- Slower training
- Requires torch_geometric
- Need more hyperparameter tuning

### 4. Available GNN Models

#### VoronoiGCN (Graph Convolutional Network)
```python
model = VoronoiGCN(
    in_features=5,
    hidden_channels=64,
    num_layers=3,
)
```
- Standard GCN layers
- Good for smooth propagation of information
- Efficient and well-understood

#### VoronoiGAT (Graph Attention Network)
```python
model = VoronoiGAT(
    in_features=5,
    hidden_channels=64,
    num_layers=3,
    num_heads=4,  # Attention heads
)
```
- Learns to weight neighbor importance
- Good for identifying boundaries
- More expressive than GCN

#### VoronoiSAGE (GraphSAGE)
```python
model = VoronoiSAGE(
    in_features=5,
    hidden_channels=64,
    num_layers=3,
    aggregator='mean',
)
```
- Sampling-based approach
- Better for large graphs
- Good inductive learning

## Usage Example

See `examples/compare_mlp_gnn_voronoi.py` for a complete example:

```bash
cd /path/to/VoidX-voronoi
python examples/compare_mlp_gnn_voronoi.py
```

This script:
1. Loads or generates galaxy data
2. Computes Voronoi tessellation
3. Trains MLP on topological features
4. Trains multiple GNN models
5. Compares performance

## Key Insights

### Addressing Data Leakage

1. **Without Voronoi (raw positions)**:
   - Random split: High accuracy ✓
   - Spatial split: Low accuracy ✗
   - **Problem**: Model memorizes positions

2. **With Voronoi features (MLP)**:
   - Random split: Good accuracy ✓
   - Spatial split: Good accuracy ✓
   - **Better**: Uses topological features

3. **With Voronoi features (GNN)**:
   - Random split: Best accuracy ✓✓
   - Spatial split: Best accuracy ✓✓
   - **Best**: Combines topology with structure

### When to Use Each Approach

**Use MLP when:**
- You want a simple, fast baseline
- Interpretability is important
- Training resources are limited
- Local cell properties are sufficient

**Use GNN when:**
- You need to capture multi-scale structure
- Void boundaries are complex
- You have computational resources
- You want state-of-the-art performance

### Recommended Workflow

1. **Start with MLP**:
   - Establishes baseline performance
   - Validates Voronoi features work
   - Fast iteration on data processing

2. **Try GNN if needed**:
   - If MLP performance is insufficient
   - If you need to model complex patterns
   - If you have the resources

3. **Test spatial generalization**:
   - Always validate with spatial splits
   - Test on different simulation boxes
   - Check performance on real data

## Installation

### Basic requirements:
```bash
pip install torch numpy scipy matplotlib scikit-learn
```

### For GNN support:
```bash
pip install torch-geometric
```

Or install everything:
```bash
pip install -r requirements.txt
```

## API Reference

### VoronoiFeatureExtractor

```python
VoronoiFeatureExtractor(
    box_size: Optional[float] = None,
    use_periodic: bool = False,
    clip_infinite: bool = True,
)
```

Main methods:
- `extract_features(positions)`: Extract all features
- `create_mlp_features(positions, include_positions=False)`: Create MLP-ready features
- `compute_voronoi(positions)`: Compute Voronoi tessellation

### GNN Models

All models share similar interface:

```python
model = VoronoiGCN/GAT/SAGE(
    in_features: int,          # Number of input features
    hidden_channels: int,      # Hidden dimension
    num_layers: int,          # Number of layers
    dropout: float,           # Dropout rate
    use_batchnorm: bool,      # Use batch normalization
)

# Forward pass
logits = model(
    x: torch.Tensor,           # Node features (N, D)
    edge_index: torch.Tensor,  # Edges (2, E)
    batch: Optional[torch.Tensor] = None,  # Batch indices
)
```

## Testing

Run tests:
```bash
python tests/test_voronoi.py
```

## Future Improvements

1. **Periodic Voronoi**: Better handling of periodic boundaries
2. **Additional features**: Cell shape, eccentricity, orientation
3. **Graph pooling**: Hierarchical void detection
4. **Hybrid models**: Combine MLP and GNN
5. **Uncertainty quantification**: Bayesian GNN

## References

- Voronoi tessellation: https://en.wikipedia.org/wiki/Voronoi_diagram
- Graph Neural Networks: https://pytorch-geometric.readthedocs.io/
- Cosmic voids: https://en.wikipedia.org/wiki/Void_(astronomy)

## Questions and Answers

**Q: Can I still include positions in the MLP?**
A: You can, but it may cause spatial leakage. Test with spatial splits to verify.

**Q: Why are positions OK in GNN but not MLP?**
A: In GNN, positions are used within graph context. The model learns from neighbor relationships, not absolute positions.

**Q: Which GNN model is best?**
A: Try GCN first (simplest). Use GAT if boundaries are important. Use SAGE for very large datasets.

**Q: How do I test for spatial leakage?**
A: Split your data spatially (e.g., by box quadrants) instead of randomly, and compare performance.

**Q: Can I use this on observational data?**
A: Yes! Just make sure to handle edge effects and sky boundaries appropriately.
