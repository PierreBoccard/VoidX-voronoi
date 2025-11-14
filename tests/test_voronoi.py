"""
Simple test to verify Voronoi feature extraction works.
"""

import numpy as np
from voidx.voronoi import VoronoiFeatureExtractor, compute_voronoi_features

def test_voronoi_features():
    """Test basic Voronoi feature extraction."""
    print("Testing Voronoi feature extraction...")
    
    # Create simple test data
    np.random.seed(42)
    n_points = 100
    positions = np.random.uniform(0, 100, size=(n_points, 3))
    
    # Test feature extraction
    extractor = VoronoiFeatureExtractor(box_size=100.0, use_periodic=False)
    features_dict = extractor.extract_features(positions)
    
    # Verify output structure
    assert 'positions' in features_dict
    assert 'volumes' in features_dict
    assert 'edge_index' in features_dict
    assert 'neighbor_count' in features_dict
    assert 'normalized_volumes' in features_dict
    
    # Check shapes
    assert features_dict['positions'].shape == (n_points, 3)
    assert features_dict['volumes'].shape == (n_points,)
    assert features_dict['edge_index'].shape[0] == 2
    assert features_dict['neighbor_count'].shape == (n_points,)
    
    print(f"✓ Basic feature extraction passed")
    print(f"  - {n_points} points processed")
    print(f"  - {features_dict['edge_index'].shape[1]} edges found")
    print(f"  - Average neighbors: {features_dict['neighbor_count'].mean():.2f}")
    
    # Test MLP feature creation
    mlp_features = extractor.create_mlp_features(positions, include_positions=False)
    assert mlp_features.shape == (n_points, 2)  # volume + neighbor_count
    print(f"✓ MLP features created: shape {mlp_features.shape}")
    
    mlp_features_with_pos = extractor.create_mlp_features(positions, include_positions=True)
    assert mlp_features_with_pos.shape == (n_points, 5)  # volume + neighbor_count + 3 positions
    print(f"✓ MLP features with positions: shape {mlp_features_with_pos.shape}")
    
    # Test convenience function
    features_mlp = compute_voronoi_features(positions, for_mlp=True)
    assert 'features' in features_mlp
    print(f"✓ Convenience function works")
    
    print("\n✅ All Voronoi tests passed!")
    return True

if __name__ == "__main__":
    test_voronoi_features()
