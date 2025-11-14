"""
Voronoi tessellation and feature extraction for void detection.

This module provides tools to compute Voronoi cells from galaxy positions
and extract features like cell positions (centroids), volumes, and neighbor
adjacency for use in machine learning models (both MLP and GNN).
"""

import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from typing import Tuple, Optional, List, Dict
import warnings


class VoronoiFeatureExtractor:
    """
    Extract features from Voronoi tessellation of galaxy positions.
    
    This helps address data leakage concerns by using topological features
    (volumes, neighbor relationships) rather than raw positions.
    
    Args:
        box_size: Size of the periodic box (for periodic boundary conditions)
        use_periodic: Whether to use periodic boundary conditions
        clip_infinite: Whether to clip infinite Voronoi cells to box boundaries
    """
    
    def __init__(
        self,
        box_size: Optional[float] = None,
        use_periodic: bool = False,
        clip_infinite: bool = True,
    ):
        self.box_size = box_size
        self.use_periodic = use_periodic
        self.clip_infinite = clip_infinite
        
    def compute_voronoi(
        self,
        positions: np.ndarray,
    ) -> Voronoi:
        """
        Compute Voronoi tessellation from galaxy positions.
        
        Args:
            positions: Galaxy positions, shape (N, 3)
            
        Returns:
            scipy.spatial.Voronoi object
        """
        if self.use_periodic and self.box_size is not None:
            # For periodic boundaries, mirror the points at boundaries
            positions = self._prepare_periodic_points(positions)
        
        return Voronoi(positions)
    
    def _prepare_periodic_points(self, positions: np.ndarray) -> np.ndarray:
        """
        Prepare positions for periodic Voronoi by mirroring points near boundaries.
        
        This is a simplified approach - for production use, consider more
        sophisticated periodic Voronoi algorithms.
        """
        # Mirror points within a threshold distance from each boundary
        box = self.box_size
        threshold = box * 0.1  # Mirror points within 10% of boundary
        
        mirrors = [positions]
        for dim in range(3):
            for shift in [-box, box]:
                # Find points near this boundary
                if shift < 0:
                    mask = positions[:, dim] < threshold
                else:
                    mask = positions[:, dim] > (box - threshold)
                
                if np.any(mask):
                    mirrored = positions[mask].copy()
                    mirrored[:, dim] += shift
                    mirrors.append(mirrored)
        
        return np.vstack(mirrors)
    
    def compute_cell_volumes(
        self,
        vor: Voronoi,
        n_original_points: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute volumes of Voronoi cells.
        
        Args:
            vor: Voronoi tessellation object
            n_original_points: Number of original points (before mirroring for periodic)
            
        Returns:
            Array of cell volumes, shape (N,)
        """
        if n_original_points is None:
            n_original_points = len(vor.points)
        
        volumes = np.zeros(n_original_points)
        
        for i in range(n_original_points):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if -1 in region:
                # Infinite region
                if self.clip_infinite and self.box_size is not None:
                    # Estimate volume by clipping to box
                    volumes[i] = self._estimate_clipped_volume(vor, region)
                else:
                    # Use a large default value or NaN
                    volumes[i] = np.nan
            else:
                # Finite region - compute convex hull volume
                if len(region) >= 4:  # Need at least 4 points for 3D volume
                    try:
                        vertices = vor.vertices[region]
                        hull = ConvexHull(vertices)
                        volumes[i] = hull.volume
                    except:
                        volumes[i] = np.nan
                else:
                    volumes[i] = np.nan
        
        return volumes
    
    def _estimate_clipped_volume(self, vor: Voronoi, region: List[int]) -> float:
        """
        Estimate volume of an infinite Voronoi region by clipping to box.
        
        This is a simplified heuristic. For better accuracy, use proper
        clipping algorithms.
        """
        # Get finite vertices
        finite_vertices = [v for v in region if v != -1]
        
        if len(finite_vertices) < 3:
            return np.nan
        
        vertices = vor.vertices[finite_vertices]
        
        # Clip vertices to box
        box = self.box_size
        vertices = np.clip(vertices, 0, box)
        
        # Try to estimate volume
        try:
            # Add box corners if needed
            if len(vertices) < 4:
                return np.nan
            hull = ConvexHull(vertices)
            return hull.volume
        except:
            return np.nan
    
    def compute_adjacency(
        self,
        vor: Voronoi,
        n_original_points: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute adjacency graph from Voronoi tessellation.
        
        Two cells are adjacent if they share a face (ridge).
        
        Args:
            vor: Voronoi tessellation object
            n_original_points: Number of original points (before mirroring)
            
        Returns:
            Tuple of (edge_index, edge_count):
            - edge_index: Array of shape (2, E) with source and target node indices
            - edge_count: Number of neighbors for each node, shape (N,)
        """
        if n_original_points is None:
            n_original_points = len(vor.points)
        
        # Build adjacency list
        adjacency_dict = {i: set() for i in range(n_original_points)}
        
        for ridge_points in vor.ridge_points:
            p1, p2 = ridge_points
            # Only consider edges within original points
            if p1 < n_original_points and p2 < n_original_points:
                adjacency_dict[p1].add(p2)
                adjacency_dict[p2].add(p1)
        
        # Convert to edge list format (for GNN)
        edge_list = []
        for node, neighbors in adjacency_dict.items():
            for neighbor in neighbors:
                edge_list.append([node, neighbor])
        
        edge_index = np.array(edge_list, dtype=np.int64).T if edge_list else np.zeros((2, 0), dtype=np.int64)
        
        # Count neighbors for each node
        edge_count = np.array([len(adjacency_dict[i]) for i in range(n_original_points)])
        
        return edge_index, edge_count
    
    def extract_features(
        self,
        positions: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Extract all Voronoi features from galaxy positions.
        
        Args:
            positions: Galaxy positions, shape (N, 3)
            
        Returns:
            Dictionary with keys:
            - 'positions': Original positions (N, 3)
            - 'volumes': Cell volumes (N,)
            - 'edge_index': Adjacency graph (2, E)
            - 'neighbor_count': Number of neighbors per cell (N,)
            - 'normalized_volumes': Log-normalized volumes (N,)
        """
        n_points = len(positions)
        
        # Compute Voronoi tessellation
        vor = self.compute_voronoi(positions)
        
        # Extract volumes
        volumes = self.compute_cell_volumes(vor, n_original_points=n_points)
        
        # Extract adjacency
        edge_index, neighbor_count = self.compute_adjacency(vor, n_original_points=n_points)
        
        # Normalize volumes (log transform to handle large range)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Replace NaN volumes with median for normalization
            finite_volumes = volumes[~np.isnan(volumes)]
            if len(finite_volumes) > 0:
                median_volume = np.median(finite_volumes)
                volumes_filled = np.where(np.isnan(volumes), median_volume, volumes)
                # Log transform (add small epsilon to avoid log(0))
                normalized_volumes = np.log10(volumes_filled + 1e-10)
            else:
                normalized_volumes = np.zeros_like(volumes)
        
        return {
            'positions': positions,
            'volumes': volumes,
            'edge_index': edge_index,
            'neighbor_count': neighbor_count,
            'normalized_volumes': normalized_volumes,
        }
    
    def create_mlp_features(
        self,
        positions: np.ndarray,
        include_positions: bool = False,
    ) -> np.ndarray:
        """
        Create feature matrix for MLP from Voronoi tessellation.
        
        Features include volumes and neighbor counts. Optionally include positions.
        
        Args:
            positions: Galaxy positions, shape (N, 3)
            include_positions: Whether to include raw positions (may cause leakage)
            
        Returns:
            Feature matrix, shape (N, D) where D depends on include_positions
        """
        features_dict = self.extract_features(positions)
        
        features_list = [
            features_dict['normalized_volumes'][:, np.newaxis],  # (N, 1)
            features_dict['neighbor_count'][:, np.newaxis],      # (N, 1)
        ]
        
        if include_positions:
            features_list.append(features_dict['positions'])  # (N, 3)
        
        return np.hstack(features_list)


def compute_voronoi_features(
    positions: np.ndarray,
    box_size: Optional[float] = None,
    use_periodic: bool = False,
    for_mlp: bool = True,
    include_positions: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Convenience function to compute Voronoi features.
    
    Args:
        positions: Galaxy positions, shape (N, 3)
        box_size: Size of periodic box (if use_periodic=True)
        use_periodic: Whether to use periodic boundary conditions
        for_mlp: If True, return MLP-ready features; if False, return dict for GNN
        include_positions: Whether to include positions in MLP features
        
    Returns:
        If for_mlp=True: Dictionary with 'features' key containing (N, D) array
        If for_mlp=False: Dictionary with full Voronoi features for GNN
    """
    extractor = VoronoiFeatureExtractor(
        box_size=box_size,
        use_periodic=use_periodic,
    )
    
    if for_mlp:
        features = extractor.create_mlp_features(positions, include_positions)
        return {'features': features}
    else:
        return extractor.extract_features(positions)
