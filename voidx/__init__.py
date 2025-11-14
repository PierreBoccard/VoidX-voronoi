"""
VoidX: A machine learning package for detecting voids in galaxy distributions.

Lightweight package init: avoid importing heavy third-party libraries at import time.
Import specific symbols from submodules when needed in user code, e.g.:
	from voidx.data import GalaxyDataset, split_indices, normalize_features
	from voidx.models import MLP, VoidMLP
	from voidx.utils import evaluate_model, convert_to_Cartesian
	from voidx.config import get_config, TrainingConfig, DataConfig, NotebookConfig
	from voidx.voronoi import VoronoiFeatureExtractor, compute_voronoi_features
"""

__version__ = "0.1.0"
__author__ = "Pierre Boccard"

# Re-export a minimal public API from internal modules without pulling in heavy deps
from .data import GalaxyDataset, split_indices, normalize_features  # noqa: F401
from .models import MLP, VoidMLP  # noqa: F401
from .utils import evaluate_model, convert_to_Cartesian  # noqa: F401
from .config import (
		get_config,
		TrainingConfig,
		DataConfig,
		NotebookConfig,
		setup_paths,
		setup_device_and_seed,
)  # noqa: F401

# Voronoi and GNN support (optional - requires scipy and torch_geometric)
try:
	from .voronoi import VoronoiFeatureExtractor, compute_voronoi_features  # noqa: F401
	_voronoi_available = True
except ImportError:
	_voronoi_available = False

try:
	from .models import VoronoiGCN, VoronoiGAT, VoronoiSAGE  # noqa: F401
	from .data import VoronoiGraphDataset  # noqa: F401
	_gnn_available = True
except ImportError:
	_gnn_available = False

__all__ = [
		"GalaxyDataset",
		"split_indices",
		"normalize_features",
		"MLP",
		"VoidMLP",
		"evaluate_model",
		"convert_to_Cartesian",
		"get_config",
		"TrainingConfig",
		"DataConfig",
		"NotebookConfig",
        "GalaxyClusteringAnalyzer",
		"setup_paths",
		"setup_device_and_seed",
		"__version__",
		"__author__",
]

# Add Voronoi/GNN exports if available
if _voronoi_available:
	__all__.extend(["VoronoiFeatureExtractor", "compute_voronoi_features"])
if _gnn_available:
	__all__.extend(["VoronoiGCN", "VoronoiGAT", "VoronoiSAGE", "VoronoiGraphDataset"])
