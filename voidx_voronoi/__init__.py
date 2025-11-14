"""
VoidX: A machine learning package for detecting voids in galaxy distributions.

Lightweight package init: avoid importing heavy third-party libraries at import time.
Import specific symbols from submodules when needed in user code, e.g.:
	from voidx.data import GalaxyDataset, split_indices, normalize_features
	from voidx.models import MLP, VoidMLP
	from voidx.utils import evaluate_model, convert_to_Cartesian
	from voidx.config import get_config, TrainingConfig, DataConfig, NotebookConfig
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
