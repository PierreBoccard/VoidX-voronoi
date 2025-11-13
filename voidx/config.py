"""
Configuration module for VoidX.

Contains default configurations for models, training, and data processing.
This module also provides notebook configuration to avoid repetition across notebooks.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import os
import json
import numpy as np
import torch


def setup_paths(name, base_dir=None, param=None, create_dirs=True) -> Dict[str, Path]:
    """Set up directory paths for data, checkpoints, plots, and results.
    
    Args:
        name: Name of the dataset/experiment
        base_dir: Base directory for data (optional, uses defaults if not provided)
        create_dirs: Whether to create the directories if they don't exist
    Returns:
        Dictionary with keys: data_dir, checkpoint_dir_spec, checkpoint_dir_global,
        plot_dir, result_dir
    """
    if base_dir is None:
        base_dir = Path('/Users/boccardpierre/Documents/PhD/Research/Code/VoidX/data')
    else:
        base_dir = Path(base_dir)
    
    data_dir = base_dir / name

    param_dir_name = str(param) if param else "default"

    param_dir = data_dir / param_dir_name
    param_dir.mkdir(parents=True, exist_ok=True)
    
    # Create galaxy info files directory
    galaxy_info_dir = param_dir / 'galaxy_info_files'
    galaxy_info_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoint directories
    checkpoint_dir_spec = param_dir / 'checkpoints'
    checkpoint_dir_spec.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir_global = base_dir / 'checkpoints'
    checkpoint_dir_global.mkdir(parents=True, exist_ok=True)
    
    # Create plot directory
    plot_dir = param_dir / 'plot'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Create result directory
    result_dir = param_dir / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'data_dir': data_dir,
        'param_dir': param_dir,
        'galaxy_info_dir': galaxy_info_dir,
        'checkpoint_dir_spec': checkpoint_dir_spec,
        'checkpoint_dir_global': checkpoint_dir_global,
        'plot_dir': plot_dir,
        'result_dir': result_dir
    }


def setup_device_and_seed(device='cuda', seed=42):
    """Configure computation device and random seed for reproducibility.
    
    Args:
        device: Device to use ('cpu' or 'cuda')
        seed: Random seed for reproducibility
    
    Returns:
        The device string that was configured
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    return device


# --- Global settings loader (path + name + toggles) ---
def _load_global_settings() -> Dict[str, Any]:
    """
    Load global settings once for all notebooks from config/global.json (or global.json).
        JSON format example:
            {
                "path": "/absolute/base/dir",
                "name": "simulation_box",
                "box": true,
                "hdf": true,
                "VIDE": true,
                "fraction_in_voids": null,
                "N_neighbours": 20,
                "box_size_mpc": 36.0,
                "void_size": 4.0
            }
    """
    candidates = [
        Path("config/global.json"),
        Path(__file__).resolve().parents[1] / "config/config_global.json",  # repo root/config/global.json
    ]
    for p in candidates:
        if p.exists():
            print(f"Loading global settings from {p}")
            with p.open("r", encoding="utf-8") as f:
                try:
                    obj = json.load(f) or {}
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON in {p}")
            return {
                "path": obj.get("path"),
                "name": obj.get("name"),
                "box": obj.get("box"),
                "hdf": obj.get("hdf"),
                "VIDE": obj.get("VIDE"),
                "param": obj.get("param"),
                "fraction_in_voids": obj.get("fraction_in_voids"),
                "N_neighbours": obj.get("N_neighbours"),
                "box_size_mpc": obj.get("box_size_mpc"),
                "void_size": obj.get("void_size"),
            }
    return {}


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 128
    learning_rate: float = 1e-3
    num_epochs: int = 100
    device: str = 'cuda'
    seed: int = 42
    
    
@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    normalize: bool = True
    seed: int = 42


@dataclass
class NotebookConfig:
    """Configuration for notebook parameters to avoid repetition.
    
    This centralizes common parameters used across all notebooks:
    - box: Whether using simulation box data
    - hdf: Whether using HDF5 format
    - VIDE: Whether using VIDE void catalogue
    - name: Dataset/experiment name
    - fraction_in_voids: Optional fraction threshold for void membership
    - model_name: Generated from name and fraction_in_voids
    - device: Computing device ('cpu' or 'cuda')
    - seed: Random seed for reproducibility
    - base_dir: Optional override for the base data directory (from global.json)
    - paths: Dictionary of directory paths (data_dir, checkpoint_dir_spec, etc.)
    """
    box: bool = True
    hdf: bool = True
    VIDE: bool = True
    name: str = 'simulation_box'
    param: Optional[Path] = None
    fraction_in_voids: Optional[str] = ''
    N_neighbours: int = 20
    box_size_mpc: float = 36.0
    void_size: float = 4.0
    device: str = 'mps'
    seed: int = 42
    base_dir: Optional[str] = None  # can be set from global.json
    model_name: str = field(init=False)
    paths: Dict[str, Path] = field(init=False, default_factory=dict)
    
    def __post_init__(self):
        """Generate derived attributes after initialization."""
        self.model_name = f'{self.name}_{self.fraction_in_voids}'
        # Setup paths automatically (uses base_dir if provided)
        self.paths = setup_paths(self.name, self.base_dir, self.param)
        # Setup device and seed
        setup_device_and_seed(device=self.device, seed=self.seed)
    
    def get_path(self, key: str) -> Path:
        """Convenience method to get a specific path."""
        return self.paths.get(key)
    
    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return self.paths['data_dir']
    
    @property
    def param_dir(self) -> Path:
        """Get specific param directory path."""
        return self.paths['param_dir']
    
    @property
    def galaxy_info_dir(self) -> Path:
        """Get galaxy info files directory path."""
        return self.paths['galaxy_info_dir']
    
    @property
    def checkpoint_dir_spec(self) -> Path:
        """Get specific checkpoint directory path."""
        return self.paths['checkpoint_dir_spec']
    
    @property
    def checkpoint_dir_global(self) -> Path:
        """Get global checkpoint directory path."""
        return self.paths['checkpoint_dir_global']
    
    @property
    def plot_dir(self) -> Path:
        """Get plot directory path."""
        return self.paths['plot_dir']
    
    @property
    def result_dir(self) -> Path:
        """Get result directory path."""
        return self.paths['result_dir']
    
    def print_info(self):
        """Print configuration information."""
        print("=" * 60)
        print("Notebook Configuration")
        print("=" * 60)
        print(f"Dataset: {self.name}")
        print(f"Model name: {self.model_name}")
        print(f"Fraction in voids: {self.fraction_in_voids}")
        print(f"Box: {self.box}, HDF: {self.hdf}, VIDE: {self.VIDE}")
        print(f"Device: {self.device}, Seed: {self.seed}")
        print(f'N neighbours: {self.N_neighbours}')
        print(f'Box size (Mpc/h): {self.box_size_mpc}')
        print(f"Void size: {self.void_size}")
        print("-" * 60)
        print("Paths:")
        for key, path in self.paths.items():
            print(f"  {key}: {path}")
        print("=" * 60)


def get_config(**kwargs) -> NotebookConfig:
    """Create a NotebookConfig with optional overrides.
    
    Behavior:
    - Reads config/global.json (or global.json) for:
        path, name, box, hdf, VIDE, fraction_in_voids
    - If you don't pass a parameter, the value from the global file is used.
    - You can still override anything via kwargs when needed.
    
    Example:
        >>> config = get_config()  # uses global.json values
        >>> print(config.data_dir)
    """
    globals_ = _load_global_settings()
    # Base dir (path)
    if 'base_dir' not in kwargs and globals_.get('path') is not None:
        kwargs['base_dir'] = globals_['path']
    # Simple fields
    for key in ('name', 'box', 'hdf', 'VIDE', 'fraction_in_voids', 'N_neighbours', 'box_size_mpc', "void_size", 'param'):
        if key not in kwargs and globals_.get(key) is not None:
            kwargs[key] = globals_[key]
    return NotebookConfig(**kwargs)