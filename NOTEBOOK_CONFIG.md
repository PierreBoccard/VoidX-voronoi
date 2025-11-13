# Notebook Configuration Guide

This guide explains how VoidX notebooks use the centralized configuration system to avoid code repetition.

## Overview

All VoidX notebooks now use the `NotebookConfig` class from `voidx.config` to manage:
- Common parameters (box, local, hdf, VIDE, name, fraction_in_voids)
- Automatic path setup (data_dir, checkpoint_dir, plot_dir, result_dir)
- Device and seed initialization
- Consistent structure across all notebooks

## Benefits

✅ **No repetition** - Parameters and paths defined once  
✅ **Consistency** - All notebooks follow the same pattern  
✅ **Easy to modify** - Change parameters in one place  
✅ **Automatic setup** - Directories created automatically  
✅ **Type-safe** - Configuration validated at creation  

## Quick Start

### In Your Notebook

**Cell 1: Imports**
```python
# Standard imports
import numpy as np
import matplotlib.pyplot as plt
# ... other imports ...

# Import VoidX configuration
from voidx import get_config
```

**Cell 2: Configuration**
```python
# Initialize notebook configuration
config = get_config(
    box=True,           # Using simulation box data
    local=False,        # Running on cluster (True for local)
    hdf=True,           # Using HDF5 format
    VIDE=True,          # Using VIDE void catalogue
    name='simulation_box',          # Dataset name
    fraction_in_voids='0.5'         # Void membership threshold
)

# For backward compatibility, extract variables
box = config.box
local = config.local
hdf = config.hdf
VIDE = config.VIDE
name = config.name
fraction_in_voids = config.fraction_in_voids
model_name = config.model_name
device = config.device
seed = config.seed

# All paths are automatically set up
data_dir = config.data_dir
checkpoint_dir_spec = config.checkpoint_dir_spec
checkpoint_dir_global = config.checkpoint_dir_global
plot_dir = config.plot_dir
result_dir = config.result_dir

# Display configuration (optional)
config.print_info()
```

**That's it!** Your notebook now has all parameters and paths configured.

## Configuration Parameters

### Required Parameters (with defaults)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `box` | bool | True | Whether using simulation box data |
| `local` | bool | False | True for local machine, False for cluster |
| `hdf` | bool | True | Whether using HDF5 file format |
| `VIDE` | bool | True | Whether using VIDE void catalogue |
| `name` | str | 'simulation_box' | Dataset/experiment name |
| `fraction_in_voids` | str | '0.5' | Void membership threshold |
| `device` | str | 'cpu' | Computing device ('cpu' or 'cuda') |
| `seed` | int | 42 | Random seed for reproducibility |

### Automatic Path Setup

The configuration automatically creates and provides:

```python
data_dir                  # Main data directory
checkpoint_dir_spec       # Checkpoint directory for this dataset
checkpoint_dir_global     # Global checkpoint directory
plot_dir                  # Directory for plots
result_dir                # Directory for results
```

**Local paths** (when `local=True`):
```
../data/{name}/
├── checkpoints/
├── plot/
└── result/
```

**Cluster paths** (when `local=False`):
```
/datadec/cppm/boccard/ML/data/{name}/
├── checkpoints/
├── plot/
└── result/
```

## Examples

### Example 1: Default Configuration
```python
from voidx import get_config

# Use all defaults (simulation_box, fraction 0.5, on cluster)
config = get_config(local=True)  # Only override local
```

### Example 2: Custom Experiment
```python
from voidx import get_config

# Custom experiment with different parameters
config = get_config(
    box=False,
    local=True,
    hdf=False,
    VIDE=False,
    name='my_custom_experiment',
    fraction_in_voids='0.3',
    device='cuda',
    seed=123
)
```

### Example 3: Different Dataset
```python
from voidx import get_config

# Use different dataset
config = get_config(
    name='z0.8-1_ra1_voronoi_dens02_frac05',
    fraction_in_voids='0.7',
    local=False
)
```

## Accessing Configuration

### Direct Access
```python
print(config.name)           # 'simulation_box'
print(config.model_name)     # 'simulation_box_0.5'
print(config.device)         # 'cpu'
```

### Path Properties
```python
print(config.data_dir)               # Path object
print(config.checkpoint_dir_global)  # Path object
print(config.plot_dir)               # Path object
```

### Path Dictionary
```python
# Access paths via dictionary
paths = config.paths
print(paths['data_dir'])
print(paths['plot_dir'])
```

### Print Configuration
```python
# Display all configuration info
config.print_info()
```

Output:
```
============================================================
Notebook Configuration
============================================================
Dataset: simulation_box
Model name: simulation_box_0.5
Fraction in voids: 0.5
Box: True, Local: False, HDF: True, VIDE: True
Device: cpu, Seed: 42
------------------------------------------------------------
Paths:
  data_dir: /datadec/cppm/boccard/ML/data/simulation_box
  checkpoint_dir_spec: /datadec/cppm/boccard/ML/data/simulation_box/checkpoints
  checkpoint_dir_global: /datadec/cppm/boccard/ML/data/checkpoints
  plot_dir: /datadec/cppm/boccard/ML/data/simulation_box/plot
  result_dir: /datadec/cppm/boccard/ML/data/simulation_box/result
============================================================
```

## Migration Guide

### Old Approach (Before)
```python
# Cell 1: Parameters
box = True
local = False
hdf = True
VIDE = True

# Cell 2: Name
name = 'simulation_box'
fraction_in_voids = '0.5'
model_name = f'{name}_{fraction_in_voids}'

# Cell 3: Local paths
if local:
    data_dir = Path(os.path.abspath(os.path.join(os.getcwd(), f'../data/{name}/')))
    # ... many more lines ...

# Cell 4: Cluster paths
if not local:
    data_dir = Path('/datadec/cppm/boccard/ML/data') / name
    # ... many more lines ...
```

### New Approach (After)
```python
# Cell 1: Configuration
from voidx import get_config

config = get_config(
    box=True, local=False, hdf=True, VIDE=True,
    name='simulation_box', fraction_in_voids='0.5'
)

# Extract for backward compatibility
box, local, hdf, VIDE = config.box, config.local, config.hdf, config.VIDE
name, fraction_in_voids, model_name = config.name, config.fraction_in_voids, config.model_name
data_dir, plot_dir, result_dir = config.data_dir, config.plot_dir, config.result_dir
checkpoint_dir_spec = config.checkpoint_dir_spec
checkpoint_dir_global = config.checkpoint_dir_global
```

**Result**: 4+ cells reduced to 1 cell! 🎉

## Updated Notebooks

All the following notebooks have been updated to use the new configuration system:

- ✅ `notebooks/data_preparation.ipynb`
- ✅ `notebooks/model_training.ipynb`
- ✅ `notebooks/model_training2.ipynb`
- ✅ `notebooks/model_training3.ipynb`
- ✅ `notebooks/void_finder.ipynb`
- ✅ `notebooks/reconstruction.ipynb`
- ✅ `notebooks/data_exploration.ipynb`

## See Also

- `examples/config_example.py` - Standalone example demonstrating configuration usage
- `voidx/config.py` - Source code for configuration classes
- `examples/README.md` - Examples documentation

## Troubleshooting

### Paths not found
Make sure you set `local=True` when running on your local machine:
```python
config = get_config(local=True, ...)
```

### Custom base directory
If you need a custom base directory:
```python
from voidx.config import setup_paths
paths = setup_paths(name='my_exp', local=True, base_dir='/custom/path')
```

### Device not available
The config automatically handles device setup. If CUDA is not available:
```python
config = get_config(device='cpu')  # Explicitly use CPU
```
