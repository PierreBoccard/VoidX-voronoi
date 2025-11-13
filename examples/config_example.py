"""
Example demonstrating the use of VoidX configuration for notebooks.

This example shows how to use the NotebookConfig class to centralize
common parameters and path setup, avoiding repetition across notebooks.
"""

from voidx import get_config

# Example 1: Create config with default values
print("Example 1: Default configuration")
print("=" * 60)
config = get_config(local=True)  # Use local=True for testing
config.print_info()

# Example 2: Create config with custom parameters
print("\n\nExample 2: Custom configuration")
print("=" * 60)
config_custom = get_config(
    box=True,
    local=True,
    hdf=True,
    VIDE=False,
    name='my_experiment',
    fraction_in_voids='0.3',
    device='cpu',
    seed=123
)
config_custom.print_info()

# Example 3: Access configuration values
print("\n\nExample 3: Accessing configuration values")
print("=" * 60)
print(f"Dataset name: {config_custom.name}")
print(f"Model name: {config_custom.model_name}")
print(f"Data directory: {config_custom.data_dir}")
print(f"Plot directory: {config_custom.plot_dir}")
print(f"Device: {config_custom.device}")
print(f"Seed: {config_custom.seed}")

# Example 4: Use in notebooks
print("\n\nExample 4: How to use in notebooks")
print("=" * 60)
print("""
In your Jupyter notebook, simply:

# Cell 1: Import
from voidx import get_config

# Cell 2: Configure
config = get_config(
    box=True,
    local=False,  # Set to True for local machine
    hdf=True,
    VIDE=True,
    name='simulation_box',
    fraction_in_voids='0.5'
)

# For backward compatibility with existing code
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

# Display configuration
config.print_info()

# Now your notebook code continues as before, using these variables!
""")

print("\n✓ Configuration examples completed!")
