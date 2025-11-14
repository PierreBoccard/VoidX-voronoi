"""
PyTorch models for void detection in galaxy distributions.

This module contains various neural network architectures for classifying
galaxies as being in voids or not, based on their positions and neighbor distances.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np
import math


class MLP(nn.Module):
    """Multi-Layer Perceptron for binary classification of void membership.
    
    Args:
        in_dim: Input feature dimension
        hidden: Tuple of hidden layer dimensions (default: (256, 128, 64))
        dropout: Dropout rate (default: 0.3)
    """
    def __init__(self, in_dim, hidden=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim 
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]  # output logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class VoidMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_layers=(256, 128, 64),  # Slightly larger for 3 features
        dropout: float = 0.3,  # Increase dropout slightly
        use_batchnorm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        assert in_features > 0, "in_features must be positive"
        assert 0.0 <= dropout < 1.0, "dropout must be in [0, 1)"
        act_layer = self._get_activation(activation)
        layers = []
        prev = in_features
        self.blocks = nn.ModuleList()
        for h in hidden_layers:
            block = []
            block.append(nn.Linear(prev, h))
            if use_batchnorm:
                block.append(nn.BatchNorm1d(h))
            block.append(act_layer())
            if dropout > 0:
                block.append(nn.Dropout(dropout))
            self.blocks.append(nn.Sequential(*block))
            prev = h
        # Final classifier head to a single logit, then sigmoid in forward
        self.out = nn.Linear(prev, 1)
        # Save config
        self.in_features = in_features
        self.hidden_layers = list(hidden_layers)
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.activation_name = activation
        self.reset_parameters()
    def _get_activation(self, name: str):
        name = name.lower()
        if name == "relu":
            return nn.ReLU
        if name == "gelu":
            return nn.GELU
        if name == "elu":
            return nn.ELU
        raise ValueError(f"Unsupported activation: {name}")
    def reset_parameters(self):
        # Kaiming initialization for hidden layers; zeros for biases
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))  # suitable for ReLU-family
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape (N_galaxies, N_features)
        returns: tensor of shape (N_galaxies, 1) with values in [0, 1]
        """
        for block in self.blocks:
            x = block(x)
        logits = self.out(x)
        probs = torch.sigmoid(logits)
        return probs
    

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = True, dropout_p: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.dropout2d = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout2d(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class DeepResidualCNN(nn.Module):
    """
    A larger-capacity CNN with residual connections and optional SE attention.
    - Robust to varying HxW via AdaptiveAvgPool2d.
    - Depth/width configurable via widths and blocks_per_stage.
    """
    def __init__(
        self,
        in_channels: int = 1,
        widths: tuple = (32, 64, 128, 256),
        blocks_per_stage: tuple = (2, 2, 2, 2),
        block_dropout: float = 0.10,
        fc_hidden: int = 256,
        fc_dropout: float = 0.30,
        use_se: bool = True,
    ):
        super().__init__()
        w0 = widths[0]
        # Stem aggressively downsamples to expose larger receptive field early
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, w0, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(w0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages
        stages = []
        in_ch = w0
        for s, (w, n_blocks) in enumerate(zip(widths, blocks_per_stage)):
            for i in range(n_blocks):
                stride = 2 if (i == 0 and s > 0) else 1  # downsample once per stage (except first)
                stages.append(ResidualBlock(in_ch, w, stride=stride, use_se=use_se, dropout_p=block_dropout))
                in_ch = w
        self.features = nn.Sequential(*stages)

        # Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(fc_dropout),
            nn.Linear(in_ch, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x):  # x: (B, 1, H, W)
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.head(x).squeeze(1)
        return logits


class TinyCNN(nn.Module):
    """
    A compact CNN baseline for 2D neighbor-distance maps (e.g., 20x20).
    Useful when you want a simpler model to reduce overfitting or speed up experimentation.

    Structure:
      Conv(16)->BN->ReLU -> Conv(32)->BN->ReLU -> GAP -> Dropout -> Linear(1)
    """
    def __init__(
        self,
        in_channels: int = 1,
        c1: int = 16,
        c2: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c2, 1),
        )

    def forward(self, x):  # x: (B, 1, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x).squeeze(1)


# ==================== Graph Neural Network Models ====================

try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    # Define dummy classes to avoid import errors
    class GCNConv:
        pass
    class GATConv:
        pass
    class SAGEConv:
        pass
    def global_mean_pool(*args, **kwargs):
        raise NotImplementedError("torch_geometric not installed")


class VoronoiGCN(nn.Module):
    """
    Graph Convolutional Network for Voronoi cell classification.
    
    This model uses the Voronoi tessellation structure (adjacency graph)
    to classify cells as void or non-void. It addresses spatial leakage
    by focusing on topological features rather than absolute positions.
    
    Args:
        in_features: Number of input node features (e.g., volume, neighbor_count, positions)
        hidden_channels: Hidden dimension for GCN layers
        num_layers: Number of GCN layers
        dropout: Dropout rate
        use_batchnorm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        in_features: int = 5,  # normalized_volume + neighbor_count + position(3)
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for GNN models. "
                "Install with: pip install torch-geometric"
            )
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        
        # Input projection
        self.input_proj = nn.Linear(in_features, hidden_channels)
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batchnorm else None
        
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            if use_batchnorm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """
        Forward pass.
        
        Args:
            x: Node features, shape (num_nodes, in_features)
            edge_index: Edge indices, shape (2, num_edges)
            batch: Batch assignment vector for batched graphs (optional)
            
        Returns:
            Logits for each node, shape (num_nodes,)
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # GCN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if self.use_batchnorm:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Classify each node
        logits = self.classifier(x).squeeze(-1)
        
        return logits


class VoronoiGAT(nn.Module):
    """
    Graph Attention Network for Voronoi cell classification.
    
    Uses attention mechanisms to weight neighbor importance, which can be
    useful for identifying void boundaries where cell properties change rapidly.
    
    Args:
        in_features: Number of input node features
        hidden_channels: Hidden dimension
        num_layers: Number of GAT layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_batchnorm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        in_features: int = 5,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for GNN models. "
                "Install with: pip install torch-geometric"
            )
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        
        # Input projection
        self.input_proj = nn.Linear(in_features, hidden_channels)
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batchnorm else None
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer: single head output
                self.convs.append(
                    GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout)
                )
            else:
                # Multi-head attention with concatenation
                self.convs.append(
                    GATConv(hidden_channels, hidden_channels // num_heads, 
                           heads=num_heads, dropout=dropout, concat=True)
                )
            
            if use_batchnorm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """
        Forward pass.
        
        Args:
            x: Node features, shape (num_nodes, in_features)
            edge_index: Edge indices, shape (2, num_edges)
            batch: Batch assignment vector for batched graphs (optional)
            
        Returns:
            Logits for each node, shape (num_nodes,)
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # GAT layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if self.use_batchnorm:
                x = self.batch_norms[i](x)
            if i < self.num_layers - 1:  # No activation after last layer
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Classify each node
        logits = self.classifier(x).squeeze(-1)
        
        return logits


class VoronoiSAGE(nn.Module):
    """
    GraphSAGE for Voronoi cell classification.
    
    Uses sampling and aggregation, which can be more efficient for large graphs
    and provides good inductive learning capabilities.
    
    Args:
        in_features: Number of input node features
        hidden_channels: Hidden dimension
        num_layers: Number of SAGE layers
        dropout: Dropout rate
        use_batchnorm: Whether to use batch normalization
        aggregator: Aggregation method ('mean', 'max', 'lstm')
    """
    
    def __init__(
        self,
        in_features: int = 5,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
        aggregator: str = 'mean',
    ):
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for GNN models. "
                "Install with: pip install torch-geometric"
            )
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        
        # Input projection
        self.input_proj = nn.Linear(in_features, hidden_channels)
        
        # SAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batchnorm else None
        
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))
            if use_batchnorm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """
        Forward pass.
        
        Args:
            x: Node features, shape (num_nodes, in_features)
            edge_index: Edge indices, shape (2, num_edges)
            batch: Batch assignment vector for batched graphs (optional)
            
        Returns:
            Logits for each node, shape (num_nodes,)
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # SAGE layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if self.use_batchnorm:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Classify each node
        logits = self.classifier(x).squeeze(-1)
        
        return logits
