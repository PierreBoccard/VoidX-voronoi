"""
PyTorch models for void detection in galaxy distributions.

This module contains various neural network architectures for classifying
galaxies as being in voids or not, based on their positions and neighbor distances.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
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
    """MLP model for void galaxy classification with configurable architecture.
        Args:
            in_features: Number of input features per galaxy
            hidden_layers: Tuple specifying the number of units in each hidden layer
            dropout: Dropout rate between layers
            use_batchnorm: Whether to use Batch Normalization
            activation: Activation function to use ('relu', 'gelu', 'elu')
        """

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
    
