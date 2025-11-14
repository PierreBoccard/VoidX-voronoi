"""
Custom loss functions for VoidX.

Includes:
- FocalLossWithLogits: BCE-with-logits with focal modulation and optional alpha/pos_weight.
- BCEWithLogitsWithLabelSmoothing: BCE-with-logits using label smoothing.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithLogits(nn.Module):
    """Focal loss for binary classification with logits.

    Args:
        gamma: focusing parameter (>=0); 0 reduces to plain BCE.
        alpha: class balance factor in [0,1]; if None, no alpha balancing is applied.
        pos_weight: same semantics as in BCEWithLogitsLoss (weights positive examples).
        reduction: 'mean' | 'sum' | 'none'.
    Notes:
        - You can use either alpha or pos_weight (or both); pos_weight is typically
          used to reflect the prevalence in training data, while alpha emphasizes
          positives (alpha close to 1) or negatives (alpha close to 0).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        *,
        alpha: Optional[float] = None,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        assert reduction in {"mean", "sum", "none"}
        self.gamma = float(gamma)
        self.alpha = alpha
        self.register_buffer("pos_weight", pos_weight if isinstance(pos_weight, torch.Tensor) else None)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Flatten to 1D for safety
        logits = logits.reshape(-1)
        targets = targets.reshape(-1)

        # Base BCE with logits, element-wise
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )

        if self.gamma == 0.0:
            loss = bce
        else:
            # p_t = p if y=1 else (1-p)
            p = torch.sigmoid(logits)
            p_t = p * targets + (1 - p) * (1 - targets)
            focal_factor = (1.0 - p_t).clamp(min=1e-12).pow(self.gamma)
            loss = focal_factor * bce

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BCEWithLogitsWithLabelSmoothing(nn.Module):
    """Binary cross-entropy with logits and label smoothing.

    Args:
        smoothing: in [0, 1). targets become y*(1-s) + 0.5*s.
        pos_weight: optional positive class weight as in PyTorch BCEWithLogitsLoss.
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        smoothing: float = 0.05,
        *,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        assert reduction in {"mean", "sum", "none"}
        self.smoothing = float(smoothing)
        self.register_buffer("pos_weight", pos_weight if isinstance(pos_weight, torch.Tensor) else None)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.reshape(-1)
        targets = targets.reshape(-1)
        # Smooth toward 0.5
        t_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        loss = F.binary_cross_entropy_with_logits(
            logits, t_smooth, reduction=self.reduction, pos_weight=self.pos_weight
        )
        return loss


__all__ = [
    "FocalLossWithLogits",
    "BCEWithLogitsWithLabelSmoothing",
]
