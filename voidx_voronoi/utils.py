"""
Visualization utilities for VoidX.

This module provides functions for visualizing galaxy distributions,
void predictions, and model performance.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, Tuple, Dict
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from astropy.cosmology import LambdaCDM, FlatLambdaCDM, FlatwCDM, wCDM, w0waCDM
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef,
    brier_score_loss, log_loss, roc_curve
)


def convert_to_Cartesian(ra, dec, z, cosmo=FlatLambdaCDM(H0=71, Om0=0.315)):
    """
    Convert RA, Dec, z to Cartesian coordinates.

    Args:
        ra: Right Ascension in degrees
        dec: Declination in degrees
        z: Redshift
        cosmo: Astropy cosmology instance

    Returns:
        x, y, z: Cartesian coordinates
    """
    # Convert RA and Dec from degrees to radians
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)

    # Compute comoving distance in Mpc
    r = cosmo.comoving_distance(z).value  # in Mpc

    # Convert to Cartesian coordinates
    x = r * np.cos(dec_rad) * np.cos(ra_rad)
    y = r * np.cos(dec_rad) * np.sin(ra_rad)
    z = r * np.sin(dec_rad)

    return x, y, z


def evaluate_model(model, loader, device='cpu'):
    """Evaluate a binary classification model on a data loader.
    
    Computes comprehensive metrics including accuracy, precision, recall, F1,
    ROC-AUC, balanced accuracy, MCC, Brier score, log loss, and TSS.
    
    Args:
        model: PyTorch model to evaluate
        loader: DataLoader with test data
        device: Device to run evaluation on ('cpu' or 'cuda')
    
    Returns:
        Dictionary of evaluation metrics and predictions
    """
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu())
            all_targets.append(yb.cpu())
    
    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs >= 0.5).astype(np.int32)

    # Base metrics
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, zero_division=0)
    rec = recall_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, zero_division=0)
    auc = roc_auc_score(targets, probs)

    # Robust to class prevalence / threshold-free
    bal_acc = balanced_accuracy_score(targets, preds)  # insensitive to imbalance
    mcc = matthews_corrcoef(targets, preds)            # correlation coefficient, robust

    # Brier score and skill score (lower is better for Brier)
    brier = brier_score_loss(targets, probs)
    # Reference model: constant prevalence (climatology)
    p_ref = targets.mean()
    brier_ref = brier_score_loss(targets, np.full_like(probs, p_ref))
    bss = 1.0 - (brier / brier_ref) if brier_ref > 0 else 0.0

    # Log-loss and skill score
    eps = 1e-12
    ll = log_loss(targets, np.clip(probs, eps, 1 - eps))
    ll_ref = log_loss(targets, np.full_like(probs, np.clip(p_ref, eps, 1 - eps)))
    llss = 1.0 - (ll / ll_ref) if ll_ref > 0 else 0.0

    # True Skill Statistic (TSS) = TPR - FPR; get max over thresholds from ROC curve
    fpr, tpr, thr = roc_curve(targets, probs)
    tss = (tpr - fpr)
    tss_max = float(np.max(tss)) if len(tss) else 0.0

    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': auc,
        'balanced_accuracy': bal_acc,
        'mcc': mcc,
        'brier': brier,
        'brier_ref': brier_ref,
        'brier_skill': bss,
        'log_loss': ll,
        'log_loss_ref': ll_ref,
        'log_loss_skill': llss,
        'tss_max': tss_max,
        'probs': probs,
        'preds': preds,
        'targets': targets,
    }
    return metrics
