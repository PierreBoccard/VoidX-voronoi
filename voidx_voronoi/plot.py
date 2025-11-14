import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    classification_report
)

def true_false_positive_negative(test_preds_thr , test_targets, test_probs, save_dir=None):
    """Compute TP, TN, FP, FN counts from true and predicted labels and plot their distributions"""
    # cm = confusion_matrix(y_true, y_pred)
    # if cm.shape == (2, 2):
    #     tn, fp, fn, tp = cm.ravel()
    # else:
    #     tn = fp = fn = tp = 0
    # return tp, tn, fp, fn

    # False positives on test set (predicted 1 with val-derived threshold, but true label 0)
    fp_mask = (test_preds_thr == 1) & (test_targets == 0)
    fp_probs = test_probs[fp_mask]

    # True positives on test set (predicted 1 with val-derived threshold, and true label 1)
    tp_mask = (test_preds_thr == 1) & (test_targets == 1)
    tp_probs = test_probs[tp_mask]

    fn_mask = (test_preds_thr == 0) & (test_targets == 1)
    fn_probs = test_probs[fn_mask]

    # True negatives on test set (predicted 0 with val-derived threshold, and true label 0)
    tn_mask = (test_preds_thr == 0) & (test_targets == 0)
    tn_probs = test_probs[tn_mask]

    plt.figure(figsize=(6,4))
    sns.histplot(fp_probs, bins=np.linspace(0, 1, 41), kde=True, color='tab:orange', label ='False Positives')
    sns.histplot(tp_probs, bins=np.linspace(0, 1, 41), kde=True, color='tab:green', label ='True Positives', alpha=0.5)
    plt.axvline(0.5, color='k', ls='--', lw=1, label='0.5')
    plt.axvline(fp_probs.mean(), color='tab:orange', ls='-', lw=1, label=f'mean={fp_probs.mean():.3f}')
    plt.axvline(tp_probs.mean(), color='tab:green', ls='-', lw=1, label=f'mean={tp_probs.mean():.3f}')
    plt.title('Probability Distribution (Test)')
    plt.xlabel('Predicted probability')
    plt.ylabel('Count')
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        save_path = save_dir / 'true_false_positive_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("Saved true/false positive distribution plot to", save_path)
    plt.show()

    print(f"FP count={fp_probs.size} | mean={fp_probs.mean():.3f} | median={np.median(fp_probs):.3f} "
        f"| >0.9: {np.mean(fp_probs > 0.9):.2%} | 0.5–0.7: {np.mean((fp_probs >= 0.5) & (fp_probs < 0.7)):.2%}")
    print(f"TP count={tp_probs.size} | mean={tp_probs.mean():.3f} | median={np.median(tp_probs):.3f} "
                f"| >0.9: {np.mean(tp_probs > 0.9):.2%} | 0.5–0.7: {np.mean((tp_probs >= 0.5) & (tp_probs < 0.7)):.2%}")

    plt.figure(figsize=(6,4))
    sns.histplot(fn_probs, bins=np.linspace(0, 1, 41), kde=True, color='tab:red', label ='False Negatives')
    sns.histplot(tn_probs, bins=np.linspace(0, 1, 41), kde=True, color='tab:blue', label ='True Negatives', alpha=0.5)
    plt.axvline(0.5, color='k', ls='--', lw=1, label='0.5')
    plt.axvline(fn_probs.mean(), color='tab:red', ls='-', lw=1, label=f'mean={fn_probs.mean():.3f}')
    plt.axvline(tn_probs.mean(), color='tab:blue', ls='-', lw=1, label=f'mean={tn_probs.mean():.3f}')
    plt.title('Probability Distribution (Test)')
    plt.xlabel('Predicted probability')
    plt.ylabel('Count')
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        save_path = save_dir / 'true_false_negative_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("Saved true/false negative distribution plot to", save_path)
    plt.show()  

    print(f"FN count={fn_probs.size} | mean={fn_probs.mean():.3f} | median={np.median(fn_probs):.3f} "
            f"| <0.1: {np.mean(fn_probs < 0.1):.2%} | 0.3–0.5: {np.mean((fn_probs >= 0.3) & (fn_probs < 0.5)):.2%}")    
    print(f"TN count={tn_probs.size} | mean={tn_probs.mean():.3f} | median={np.median(tn_probs):.3f} "
                f"| <0.1: {np.mean(tn_probs < 0.1):.2%} | 0.3–0.5: {np.mean((tn_probs >= 0.3) & (tn_probs < 0.5)):.2%}")



def plot_precision_recall_curve(y_true, y_pred_proba, save_dir=None):
    """Plot Precision-Recall curve and baseline.

    Args:
        y_true: array-like of shape (N,), ground-truth binary labels {0,1}
        y_pred_proba: array-like of shape (N,), predicted probabilities in [0,1]
        save_path: optional path to save the figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {avg_precision:.4f})')

    baseline = np.sum(y_true) / max(len(y_true), 1)
    ax.plot([0, 1], [baseline, baseline], 'k--', linewidth=1, label=f'Baseline (AP = {baseline:.4f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        save_path = save_dir / 'precision_recall_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("Saved precision-recall curve to", save_path)
    plt.show()


def plot_prediction_distribution(y_true, y_pred_proba, save_dir=None):
    """Plot distribution of predicted probabilities for each class and boxplot summary.

    Args:
        y_true: array-like of shape (N,), ground-truth binary labels {0,1}
        y_pred_proba: array-like of shape (N,), predicted probabilities in [0,1]
        save_path: optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.hist(y_pred_proba[y_true == 0], bins=50, alpha=0.6,
             label='True Class 0', color='blue', edgecolor='black')
    ax1.hist(y_pred_proba[y_true == 1], bins=50, alpha=0.6,
             label='True Class 1', color='red', edgecolor='black')
    ax1.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Threshold')
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    data_to_plot = [y_pred_proba[y_true == 0], y_pred_proba[y_true == 1]]
    bp = ax2.boxplot(data_to_plot, labels=['Not in Void (0)', 'In Void (1)'],
                     patch_artist=True, showmeans=True)

    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax2.set_ylabel('Predicted Probability', fontsize=12)
    ax2.set_title('Predicted Probability by True Class', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_dir:
        save_path = save_dir / 'prediction_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("Saved prediction distribution plot to", save_path)
    plt.show()


def plot_threshold_analysis(y_true, y_pred_proba, save_dir=None):
    """Analyze performance over thresholds and plot Accuracy/Precision/Recall/F1/Specificity.

    Args:
        y_true: array-like of shape (N,), ground-truth binary labels {0,1}
        y_pred_proba: array-like of shape (N,), predicted probabilities in [0,1]
        save_path: optional path to save the figure

    Returns:
        optimal_threshold (float): threshold that maximizes F1 on the given data
    """
    thresholds = np.arange(0.0, 1.01, 0.01)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'specificity': []}

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            denom = (tp + tn + fp + fn)
            acc = (tp + tn) / denom if denom > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            metrics['accuracy'].append(acc)
            metrics['precision'].append(prec)
            metrics['recall'].append(rec)
            metrics['f1'].append(f1)
            metrics['specificity'].append(spec)
        else:
            # If labels are degenerate, append zeros
            for k in metrics:
                metrics[k].append(0.0)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(thresholds, metrics['accuracy'], label='Accuracy', linewidth=2)
    ax.plot(thresholds, metrics['precision'], label='Precision', linewidth=2)
    ax.plot(thresholds, metrics['recall'], label='Recall', linewidth=2)
    ax.plot(thresholds, metrics['f1'], label='F1 Score', linewidth=2)
    ax.plot(thresholds, metrics['specificity'], label='Specificity', linewidth=2)

    optimal_idx = int(np.argmax(metrics['f1'])) if len(metrics['f1']) else 0
    optimal_threshold = float(thresholds[optimal_idx]) if len(thresholds) else 0.5
    ax.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2,
               label=f'Optimal = {optimal_threshold:.3f}')
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, label='Default = 0.5')

    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        save_path = save_dir / 'threshold_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("Saved threshold analysis plot to", save_path)
    plt.show()

    print(f"\n=== Optimal Threshold: {optimal_threshold:.3f} ===")
    print(f"F1 Score: {metrics['f1'][optimal_idx]:.4f}")
    return optimal_threshold


# =========================
# Neuron activation helpers
# =========================
def register_activation_hooks(model, max_hooks=None):
    """Register forward hooks on Conv2d modules and collect activations.

    Args:
        model: torch.nn.Module
        max_hooks: optional int to limit number of hooked conv layers

    Returns:
        hooks (list): list of registered hook handles (call .remove() to detach)
        activations (dict): mapping name -> numpy array with shape (B,C,H,W) or variants
    """
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        raise RuntimeError("PyTorch must be installed to use activation hooks") from e

    activations = {}
    hooks = []

    def _get_hook(name):
        def hook(module, input, output):
            # Save CPU numpy for plotting
            try:
                activations[name] = output.detach().cpu().numpy()
            except Exception:
                # Fallback: convert to tensor then numpy
                activations[name] = output.cpu().numpy()
        return hook

    for name, module in model.named_modules():
        if hasattr(module, '__class__') and module.__class__.__name__ == 'Conv2d':
            hooks.append(module.register_forward_hook(_get_hook(name)))
            if max_hooks is not None and len(hooks) >= max_hooks:
                break

    if len(hooks) == 0:
        print("Warning: no Conv2d modules found to register hooks on.")

    return hooks, activations

def visualize_feature_maps(acts_dict, sample_idx=0, max_channels=8, cmap='viridis', save_dir=None):
    """Visualize feature maps for captured activations.

    Args:
        acts_dict: dict from register_activation_hooks
        sample_idx: which batch sample to visualize
        max_channels: number of channels per layer to display
        cmap: matplotlib colormap
    """
    for name, act in acts_dict.items():
        fmap = act[sample_idx]
        if fmap.ndim == 3:
            C, H, W = fmap.shape
        elif fmap.ndim == 2:
            C = fmap.shape[0]
            H, W = 1, fmap.shape[1]
            fmap = fmap.reshape(C, H, W)
        else:
            C = fmap.shape[0]
            fmap = fmap.reshape(C, -1, 1)
            H, W = fmap.shape[1], 1

        n = min(C, max_channels)
        cols = n
        rows = 1
        plt.figure(figsize=(3 * cols, 3 * rows))
        for i in range(n):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(fmap[i], cmap=cmap)
            plt.axis('off')
            plt.title(f"{name}: c{i}")
        plt.tight_layout()
        if save_dir:
            save_path = save_dir / f'feature_maps.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print("Saved feature maps to", save_path)
        plt.show()


def plot_first_global_embedding(acts_dict, save_dir=None):
    """Find first activation with shape (B,C,1,1) and bar-plot its channel vector.

    Returns True if a vector was plotted, else False.
    """
    for name, act in acts_dict.items():
        arr = act[0]
        if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == 1:
            vec = arr.reshape(-1)
            plt.figure(figsize=(6, 3))
            plt.bar(np.arange(len(vec)), vec)
            plt.title(f'Embedding (post-{name} pooled)')
            plt.xlabel('Channel')
            plt.ylabel('Activation')
            plt.tight_layout()
            if save_dir: 
                save_path = save_dir / f'global_embedding.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print("Saved global embedding plot to", save_path)
            plt.show()
            return True
    return False


# ===================
# Training curve plots
# ===================
def plot_training_and_validation_loss(history, include_val_bce=False, save_dir=None):
    """Plot train/val loss curves from a history dict.

    Accepts optional 'val_bce' in history for an unweighted BCE line.
    """
    plt.figure(figsize=(6, 4))
    if 'train_loss' in history:
        plt.plot(history['train_loss'], label='train loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='val loss (weighted)')
    if include_val_bce and 'val_bce' in history:
        plt.plot(history['val_bce'], label='val loss (unweighted BCE)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        save_path = save_dir / 'training_validation_loss.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("Saved training/validation loss plot to", save_path)
    plt.show()


# ==================
# Confusion matrices
# ==================
def plot_confusion_matrix_basic(cm, class_labels, title='Confusion Matrix', cmap=None, annotate_zeros_red=True, save_dir=None):
    """Basic confusion matrix with annotations using matplotlib."""
    if cmap is None:
        cmap = plt.cm.Blues
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'red' if annotate_zeros_red and cm[i, j] == 0 else 'black'
            plt.text(j, i, cm[i, j], ha='center', va='center', color=color, fontsize=12)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_dir:
        save_path = save_dir / 'confusion_matrix_basic.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("Saved confusion matrix to", save_path)
    plt.show()


def plot_confusion_matrices_all_views(cm, class_labels, cmap=plt.cm.Blues, save_dir=None):
    """Plot row-normalized, column-normalized, and globally-normalized confusion matrices."""
    # Row-normalized (recall view)
    row_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    row_norm[np.isnan(row_norm)] = 0
    plt.figure(figsize=(10, 8))
    sns.heatmap(row_norm, annot=True, fmt='.2f', cmap=cmap, cbar=True,
                xticklabels=class_labels, yticklabels=class_labels,
                linewidths=0.5, linecolor='black', square=True,
                annot_kws={'fontsize': 10})
    plt.title('Row-Normalized Confusion Matrix (Recall View)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    if save_dir:
        save_path = save_dir / 'confusion_matrix_row_normalized.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("Saved confusion matrices to", save_path)
    plt.show()

    # Column-normalized (precision-like view)
    col_norm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
    col_norm[np.isnan(col_norm)] = 0
    plt.figure(figsize=(10, 8))
    sns.heatmap(col_norm, annot=True, fmt='.2f', cmap=cmap, cbar=True,
                xticklabels=class_labels, yticklabels=class_labels,
                linewidths=0.5, linecolor='black', square=True,
                annot_kws={'fontsize': 10})
    plt.title('Column-Normalized Confusion Matrix (Precision-Like View)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    if save_dir:
        save_path = save_dir / 'confusion_matrix_column_normalized.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("Saved confusion matrices to", save_path)
    plt.show()

    # Globally-normalized (overall distribution)
    global_norm = cm.astype('float') / cm.sum()
    global_norm[np.isnan(global_norm)] = 0
    plt.figure(figsize=(10, 8))
    sns.heatmap(global_norm, annot=True, fmt='.2f', cmap=cmap, cbar=True,
                xticklabels=class_labels, yticklabels=class_labels,
                linewidths=0.5, linecolor='black', square=True,
                annot_kws={'fontsize': 10})
    plt.title('Globally-Normalized Confusion Matrix (Overall Distribution)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    if save_dir:
        save_path = save_dir / 'confusion_matrix_global_normalized.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("Saved confusion matrices to", save_path)
    plt.show()


# ========
# ROC plot
# ========
def plot_roc_curve(fpr, tpr, auc_value, title='ROC Curve', save_dir=None):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc_value:.4f}")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        save_path = save_dir / 'roc_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("Saved ROC curve to", save_path)
    plt.show()


# =====================
# Reliability diagram/ECE
# =====================
def expected_calibration_error(y_true, y_prob, n_bins=15):
    from sklearn.calibration import calibration_curve as _cal_curve
    prob_true, prob_pred = _cal_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = inds == b
        if np.any(mask):
            acc_b = y_true[mask].mean()
            conf_b = y_prob[mask].mean()
            w_b = mask.mean()
            ece += w_b * abs(acc_b - conf_b)
    return ece, prob_true, prob_pred


def plot_reliability_diagram(prob_pred, prob_true, ece, title='Reliability Diagram', save_dir=None):
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    plt.plot(prob_pred, prob_true, marker='o', label=f'Reliability (ECE={ece:.3f})')
    plt.xlabel('Predicted probability')
    plt.ylabel('Empirical frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        save_path = save_dir / 'reliability_diagram.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("Saved reliability diagram to", save_path)
    plt.show()
    

