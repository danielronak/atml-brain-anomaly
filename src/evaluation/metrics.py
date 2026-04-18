"""
Evaluation Metrics
-------------------
Dice, AUROC (voxel-wise), and Hausdorff95 for anomaly detection.

Usage:
    metrics = compute_all_metrics(anomaly_map, true_mask, config)

All metrics expect:
    anomaly_map: Tensor[1, 1, D, H, W] — continuous residual scores in [0, 1]
    true_mask:   Tensor[1, 1, D, H, W] — binary (0=healthy, 1=tumor)
"""

import numpy as np
import torch
from scipy.ndimage import label as scipy_label
from sklearn.metrics import roc_auc_score


def dice_score(pred_binary: torch.Tensor, true_mask: torch.Tensor) -> float:
    """Dice = 2|P∩T| / (|P| + |T|)"""
    pred = pred_binary.float()
    truth = true_mask.float()
    intersection = (pred * truth).sum()
    total = pred.sum() + truth.sum()
    if total == 0:
        return 1.0  # Both empty — perfect score
    return (2.0 * intersection / total).item()


def compute_auroc(anomaly_map: torch.Tensor, true_mask: torch.Tensor) -> float:
    """
    Voxel-level AUROC.
    Treats each voxel as a binary classification problem:
        positive = tumor voxel, score = anomaly map value
    """
    scores = anomaly_map.cpu().numpy().flatten()
    labels = (true_mask > 0).cpu().numpy().flatten().astype(int)

    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5  # Degenerate case

    return roc_auc_score(labels, scores)


def hausdorff95(pred_binary: torch.Tensor, true_mask: torch.Tensor) -> float:
    """
    95th percentile Hausdorff Distance (mm) between predicted and true masks.
    Lower is better. Returns inf if either mask is empty.
    """
    pred_np = pred_binary.squeeze().cpu().numpy().astype(bool)
    true_np = (true_mask > 0).squeeze().cpu().numpy().astype(bool)

    if pred_np.sum() == 0 or true_np.sum() == 0:
        return float("inf")

    # Surface point distances — using coordinate differences
    pred_pts = np.argwhere(pred_np).astype(float)
    true_pts = np.argwhere(true_np).astype(float)

    from scipy.spatial import cKDTree
    tree_pred = cKDTree(pred_pts)
    tree_true = cKDTree(true_pts)

    d_pred_to_true, _ = tree_true.query(pred_pts)
    d_true_to_pred, _ = tree_pred.query(true_pts)

    hd95 = max(np.percentile(d_pred_to_true, 95),
                np.percentile(d_true_to_pred, 95))
    return float(hd95)


def threshold_sweep(anomaly_map: torch.Tensor,
                    true_mask: torch.Tensor,
                    t_min: float = 0.01,
                    t_max: float = 0.50,
                    n_steps: int = 50) -> dict:
    """
    Sweep thresholds, compute Dice at each, return best.
    """
    thresholds = np.linspace(t_min, t_max, n_steps)
    dices = []

    for t in thresholds:
        pred_binary = (anomaly_map >= t).float()
        dices.append(dice_score(pred_binary, true_mask))

    best_idx = int(np.argmax(dices))
    return {
        "thresholds": thresholds.tolist(),
        "dices": dices,
        "best_dice": dices[best_idx],
        "best_threshold": float(thresholds[best_idx]),
    }


def compute_all_metrics(anomaly_map: torch.Tensor,
                         true_mask: torch.Tensor,
                         config: dict) -> dict:
    """
    Full evaluation pipeline for one patient.

    Returns:
        dict with best_dice, best_threshold, auroc, hausdorff95,
              + threshold sweep arrays for plotting
    """
    t_min = config["evaluation"]["threshold_min"]
    t_max = config["evaluation"]["threshold_max"]
    n_steps = config["evaluation"]["threshold_steps"]

    sweep = threshold_sweep(anomaly_map, true_mask, t_min, t_max, n_steps)

    best_t = sweep["best_threshold"]
    pred_binary = (anomaly_map >= best_t).float()

    auroc = compute_auroc(anomaly_map, true_mask)
    hd95 = hausdorff95(pred_binary, true_mask)

    return {
        "best_dice": sweep["best_dice"],
        "best_threshold": best_t,
        "auroc": auroc,
        "hausdorff95": hd95,
        "sweep_thresholds": sweep["thresholds"],
        "sweep_dices": sweep["dices"],
    }
