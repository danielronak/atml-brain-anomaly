"""
Visualisation — Publication-quality comparison figures
-------------------------------------------------------
Generates all figures for the final report:
  1. 6-panel reconstruction grid (one patient, all models)
  2. Threshold-vs-Dice curves (all models overlaid)
  3. Dice score box plots across 50 patients
  4. Training loss curves panel (all models)
  5. Summary comparison table printed to console

Usage (from notebooks/05_evaluation.ipynb):
    from src.evaluation.visualize import (
        plot_reconstruction_grid,
        plot_threshold_curves,
        plot_dice_boxplot,
        plot_training_curves
    )
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch


# ── Colour palette (consistent across all figures) ────────────
COLOURS = {
    "cnn_naive":  "#e05c5c",   # red
    "cnn_proper": "#e09c55",   # orange
    "swin_gan":   "#5c8de0",   # blue
    "vqvae":      "#5cc47a",   # green
}
LABELS = {
    "cnn_naive":  "CNN GAN (Naive)",
    "cnn_proper": "CNN GAN (Proper)",
    "swin_gan":   "Swin-UNET GAN",
    "vqvae":      "VQ-VAE ⭐",
}


def _axial_slice(tensor: torch.Tensor, s: int | None = None) -> np.ndarray:
    """Extract a 2D axial slice from a (C, D, H, W) volume tensor."""
    arr = tensor.float().cpu().numpy()
    if arr.ndim == 4:
        arr = arr.mean(0)           # average channels for display
    s = s if s is not None else arr.shape[0] // 2
    return arr[s]


def plot_reconstruction_grid(patient_data: dict,
                              model_order: list[str],
                              save_path: str | Path,
                              patient_id: str = "example patient",
                              slice_idx: int | None = None) -> None:
    """
    6-panel (or N-panel) grid showing reconstruction quality per model.

    patient_data: dict keyed by model_name →
        {"volume": Tensor, "recon": Tensor, "residual": Tensor, "mask": Tensor}
    model_order: list of model keys in display order
    """
    n_models = len(model_order)
    # Rows: Original | Recon | Residual | [True Mask on last row]
    n_rows = 3
    n_cols = n_models + 1      # +1 for the ground-truth column

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.08, wspace=0.04)

    col_labels = ["Ground Truth"] + [LABELS.get(m, m) for m in model_order]
    row_labels = ["Input (T1)", "Reconstruction", "Anomaly Map"]

    first_model = model_order[0]
    vol = patient_data[first_model]["volume"]
    mask = patient_data[first_model]["mask"]
    s = slice_idx if slice_idx is not None else vol.shape[1] // 2

    for row in range(n_rows):
        for col in range(n_cols):
            ax = fig.add_subplot(gs[row, col])
            ax.axis("off")

            if col == 0:
                # Ground truth column
                if row == 0:
                    ax.imshow(_axial_slice(vol, s), cmap="gray", interpolation="nearest")
                elif row == 1:
                    ax.imshow(_axial_slice(vol, s), cmap="gray", interpolation="nearest")
                    ax.set_title("(no recon)", fontsize=8, color="grey")
                else:
                    ax.imshow(_axial_slice(mask, s), cmap="Reds", alpha=0.9)
                if row == 0:
                    ax.set_title(col_labels[col], fontsize=11, fontweight="bold", pad=6)
            else:
                model = model_order[col - 1]
                data = patient_data[model]
                colour = COLOURS.get(model, "black")

                if row == 0:
                    ax.imshow(_axial_slice(data["volume"], s), cmap="gray")
                    ax.set_title(col_labels[col], fontsize=11, fontweight="bold",
                                  pad=6, color=colour)
                elif row == 1:
                    ax.imshow(_axial_slice(data["recon"], s), cmap="gray")
                else:
                    ax.imshow(_axial_slice(data["residual"], s), cmap="inferno")

            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=10, rotation=90,
                               labelpad=4, fontweight="bold")

    fig.suptitle(
        f"Reconstruction Comparison — {patient_id}",
        fontsize=14, fontweight="bold", y=1.01
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def plot_threshold_curves(sweep_data: dict[str, dict],
                           save_path: str | Path) -> None:
    """
    Threshold-vs-Dice curves for all models on the same axes.

    sweep_data: dict keyed by model_name →
        {"thresholds": list, "mean_dices": list, "std_dices": list}
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for model, data in sweep_data.items():
        t   = np.array(data["thresholds"])
        mu  = np.array(data["mean_dices"])
        std = np.array(data["std_dices"])
        c   = COLOURS.get(model, "grey")
        L   = LABELS.get(model, model)

        best_idx = mu.argmax()
        ax.plot(t, mu, color=c, linewidth=2.5, label=f"{L} (best={mu[best_idx]:.3f})")
        ax.fill_between(t, mu - std, mu + std, color=c, alpha=0.15)
        ax.axvline(t[best_idx], color=c, linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Dice Score", fontsize=12)
    ax.set_title("Threshold Sweep — Mean Dice Across 50 BraTS Patients", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, None)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def plot_dice_boxplot(results_dfs: dict[str, pd.DataFrame],
                      save_path: str | Path) -> None:
    """
    Side-by-side box plots of Dice distributions across 50 patients.

    results_dfs: dict keyed by model_name → DataFrame with 'best_dice' column
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    models = list(results_dfs.keys())
    data   = [results_dfs[m]["best_dice"].values for m in models]
    labels = [LABELS.get(m, m) for m in models]
    colours = [COLOURS.get(m, "steelblue") for m in models]

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                     medianprops={"color": "white", "linewidth": 2},
                     whiskerprops={"linewidth": 1.5},
                     capprops={"linewidth": 1.5},
                     flierprops={"marker": "o", "markersize": 4, "alpha": 0.5})

    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)

    # Overlay individual data points
    for i, (d, c) in enumerate(zip(data, colours), 1):
        jitter = np.random.uniform(-0.15, 0.15, len(d))
        ax.scatter(np.full(len(d), i) + jitter, d, color=c, alpha=0.4, s=20, zorder=5)

    ax.set_ylabel("Dice Score (at optimal threshold)", fontsize=12)
    ax.set_title("Anomaly Detection Dice — 50 BraTS 2021 Patients", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, None)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def plot_training_curves(histories: dict[str, dict],
                          save_path: str | Path) -> None:
    """
    Multi-panel training loss comparison.

    histories: dict keyed by model_name → loss history dict
        VQ-VAE:  {"train_recon": [...], "val_recon": [...]}
        GAN:     {"d_loss": [...], "g_loss": [...], "gp": [...]}
    """
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (model, hist) in zip(axes, histories.items()):
        c = COLOURS.get(model, "steelblue")
        L = LABELS.get(model, model)

        if "train_recon" in hist:
            ax.plot(hist["train_recon"], color=c, linewidth=2, label="Train Recon")
            ax.plot(hist["val_recon"],   color=c, linewidth=2, linestyle="--", label="Val Recon", alpha=0.7)
            ax.set_ylabel("L1 Loss")
        elif "d_loss" in hist:
            ax.plot(hist["d_loss"], color=c, linewidth=2, label="Discriminator")
            ax.plot(hist["g_loss"], color="grey", linewidth=2, label="Generator")
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_ylabel("WGAN Loss")

        ax.set_title(L, fontsize=11, fontweight="bold", color=c)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def print_summary_table(results_dfs: dict[str, pd.DataFrame]) -> None:
    """Print the final comparison table to console."""
    print("\n" + "="*70)
    print(f"{'Model':<25} {'Mean Dice':>10} {'Std':>8} {'Mean AUROC':>12} {'Mean HD95':>11}")
    print("="*70)
    for model, df in results_dfs.items():
        label = LABELS.get(model, model)
        dice  = df["best_dice"].mean()
        std   = df["best_dice"].std()
        auroc = df["auroc"].mean()
        hd95  = df["hausdorff95"].replace(float("inf"), float("nan")).mean()
        print(f"  {label:<23} {dice:>10.4f} {std:>8.4f} {auroc:>12.4f} {hd95:>10.2f}")
    print("="*70 + "\n")
