"""
Anomaly Scorer — Unified Inference Pipeline
---------------------------------------------
Works with any model type (VQ-VAE or GAN) and produces:
  - Reconstructed volume
  - Smoothed residual anomaly map
  - All evaluation metrics per patient

Usage:
    scorer = AnomalyScorer(model, model_type='vqvae', config=config)
    results = scorer.score_patient(volume, mask, patient_id)
    scorer.run_all(test_loader, save_dir)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.evaluation.metrics import compute_all_metrics


class AnomalyScorer:
    """
    Inference wrapper that works for all 4 model types.

    model_type:
        'vqvae'   — model(x) returns (reconstruction, quant_loss)
        'gan'     — generator(encoder(x)) returns reconstruction
                    Provide encoder= kwarg for GAN models
    """

    def __init__(self,
                 model: nn.Module,
                 model_type: str,
                 config: dict,
                 device: torch.device,
                 encoder: nn.Module | None = None,
                 discriminator: nn.Module | None = None):

        assert model_type in ("vqvae", "gan"), f"Unknown model_type: {model_type}"

        self.model = model.eval().to(device)
        self.model_type = model_type
        self.config = config
        self.device = device

        if model_type == "gan":
            assert encoder is not None, "GAN inference requires an encoder."
            self.encoder = encoder.eval().to(device)
            self.discriminator = discriminator.eval().to(device) if discriminator else None

        self.sigma = config["evaluation"]["gaussian_smooth_sigma"]

    def _reconstruct(self, volume: torch.Tensor) -> torch.Tensor:
        """Get pseudo-healthy reconstruction from the model."""
        x = volume.to(self.device)
        with torch.no_grad():
            if self.model_type == "vqvae":
                recon, _ = self.model(x)
            else:  # gan
                z = self.encoder(x)
                recon = self.model(z)
        return recon

    def _compute_residual(self,
                           volume: torch.Tensor,
                           recon: torch.Tensor) -> torch.Tensor:
        """
        Pixel-wise absolute difference, Gaussian-smoothed.
        For dual-channel, take channel-wise mean of residual.
        """
        residual = torch.abs(volume - recon)  # (1, C, D, H, W)

        if residual.shape[1] > 1:
            residual = residual.mean(dim=1, keepdim=True)  # (1, 1, D, H, W)

        if self.sigma > 0:
            residual = self._gaussian_smooth(residual, self.sigma)

        # Normalize to [0, 1] for comparable threshold sweeps across models
        r_min = residual.min()
        r_max = residual.max()
        if r_max > r_min:
            residual = (residual - r_min) / (r_max - r_min)

        return residual

    @staticmethod
    def _gaussian_smooth(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """Simple 3D Gaussian smoothing via separable 1D convolutions."""
        kernel_size = int(6 * sigma + 1) | 1  # odd
        half = kernel_size // 2
        coords = torch.arange(kernel_size, dtype=torch.float32) - half
        kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        device = x.device
        k = kernel_1d.to(device)

        def conv_along(t, dim):
            # Reshape to apply 1D conv along one spatial axis
            shape = t.shape  # (1, 1, D, H, W)
            t = t.squeeze(0).squeeze(0)  # (D, H, W)
            t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            weight = k.view(1, 1, *([1] * (dim - 2)), -1, *([1] * (4 - dim)))
            pad = [0] * (2 * (4 - dim)) + [half, half]
            return F.conv3d(F.pad(t, pad, mode="replicate"), weight)

        for dim in [2, 3, 4]:
            x = conv_along(x, dim)
        return x

    def score_patient(self,
                       volume: torch.Tensor,
                       mask: torch.Tensor,
                       patient_id: str = "unknown") -> dict:
        """
        Score a single patient.

        Args:
            volume: (1, C, D, H, W) — preprocessed MRI volume
            mask:   (1, 1, D, H, W) — binary tumor mask
            patient_id: string for logging

        Returns:
            dict with reconstruction, residual, and all metrics
        """
        t0 = time.time()
        recon = self._reconstruct(volume)
        residual = self._compute_residual(volume.to(self.device), recon)
        mask_device = mask.to(self.device)

        metrics = compute_all_metrics(residual, mask_device, self.config)
        metrics["patient_id"] = patient_id
        metrics["inference_time_s"] = round(time.time() - t0, 2)

        return {
            "patient_id": patient_id,
            "volume": volume,
            "recon": recon.cpu(),
            "residual": residual.cpu(),
            "mask": mask,
            "metrics": metrics,
        }

    def run_all(self,
                test_loader,
                model_name: str,
                save_dir: str | Path,
                verbose: bool = True) -> pd.DataFrame:
        """
        Run inference on all patients in test_loader.
        Saves CSV of metrics and individual patient tensors.

        Returns: DataFrame of all patient metrics.
        """
        save_dir = Path(save_dir) / model_name
        save_dir.mkdir(parents=True, exist_ok=True)

        records = []

        for i, batch in enumerate(test_loader):
            volume = batch["image"]         # (1, C, D, H, W)
            mask = batch["mask"]            # (1, 1, D, H, W)
            # Patient ID from filename if available; else index
            pid = batch.get("patient_id", [f"patient_{i:03d}"])[0]

            result = self.score_patient(volume, mask, pid)
            m = result["metrics"]
            records.append({
                "patient_id": pid,
                "best_dice": m["best_dice"],
                "best_threshold": m["best_threshold"],
                "auroc": m["auroc"],
                "hausdorff95": m["hausdorff95"],
                "inference_time_s": m["inference_time_s"],
            })

            if verbose:
                print(f"  [{i+1:3d}] {pid:30s} | "
                      f"Dice: {m['best_dice']:.4f} | "
                      f"AUROC: {m['auroc']:.4f} | "
                      f"HD95: {m['hausdorff95']:.1f}")

            # Save residual + recon tensors for visualisation
            pt_dir = save_dir / "patient_tensors"
            pt_dir.mkdir(exist_ok=True)
            torch.save({
                "volume": result["volume"].half(),   # half precision to save Drive space
                "recon": result["recon"].half(),
                "residual": result["residual"].half(),
                "mask": result["mask"].half(),
            }, pt_dir / f"{pid}.pt")

        df = pd.DataFrame(records)
        csv_path = save_dir / "metrics.csv"
        df.to_csv(csv_path, index=False)

        print(f"\n{'─'*60}")
        print(f"  {model_name} Results ({len(df)} patients):")
        print(f"  Mean Dice:     {df['best_dice'].mean():.4f} ± {df['best_dice'].std():.4f}")
        print(f"  Mean AUROC:    {df['auroc'].mean():.4f} ± {df['auroc'].std():.4f}")
        print(f"  Mean HD95:     {df['hausdorff95'].replace(float('inf'), np.nan).mean():.2f}")
        print(f"  Results saved: {csv_path}")
        print(f"{'─'*60}")

        return df
