"""
BraTS 2021 Dataset Loader
--------------------------
What this does:
  Provides PyTorch Dataset and DataLoader classes for loading
  preprocessed BraTS 2021 slices during training and evaluation.

Two datasets:
  - HealthyDataset: loads only healthy slices (used for training)
  - AnomalyDataset: loads tumor slices + masks (used for evaluation)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ── HEALTHY DATASET (Training) ─────────────────────────────────
class HealthyDataset(Dataset):
    """
    Loads healthy brain slices for training.
    The model ONLY sees healthy data during training — this is the
    core principle of unsupervised anomaly detection.
    """

    def __init__(self, data_dir, split="train", val_fraction=0.1, seed=42):
        """
        Args:
            data_dir     : path to folder containing .npy files
            split        : "train" or "val"
            val_fraction : fraction of healthy slices reserved for validation
            seed         : random seed for reproducibility
        """
        # Load all healthy slices from disk
        slices_path = f"{data_dir}/healthy_slices.npy"
        print(f"Loading healthy slices from {slices_path}...")
        all_slices = np.load(slices_path)  # shape: (84216, 128, 128)
        print(f"Loaded {len(all_slices)} healthy slices")

        # Split into train and validation sets
        train_slices, val_slices = train_test_split(
            all_slices, test_size=val_fraction,
            random_state=seed, shuffle=True
        )

        self.slices = train_slices if split == "train" else val_slices
        print(f"Split '{split}': {len(self.slices)} slices")

    def __len__(self):
        """Returns total number of slices in this split."""
        return len(self.slices)

    def __getitem__(self, idx):
        """
        Returns one slice as a PyTorch tensor.
        Shape: (1, 128, 128)
        """
        slice_2d = self.slices[idx]  # shape: (128, 128)

        # Add channel dimension and convert to tensor
        tensor = torch.tensor(slice_2d, dtype=torch.float32).unsqueeze(0)
        
        # ==========================================
        # CRITICAL WGAN-GP NORMALIZATION FIX
        # ==========================================
        # 1. Scale to [0, 1] safely (avoid division by zero)
        t_min, t_max = tensor.min(), tensor.max()
        if t_max > t_min:
            tensor = (tensor - t_min) / (t_max - t_min)
            
        # 2. Scale to [-1, 1] to perfectly match Generator's Tanh() output
        tensor = (tensor * 2.0) - 1.0
        # ==========================================

        return tensor


# ── ANOMALY DATASET (Evaluation) ──────────────────────────────
class AnomalyDataset(Dataset):
    """
    Loads tumor slices AND their segmentation masks for evaluation.
    Used ONLY at test time — never during training.
    """

    def __init__(self, data_dir):
        """
        Args:
            data_dir: path to folder containing .npy files
        """
        print(f"Loading tumor slices and masks...")
        self.slices = np.load(f"{data_dir}/tumor_slices.npy")
        self.masks  = np.load(f"{data_dir}/tumor_masks.npy")
        print(f"Loaded {len(self.slices)} tumor slices with masks")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        """
        Returns one tumor slice and its mask as PyTorch tensors.
        Both shapes: (1, 128, 128)
        """
        slice_2d = self.slices[idx]
        mask_2d  = self.masks[idx]

        slice_tensor = torch.tensor(slice_2d, dtype=torch.float32).unsqueeze(0)
        mask_tensor  = torch.tensor(mask_2d,  dtype=torch.float32).unsqueeze(0)

        # ==========================================
        # CRITICAL WGAN-GP NORMALIZATION FIX
        # ==========================================
        t_min, t_max = slice_tensor.min(), slice_tensor.max()
        if t_max > t_min:
            slice_tensor = (slice_tensor - t_min) / (t_max - t_min)
        slice_tensor = (slice_tensor * 2.0) - 1.0
        # ==========================================

        # Binarise mask: any tumor label > 0 becomes 1
        mask_tensor = (mask_tensor > 0).float()

        return slice_tensor, mask_tensor


# ── DATALOADER FACTORY ─────────────────────────────────────────
def get_dataloaders(data_dir, batch_size=16, num_workers=0):
    """
    Creates and returns all three DataLoaders needed for training.
    """
    train_dataset = HealthyDataset(data_dir, split="train")
    val_dataset   = HealthyDataset(data_dir, split="val")
    test_dataset  = AnomalyDataset(data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           
        num_workers=num_workers,
        pin_memory=True         
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,          
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ── QUICK TEST ─────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run this file directly to verify the DataLoader works correctly.
    Usage: python src/data/dataset.py
    """
    import os

    DATA_DIR = r"C:\Users\Ronak Daniel\Documents\atml-brain-anomaly\data\processed"

    print("=" * 50)
    print("Testing DataLoaders...")
    print("=" * 50)

    train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR)

    # Test train loader
    batch = next(iter(train_loader))
    print(f"\nTrain batch shape : {batch.shape}")
    print(f"Value range       : [{batch.min():.3f}, {batch.max():.3f}]")

    # Test test loader
    slices, masks = next(iter(test_loader))
    print(f"\nTest slice shape  : {slices.shape}")
    print(f"Test mask shape   : {masks.shape}")
    print(f"Unique mask values: {masks.unique()}")

    print("\n✅ DataLoader test passed!")