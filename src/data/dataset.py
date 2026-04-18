"""
Dataset — IXI (Training) + BraTS 2021 (Testing)
-------------------------------------------------
Supports single-modality (T1 or T2) and dual-modality (T1+T2 stacked)
for both the IXI healthy training set and the BraTS 2021 test set.

Key changes from original dataset.py:
  - Added IXI dataloader (healthy training data — no tumors)
  - Added ScaleIntensityRangePercentilesd (robust vs. ScaleIntensityd)
  - Added CropForegroundd (removes black skull border before resize)
  - Added dual-modality support (in_channels=2: [T1, T2])
  - Added BraTS test loader that also returns segmentation masks
  - Resolution now pulled from config (128³ default)
"""

import os
import glob
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    CropForegroundd,
    Resized,
    ConcatItemsd,
    DeleteItemsd,
    ToTensord,
)
from monai.data import Dataset as MonaiDataset, CacheDataset


# ──────────────────────────────────────────────────────────────
# IXI — Healthy Training Data
# ──────────────────────────────────────────────────────────────

def _ixi_paired_subjects(ixi_dir: str):
    """
    Return a list of dicts with paired T1 + T2 paths.
    IXI naming: IXI012-HH-1211-T1.nii.gz
    """
    t1_dir = Path(ixi_dir) / "T1"
    t2_dir = Path(ixi_dir) / "T2"

    t1_files = {f.stem.replace(".nii", "").replace("-T1", ""): f
                for f in t1_dir.glob("*.nii.gz")}
    t2_files = {f.stem.replace(".nii", "").replace("-T2", ""): f
                for f in t2_dir.glob("*.nii.gz")}

    paired_ids = sorted(set(t1_files) & set(t2_files))
    print(f"[IXI] Found {len(paired_ids)} paired T1+T2 subjects.")

    return [{"t1": str(t1_files[sid]), "t2": str(t2_files[sid])} for sid in paired_ids]


def _ixi_single_subjects(ixi_dir: str, modality: Literal["T1", "T2"] = "T1"):
    files = sorted(Path(ixi_dir).glob(f"{modality}/*.nii.gz"))
    print(f"[IXI] Found {len(files)} {modality} scans.")
    return [{"image": str(f)} for f in files]


def _ixi_transforms_dual(config: dict):
    """Dual T1+T2 transform: loads both, preprocesses identically, stacks to 2-channel."""
    res = tuple(config["data"]["resolution"])
    pct_lo = config["data"]["intensity_percentile_low"]
    pct_hi = config["data"]["intensity_percentile_high"]

    return Compose([
        LoadImaged(keys=["t1", "t2"]),
        EnsureChannelFirstd(keys=["t1", "t2"]),
        Spacingd(keys=["t1", "t2"],
                 pixdim=tuple(config["data"]["spacing"]),
                 mode=("bilinear", "bilinear")),
        Orientationd(keys=["t1", "t2"], axcodes="RAS"),
        CropForegroundd(keys=["t1", "t2"], source_key="t1"),
        ScaleIntensityRangePercentilesd(
            keys=["t1", "t2"],
            lower=pct_lo, upper=pct_hi,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        Resized(keys=["t1", "t2"], spatial_size=res),
        # Stack T1 and T2 into a single 2-channel tensor
        ConcatItemsd(keys=["t1", "t2"], name="image", dim=0),
        DeleteItemsd(keys=["t1", "t2"]),
        ToTensord(keys=["image"]),
    ])


def _ixi_transforms_single(config: dict, modality: str = "t1"):
    res = tuple(config["data"]["resolution"])
    pct_lo = config["data"]["intensity_percentile_low"]
    pct_hi = config["data"]["intensity_percentile_high"]

    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=tuple(config["data"]["spacing"]), mode="bilinear"),
        Orientationd(keys=["image"], axcodes="RAS"),
        CropForegroundd(keys=["image"], source_key="image"),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=pct_lo, upper=pct_hi,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        Resized(keys=["image"], spatial_size=res),
        ToTensord(keys=["image"]),
    ])


def get_ixi_dataloaders(config: dict):
    """
    Returns (train_loader, val_loader) for IXI healthy brains.
    Dual-modality if config['data']['modality'] == 'dual'.
    """
    ixi_dir = config["data"]["ixi_dir"]
    modality = config["data"]["modality"]
    split = config["data"]["train_val_split"]
    bs = config["data"]["batch_size"]
    nw = config["data"]["num_workers"]

    if modality == "dual":
        subjects = _ixi_paired_subjects(ixi_dir)
        transforms = _ixi_transforms_dual(config)
    else:
        subjects = _ixi_single_subjects(ixi_dir, modality.upper())
        transforms = _ixi_transforms_single(config, modality)

    n_train = int(len(subjects) * split)
    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:]

    print(f"[IXI] Train: {len(train_subjects)} | Val: {len(val_subjects)}")

    # CacheDataset: preprocessed volumes cached in RAM — faster training
    # Use MonaiDataset if RAM is tight on Colab
    train_ds = CacheDataset(data=train_subjects, transform=transforms,
                             cache_rate=0.5, num_workers=nw)
    val_ds = CacheDataset(data=val_subjects, transform=transforms,
                           cache_rate=1.0, num_workers=nw)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                               num_workers=nw, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                             num_workers=nw, pin_memory=False)

    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────
# BraTS 2021 — Test Data (with segmentation masks)
# ──────────────────────────────────────────────────────────────

def _brats_subjects(brats_dir: str, n_patients: int = 50, modality: str = "dual"):
    """
    Find BraTS patient folders and return list of dicts with image + mask paths.
    For dual modality: loads T1 + T2 (not T1ce, for maximum modality alignment with IXI).
    For single: loads T1ce.
    """
    data_dir = Path(brats_dir) / "BraTS2021_Training_Data"
    patient_dirs = sorted(data_dir.glob("BraTS2021_*"))

    if not patient_dirs:
        raise FileNotFoundError(f"No BraTS patient folders found in {data_dir}")

    print(f"[BraTS] Found {len(patient_dirs)} patients. Using first {n_patients}.")
    patient_dirs = patient_dirs[:n_patients]

    subjects = []
    for p in patient_dirs:
        pid = p.name
        seg = p / f"{pid}_seg.nii.gz"
        if not seg.exists():
            continue

        if modality == "dual":
            t1 = p / f"{pid}_t1.nii.gz"   # T1 (not T1ce) to match IXI
            t2 = p / f"{pid}_t2.nii.gz"
            if t1.exists() and t2.exists():
                subjects.append({"t1": str(t1), "t2": str(t2), "mask": str(seg)})
        else:
            img = p / f"{pid}_t1ce.nii.gz"
            if img.exists():
                subjects.append({"image": str(img), "mask": str(seg)})

    print(f"[BraTS] Loaded {len(subjects)} valid test subjects.")
    return subjects


def _brats_transforms_dual(config: dict):
    res = tuple(config["data"]["resolution"])
    pct_lo = config["data"]["intensity_percentile_low"]
    pct_hi = config["data"]["intensity_percentile_high"]

    return Compose([
        LoadImaged(keys=["t1", "t2", "mask"]),
        EnsureChannelFirstd(keys=["t1", "t2", "mask"]),
        Spacingd(keys=["t1", "t2", "mask"],
                 pixdim=tuple(config["data"]["spacing"]),
                 mode=("bilinear", "bilinear", "nearest")),
        Orientationd(keys=["t1", "t2", "mask"], axcodes="RAS"),
        CropForegroundd(keys=["t1", "t2", "mask"], source_key="t1"),
        ScaleIntensityRangePercentilesd(
            keys=["t1", "t2"],
            lower=pct_lo, upper=pct_hi,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        Resized(keys=["t1", "t2", "mask"], spatial_size=res,
                mode=("trilinear", "trilinear", "nearest")),
        ConcatItemsd(keys=["t1", "t2"], name="image", dim=0),
        DeleteItemsd(keys=["t1", "t2"]),
        ToTensord(keys=["image", "mask"]),
    ])


def get_brats_test_loader(config: dict):
    """
    Returns a DataLoader for BraTS test set (batch_size=1, no shuffle).
    Each item: {"image": Tensor[C, D, H, W], "mask": Tensor[1, D, H, W]}
    """
    brats_dir = config["data"]["brats_dir"]
    modality = config["data"]["modality"]
    n = config["data"]["n_test_patients"]
    nw = config["data"]["num_workers"]

    subjects = _brats_subjects(brats_dir, n_patients=n, modality=modality)

    if modality == "dual":
        transforms = _brats_transforms_dual(config)
    else:
        raise NotImplementedError("Single-modality BraTS loader not needed — use dual.")

    ds = MonaiDataset(data=subjects, transform=transforms)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=nw)


# ──────────────────────────────────────────────────────────────
# Legacy BraTS loader — backward compat with existing naive baseline
# ──────────────────────────────────────────────────────────────

def get_brats_dataloader(data_dir: str, batch_size: int = 2):
    """Original loader kept for CNN naive baseline compatibility."""
    from monai.transforms import ScaleIntensityd
    from monai.data import Dataset as MonaiDataset

    search_path = os.path.join(data_dir, "**", "*_t1ce.nii.gz")
    images = sorted(glob.glob(search_path, recursive=True))

    if not images:
        raise FileNotFoundError(f"No *_t1ce.nii.gz files found in {data_dir}")

    print(f"[BraTS Legacy] Found {len(images)} T1ce volumes.")
    data_dicts = [{"image": img} for img in images]

    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(64, 64, 64))
    ])

    dataset = MonaiDataset(data=data_dicts, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)