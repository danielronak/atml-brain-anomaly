import os
import glob
import torch
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd,
    Spacingd, 
    Orientationd, 
    ScaleIntensityd, 
    Resized
)
from monai.data import Dataset as MonaiDataset

def get_brats_transforms():
    """
    Standard transform pipeline for 3D Brain MRIs.
    """
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        # Normalize the voxel spacing to 1mm x 1mm x 1mm
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        # Ensure standard Right-Anterior-Superior orientation
        Orientationd(keys=["image"], axcodes="RAS"),
        # Normalize pixel values between 0 and 1
        ScaleIntensityd(keys=["image"]),
        # Downsample to 64x64x64 for rapid prototyping and lower VRAM usage
        Resized(keys=["image"], spatial_size=(64, 64, 64))
    ])

def get_brats_dataloader(data_dir, batch_size=2):
    """
    Finds all T1ce (contrast-enhanced) scans and creates a PyTorch DataLoader.
    For f-AnoGAN, we will initially train on just one modality.
    """
    # Look for all T1ce files in the raw data directory
    search_path = os.path.join(data_dir, "**", "*_t1ce.nii.gz")
    images = sorted(glob.glob(search_path, recursive=True))
    
    if not images:
        raise FileNotFoundError(f"No *_t1ce.nii.gz files found in {data_dir}. Check your path!")
        
    print(f"Found {len(images)} T1ce MRI volumes.")

    # Format for MONAI dictionary transforms
    data_dicts = [{"image": img} for img in images]
    
    transforms = get_brats_transforms()
    dataset = MonaiDataset(data=data_dicts, transform=transforms)
    
    # Create the PyTorch DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return loader