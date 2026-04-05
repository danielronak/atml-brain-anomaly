"""
BraTS 2021 Preprocessing Script
--------------------------------
What this does:
  1. Loads each patient's FLAIR scan (3D MRI volume)
  2. Extracts 2D slices along the axial axis (top-down view)
  3. Filters out empty/near-empty slices (no brain tissue)
  4. Normalises pixel values to [0, 1]
  5. Resizes to 128x128
  6. Separates healthy slices from tumor slices
  7. Saves everything as .npy arrays ready for training

Why FLAIR?
  FLAIR (Fluid Attenuated Inversion Recovery) makes tumors appear
  bright white against grey brain tissue — easiest modality for
  anomaly detection.
"""

import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
from tqdm import tqdm

# ── CONFIGURATION ──────────────────────────────────────────────
RAW_DATA_DIR = r"C:\Users\Ronak Daniel\Documents\brats2021_raw"
OUTPUT_DIR   = r"C:\Users\Ronak Daniel\Documents\atml-brain-anomaly\data\processed"
IMG_SIZE     = 128       # resize all slices to 128x128
MIN_BRAIN_FRACTION = 0.01  # skip slices where less than 1% of pixels have brain tissue

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── HELPER FUNCTIONS ───────────────────────────────────────────
def load_nifti(path):
    """Load a .nii.gz file and return it as a numpy array."""
    return nib.load(path).get_fdata()

def normalise(volume):
    """Normalise a volume to [0, 1] using its own min/max."""
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin == 0:
        return volume
    return (volume - vmin) / (vmax - vmin)

def resize_slice(slice_2d):
    """Resize a 2D slice to IMG_SIZE x IMG_SIZE."""
    return resize(slice_2d, (IMG_SIZE, IMG_SIZE),
                  anti_aliasing=True, preserve_range=True)

def has_brain(slice_2d, threshold=MIN_BRAIN_FRACTION):
    """Return True if slice contains enough brain tissue to be useful."""
    return (slice_2d > 0.1).mean() > threshold

def has_tumor(seg_slice):
    """Return True if the segmentation mask has any tumor pixels."""
    return seg_slice.max() > 0

# ── MAIN PROCESSING LOOP ───────────────────────────────────────
healthy_slices = []
tumor_slices   = []
healthy_masks  = []
tumor_masks    = []

# Get all patient folders
patient_dirs = sorted([
    d for d in os.listdir(RAW_DATA_DIR)
    if os.path.isdir(os.path.join(RAW_DATA_DIR, d))
    and d.startswith("BraTS2021_")
])

print(f"Found {len(patient_dirs)} patient folders")
print(f"Processing... this will take several minutes\n")

for patient in tqdm(patient_dirs):
    patient_path = os.path.join(RAW_DATA_DIR, patient)

    flair_path = os.path.join(patient_path, f"{patient}_flair.nii.gz")
    seg_path   = os.path.join(patient_path, f"{patient}_seg.nii.gz")

    # Skip if files missing
    if not os.path.exists(flair_path) or not os.path.exists(seg_path):
        continue

    # Load volumes
    flair_vol = normalise(load_nifti(flair_path))  # shape: (240, 240, 155)
    seg_vol   = load_nifti(seg_path)               # shape: (240, 240, 155)

    # Extract 2D axial slices (axis=2 is the top-down view)
    num_slices = flair_vol.shape[2]
    for i in range(num_slices):
        flair_slice = flair_vol[:, :, i]
        seg_slice   = seg_vol[:, :, i]

        # Skip slices with too little brain tissue
        if not has_brain(flair_slice):
            continue

        # Resize to 128x128
        flair_resized = resize_slice(flair_slice)
        seg_resized   = resize_slice(seg_slice)

        # Sort into healthy vs tumor
        if has_tumor(seg_slice):
            tumor_slices.append(flair_resized)
            tumor_masks.append(seg_resized)
        else:
            healthy_slices.append(flair_resized)
            healthy_masks.append(seg_resized)

# ── SAVE RESULTS ───────────────────────────────────────────────
print(f"\nSaving arrays...")

healthy_slices = np.array(healthy_slices, dtype=np.float32)
tumor_slices   = np.array(tumor_slices,   dtype=np.float32)
healthy_masks  = np.array(healthy_masks,  dtype=np.float32)
tumor_masks    = np.array(tumor_masks,    dtype=np.float32)

np.save(os.path.join(OUTPUT_DIR, "healthy_slices.npy"), healthy_slices)
np.save(os.path.join(OUTPUT_DIR, "tumor_slices.npy"),   tumor_slices)
np.save(os.path.join(OUTPUT_DIR, "healthy_masks.npy"),  healthy_masks)
np.save(os.path.join(OUTPUT_DIR, "tumor_masks.npy"),    tumor_masks)

print(f"\n✅ Done!")
print(f"   Healthy slices : {healthy_slices.shape}")
print(f"   Tumor slices   : {tumor_slices.shape}")
print(f"   Saved to       : {OUTPUT_DIR}")