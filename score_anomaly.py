import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from src.models.baseline import Generator3D, Discriminator3D
from src.models.encoder import Encoder3D # Make sure this matches your actual import!

def load_sample(image_path, mask_path):
    """Loads and preprocesses a single MRI volume and its mask to 64x64x64."""
    vol_data = nib.load(image_path).get_fdata()
    mask_data = nib.load(mask_path).get_fdata()
    
    vol_data = (vol_data - vol_data.min()) / (vol_data.max() - vol_data.min() + 1e-8)
    
    vol_tensor = torch.tensor(vol_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    mask_data = (mask_data > 0).astype(np.float32)
    mask_tensor = torch.tensor(mask_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    vol_resized = F.interpolate(vol_tensor, size=(64, 64, 64), mode='trilinear', align_corners=False)
    mask_resized = F.interpolate(mask_tensor, size=(64, 64, 64), mode='nearest')
    
    return vol_resized, mask_resized

def calculate_dice(pred_mask, true_mask, threshold=0.1):
    """Calculates the Dice Coefficient between the prediction and ground truth."""
    pred_binary = (pred_mask > threshold).float()
    
    intersection = (pred_binary * true_mask).sum()
    total = pred_binary.sum() + true_mask.sum()
    
    if total == 0:
        return 1.0 
    
    return (2. * intersection / total).item()

def score_anomaly():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    latent_dim = 128
    
    print("Loading Baseline Models...")
    generator = Generator3D(latent_dim).to(device)
    discriminator = Discriminator3D().to(device)
    encoder = Encoder3D(latent_dim).to(device)

    try:
        generator.load_state_dict(torch.load("checkpoints/generator_baseline_final.pth", map_location=device))
        discriminator.load_state_dict(torch.load("checkpoints/discriminator_baseline_final.pth", map_location=device))
        encoder.load_state_dict(torch.load("checkpoints/encoder_baseline_final.pth", map_location=device))
    except FileNotFoundError as e:
        print(f"Error loading weights: {e}")
        return

    generator.eval()
    discriminator.eval()
    encoder.eval()

    # 3. Find a Test Image (UPDATED FOR BRATS 2021)
    test_data_dir = "data/raw/BraTS2021_Training_Data/"
    
    # Grab all patient folders
    patient_folders = sorted(glob.glob(os.path.join(test_data_dir, "BraTS2021_*")))
    
    if not patient_folders:
        print(f"Could not find any patient folders in {test_data_dir}")
        print("Make sure you dragged the folder exactly as instructed!")
        return
        
    # Let's grab the 5th patient just to ensure we get a good solid tumor example
    target_patient_dir = patient_folders[5]
    patient_id = os.path.basename(target_patient_dir)
    
    # In BraTS 2021, the files are named exactly after the patient folder
    test_img_path = os.path.join(target_patient_dir, f"{patient_id}_t1ce.nii.gz")
    test_label_path = os.path.join(target_patient_dir, f"{patient_id}_seg.nii.gz")
    
    print(f"Scoring Patient: {patient_id}")

    real_img, true_mask = load_sample(test_img_path, test_label_path)
    real_img = real_img.to(device)
    true_mask = true_mask.to(device)

    print("Calculating Anomaly...")
    with torch.no_grad():
        z_guess = encoder(real_img)
        fake_img = generator(z_guess)
        residual_img = torch.abs(real_img - fake_img)
        
        _,real_features = discriminator(real_img)
        _,fake_features = discriminator(fake_img)
        
        loss_residual = F.mse_loss(real_img, fake_img)
        loss_feature = F.mse_loss(real_features, fake_features)
        total_anomaly_score = loss_residual + loss_feature

    dice_score = calculate_dice(residual_img, true_mask, threshold=0.1)

    print("-" * 30)
    print(f"Total Anomaly Score: {total_anomaly_score.item():.4f}")
    print(f"Dice Coefficient: {dice_score:.4f}")
    print("-" * 30)

    print("Plotting the results...")
    slice_idx = 32
    
    real_slice = real_img.squeeze().cpu().numpy()[slice_idx, :, :]
    fake_slice = fake_img.squeeze().cpu().numpy()[slice_idx, :, :]
    residual_slice = residual_img.squeeze().cpu().numpy()[slice_idx, :, :]
    true_mask_slice = true_mask.squeeze().cpu().numpy()[slice_idx, :, :]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Baseline Anomaly Detection - {patient_id}", fontsize=14)
    
    axes[0].imshow(real_slice, cmap='gray')
    axes[0].set_title("Real Image (Tumor)")
    axes[0].axis('off')
    
    axes[1].imshow(fake_slice, cmap='gray')
    axes[1].set_title("Reconstructed (Healthy)")
    axes[1].axis('off')
    
    axes[2].imshow(residual_slice, cmap='hot')
    axes[2].set_title("Predicted Anomaly (Residual)")
    axes[2].axis('off')
    
    axes[3].imshow(true_mask_slice, cmap='Reds')
    axes[3].set_title("True Tumor Mask")
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    score_anomaly()