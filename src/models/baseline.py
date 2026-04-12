import torch
import torch.nn as nn

class Generator3D(nn.Module):
    """
    Takes a 1D random noise vector (latent space) and upsamples it 
    into a 3D MRI volume of shape (1, 64, 64, 64).
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Initial linear layer to project the latent vector into a 3D spatial tensor
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4 * 4)
        
        self.conv_blocks = nn.Sequential(
            # Input: (Batch, 512, 4, 4, 4)
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # Input: (Batch, 256, 8, 8, 8)
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # Input: (Batch, 128, 16, 16, 16)
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # Input: (Batch, 64, 32, 32, 32) -> Output: (Batch, 1, 64, 64, 64)
            nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Tanh scales pixel values between -1 and 1
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4, 4) # Reshape into a 3D volume
        return self.conv_blocks(x)


class Discriminator3D(nn.Module):
    """
    Takes a 3D MRI volume (Real or Fake) and downsamples it to predict
    whether it is real (1) or fake (0).
    """
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            # Input: (Batch, 1, 64, 64, 64)
            nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Input: (Batch, 64, 32, 32, 32)
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Input: (Batch, 128, 16, 16, 16)
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Input: (Batch, 256, 8, 8, 8) -> Output: (Batch, 512, 4, 4, 4)
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4 * 4, 1),
            nn.Sigmoid() # Output a probability (0 to 1)
        )

    def forward(self, x):
        # We extract 'features' separately because f-AnoGAN relies on 
        # intermediate feature mapping to calculate anomaly scores later!
        features = self.conv_blocks(x)
        validity = self.fc(features)
        return validity, features