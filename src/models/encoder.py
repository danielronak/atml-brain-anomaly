import torch
import torch.nn as nn

class Encoder3D(nn.Module):
    """
    Takes a 3D MRI volume (Real or Fake) and compresses it down into 
    a 1D latent vector (z) of size 128. This is the core of f-AnoGAN.
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
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
        
        # Compress the 3D features into the 128-dimensional latent space
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4 * 4, self.latent_dim)
        )

    def forward(self, x):
        features = self.conv_blocks(x)
        z = self.fc(features)
        return z