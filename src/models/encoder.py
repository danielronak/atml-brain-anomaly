import torch
import torch.nn as nn

class Encoder3D(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.conv_blocks = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4 * 4, self.latent_dim)
        )

    def forward(self, x):
        features = self.conv_blocks(x)
        z = self.fc(features)
        return z