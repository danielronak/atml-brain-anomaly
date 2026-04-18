import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=512, features=32, out_channels=1):
        super().__init__()
        
        self.net = nn.Sequential(
            # Input: (batch, 512, 1, 1)
            nn.ConvTranspose2d(z_dim, features * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 16),
            nn.ReLU(True),
            # Size: (batch, 512, 4, 4)

            nn.ConvTranspose2d(features * 16, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # Size: (batch, 256, 8, 8)

            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # Size: (batch, 128, 16, 16)

            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # Size: (batch, 64, 32, 32)

            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            # Size: (batch, 32, 64, 64)

            nn.ConvTranspose2d(features, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Final Size: (batch, 1, 128, 128)
        )

    def forward(self, z):
        return self.net(z)