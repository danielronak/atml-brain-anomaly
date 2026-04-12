import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels=1, features=32, z_dim=512):
        super().__init__()
        
        # Architecture mirrors the Discriminator but outputs a latent vector
        self.net = nn.Sequential(
            # Input: (batch, 1, 128, 128)
            nn.Conv2d(in_channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (batch, 32, 64, 64)

            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (batch, 64, 32, 32)

            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (batch, 128, 16, 16)

            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (batch, 256, 8, 8)
            
            nn.Conv2d(features * 8, features * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (batch, 512, 4, 4)

            # Final mapping layer: compress down to the exact z_dim (512)
            nn.Conv2d(features * 16, z_dim, 4, 1, 0, bias=False)
            # Final Size: (batch, 512, 1, 1)
        )

    def forward(self, x):
        return self.net(x)