import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=32):
        super().__init__()
        
        self.net = nn.Sequential(
            # Input: (batch, 1, 128, 128)
            nn.Conv2d(in_channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (batch, 32, 64, 64)

            # WGAN-GP FIX: Replaced BatchNorm with InstanceNorm
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (batch, 64, 32, 32)

            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (batch, 128, 16, 16)

            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (batch, 256, 8, 8)
            
            nn.Conv2d(features * 8, features * 16, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (batch, 512, 4, 4)

            # Final layer flattens to a single value
            # Note: No Sigmoid activation here! This is correct for WGAN.
            nn.Conv2d(features * 16, 1, 4, 1, 0, bias=False),
            # Final Size: (batch, 1, 1, 1)
        )

    def forward(self, x):
        # Flatten the (batch, 1, 1, 1) output to (batch, 1)
        return self.net(x).view(-1, 1)