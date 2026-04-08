"""
f-AnoGAN Encoder (CNN Baseline)
-------------------------------
Architecture: CNN Encoder
What this does:
Mirrors the Generator's encoder path. It takes a 128x128 image 
and compresses it down into a 1D latent vector (size 512).
During anomaly scoring, the encoder helps find the closest healthy
latent representation of a given test image.
"""
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Standard convolutional block used in the downsampling path."""
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                      kernel_size=4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    """
    Input:  (batch, 1, 128, 128)
    Output: (batch, 512) — Latent vector
    """
    def __init__(self, in_channels=1, features=64, latent_dim=512):
        super().__init__()
        
        # Mirrors the generator's encoder path exactly
        self.features = nn.Sequential(
            # Layer 1: No batch norm
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),                      # -> (batch, 64,  64, 64)
            ConvBlock(features,     features * 2),                # -> (batch, 128, 32, 32)
            ConvBlock(features * 2, features * 4),                # -> (batch, 256, 16, 16)
            ConvBlock(features * 4, features * 8),                # -> (batch, 512,  8,  8)
            ConvBlock(features * 8, features * 8),                # -> (batch, 512,  4,  4)
            ConvBlock(features * 8, features * 8),                # -> (batch, 512,  2,  2)
            ConvBlock(features * 8, features * 8)                 # -> (batch, 512,  1,  1)
        )
        
        # Flatten the 1x1 spatial grid and map to the final latent dimension
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(features * 8, latent_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.fc(x)

# ── QUICK TEST ────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Encoder on {device}...")
    model = Encoder().to(device)
    
    # Simulate a batch of 4 MRI slices
    dummy_input = torch.randn(4, 1, 128, 128).to(device)
    output = model(dummy_input)
    
    print(f"Input shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters   : {params:,}")
    print(" ✅ Encoder test passed!")