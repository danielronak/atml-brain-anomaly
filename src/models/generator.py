"""
f-AnoGAN Generator (CNN Baseline)
-----------------------------------
Architecture: Encoder-Decoder U-Net style CNN

What this does:
  Takes a brain scan as input and reconstructs it as a "healthy" version.
  During training it only ever sees healthy scans, so it learns to
  reconstruct healthy anatomy. When given a tumor scan at test time,
  it reconstructs the healthy version — the difference reveals the tumor.

Architecture details:
  - Encoder: progressively downsamples 128x128 → 8x8 (extracts features)
  - Bottleneck: compressed representation of the image (latent space)
  - Decoder: progressively upsamples 8x8 → 128x128 (reconstructs image)
  - Skip connections: pass encoder features to decoder (preserves detail)

Based on: Schlegl et al., f-AnoGAN (2019)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    A single convolutional block: Conv → BatchNorm → LeakyReLU

    Why LeakyReLU and not ReLU?
      Regular ReLU kills any negative value (sets to 0). In GANs this
      causes "dying neurons" — units that never activate again.
      LeakyReLU allows a small negative slope (0.2) to keep gradients flowing.

    Why BatchNorm?
      Normalises activations within each batch to keep training stable.
      Without it, GAN training tends to collapse or oscillate wildly.
    """

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


class UpConvBlock(nn.Module):
    """
    A single upsampling block: ConvTranspose → BatchNorm → ReLU → Dropout

    ConvTranspose2d is the reverse of Conv2d — it increases spatial size.
    Think of it as "unpixelating" the compressed representation.

    Why Dropout in the decoder?
      Acts as regularisation — prevents the generator from memorising
      training images instead of learning general healthy brain anatomy.
    """

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    """
    U-Net style Generator for f-AnoGAN.

    Input:  (batch, 1, 128, 128) — grayscale MRI slice
    Output: (batch, 1, 128, 128) — reconstructed healthy slice

    Encoder path (downsampling):
      128x128 → 64x64 → 32x32 → 16x16 → 8x8 → 4x4 → 2x2 → 1x1

    Decoder path (upsampling):
      1x1 → 2x2 → 4x4 → 8x8 → 16x16 → 32x32 → 64x64 → 128x128

    Skip connections: each encoder layer connects to its mirror decoder layer.
    This lets the decoder access fine spatial details lost during compression.
    """

    def __init__(self, in_channels=1, features=64):
        super().__init__()

        # ── ENCODER (no BatchNorm on first layer — standard practice) ──
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )                                          # → (batch, 64,  64, 64)

        self.enc2 = ConvBlock(features,     features * 2)   # → (batch, 128, 32, 32)
        self.enc3 = ConvBlock(features * 2, features * 4)   # → (batch, 256, 16, 16)
        self.enc4 = ConvBlock(features * 4, features * 8)   # → (batch, 512,  8,  8)
        self.enc5 = ConvBlock(features * 8, features * 8)   # → (batch, 512,  4,  4)
        self.enc6 = ConvBlock(features * 8, features * 8)   # → (batch, 512,  2,  2)
        self.enc7 = ConvBlock(features * 8, features * 8)   # → (batch, 512,  1,  1)

        # ── DECODER (channels doubled because of skip connections) ──────
        # Each decoder block receives its own output + skip from encoder
        # so input channels are doubled (hence features*8*2 etc.)

        self.dec1 = UpConvBlock(features * 8,     features * 8, dropout=True)   # → (batch, 512, 2, 2)
        self.dec2 = UpConvBlock(features * 8 * 2, features * 8, dropout=True)   # → (batch, 512, 4, 4)
        self.dec3 = UpConvBlock(features * 8 * 2, features * 8, dropout=True)   # → (batch, 512, 8, 8)
        self.dec4 = UpConvBlock(features * 8 * 2, features * 4)                 # → (batch, 256,16,16)
        self.dec5 = UpConvBlock(features * 4 * 2, features * 2)                 # → (batch, 128,32,32)
        self.dec6 = UpConvBlock(features * 2 * 2, features)                     # → (batch,  64,64,64)

        # Final layer: upsample to original size, Tanh squashes to [-1, 1]
        self.dec7 = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()   # output in [-1,1] — matches our normalised input
        )

    def forward(self, x):
        # Encoder — save each output for skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)

        # Decoder — concatenate skip connections along channel dimension
        d1 = self.dec1(e7)
        d2 = self.dec2(torch.cat([d1, e6], dim=1))
        d3 = self.dec3(torch.cat([d2, e5], dim=1))
        d4 = self.dec4(torch.cat([d3, e4], dim=1))
        d5 = self.dec5(torch.cat([d4, e3], dim=1))
        d6 = self.dec6(torch.cat([d5, e2], dim=1))
        d7 = self.dec7(torch.cat([d6, e1], dim=1))

        return d7


# ── QUICK TEST ────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Generator on {device}...")

    model = Generator().to(device)

    # Simulate a batch of 4 MRI slices
    dummy_input = torch.randn(4, 1, 128, 128).to(device)
    output = model(dummy_input)

    print(f"Input shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Output range : [{output.min():.3f}, {output.max():.3f}]")

    # Count parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters   : {params:,}")
    print("✅ Generator test passed!")