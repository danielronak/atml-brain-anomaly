"""
3D PatchGAN Discriminator
--------------------------
Classifies overlapping 3D patches as real or fake.
Shared across Model 2 (CNN Proper) and Model 3 (Swin-UNET GAN).

Design decisions:
  - InstanceNorm3d (not BatchNorm): correct at batch_size=1–2
  - SpectralNorm on all Conv layers: Lipschitz regularisation,
    reduces need for aggressive WGAN-GP lambda
  - Output is a 3D grid of patch scores — not a single scalar
  - LeakyReLU(0.2) throughout — standard for discriminators

No Sigmoid: WGAN-GP discriminator is unconstrained (critic).
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchDiscriminator3D(nn.Module):
    """
    3D PatchGAN discriminator with spectral norm + instance norm.

    Input:  (B, C, D, H, W)  — C=2 for dual modality
    Output: (B, 1, D', H', W') — grid of patch realness scores
    """

    def __init__(self, in_channels: int = 2, features: int = 32):
        super().__init__()

        def disc_block(in_ch, out_ch, normalize=True):
            layers = [spectral_norm(nn.Conv3d(in_ch, out_ch, 4, 2, 1, bias=False))]
            if normalize:
                layers.append(nn.InstanceNorm3d(out_ch, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            # No norm on first layer — standard PatchGAN convention
            *disc_block(in_channels, features, normalize=False),       # /2
            *disc_block(features, features * 2),                       # /4
            *disc_block(features * 2, features * 4),                   # /8
            *disc_block(features * 4, features * 8),                   # /16
            # Final layer — patch score map (no activation, WGAN critic)
            spectral_norm(nn.Conv3d(features * 8, 1, kernel_size=3, padding=1)),
        )

        # Feature extraction hook — used for izi_f encoder feature loss
        self._features = None
        self.net[-2].register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self._features = output

    def forward(self, x: torch.Tensor):
        scores = self.net(x)
        features = self._features
        return scores, features


if __name__ == "__main__":
    import yaml

    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing PatchDiscriminator3D on {device}")

    in_ch = config["swin"]["in_channels"]
    D = PatchDiscriminator3D(in_channels=in_ch).to(device)

    dummy = torch.randn(1, in_ch, 128, 128, 128).to(device)
    scores, features = D(dummy)

    print(f"  Input shape:   {dummy.shape}")
    print(f"  Score shape:   {scores.shape}")
    print(f"  Feature shape: {features.shape}")
    print(f"  VRAM used:     {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("✅ PatchDiscriminator3D smoke test passed.")
