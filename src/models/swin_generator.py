"""
3D Attention U-Net Generator
------------------------------
Custom implementation — no MONAI dependency. Architecturally equivalent to
Swin-UNET for our anomaly detection use case.

Design choices:
  - U-Net skip connections: preserve spatial boundaries (critical for residual maps)
  - Self-attention at the bottleneck (8³ tokens): captures global anatomy context
    without the quadratic cost of full-volume attention
  - DoubleConv blocks with InstanceNorm: stable for generative tasks
  - Tanh output: constrains reconstruction to input normalisation range

Why attention at the bottleneck only?
  For 128³ volumes, the bottleneck is at 8³ = 512 tokens — trivial for attention.
  Full-volume attention (2M tokens) would OOM on any GPU. Swin uses window-based
  attention to solve this; we use the simpler bottleneck approach which achieves
  the same goal (global context) for reconstruction tasks.

VRAM / feature_size guide (batch=2, 128³ volumes):
  feature_size=24 → ~4 GB VRAM (T4 safe)
  feature_size=36 → ~7 GB VRAM (L4 safe)
  feature_size=48 → ~12 GB VRAM (A100 safe)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


# ─────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────

class DoubleConv3D(nn.Module):
    """Two 3×3×3 convolutions with InstanceNorm + LeakyReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down3D(nn.Module):
    """Stride-2 max-pool downsampling followed by DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up3D(nn.Module):
    """Trilinear upsample + skip connection + DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.conv = DoubleConv3D(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd spatial dims from skip (pad if needed)
        diff = [s - x.shape[i+2] for i, s in enumerate(skip.shape[2:])]
        x = F.pad(x, [0, diff[2], 0, diff[1], 0, diff[0]])
        return self.conv(torch.cat([skip, x], dim=1))


class BottleneckAttention3D(nn.Module):
    """
    Multi-head self-attention over the 3D bottleneck tensor.

    Operates on flattened spatial tokens, so only feasible at small resolutions.
    For 128³ input with 4 downsampling steps → bottleneck at 8³ = 512 tokens.

    Args:
        channels: number of feature channels (= 8 × feature_size)
        num_heads: number of attention heads (channels must be divisible by num_heads)
    """
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        assert channels % num_heads == 0, \
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.norm    = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.attn    = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads,
                                              batch_first=True)
        self.out_proj = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        h = self.norm(x)
        # Flatten spatial → sequence: (B, N, C) where N = D*H*W
        tokens = h.flatten(2).permute(0, 2, 1)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, D, H, W)
        return x + self.out_proj(attn_out)   # residual connection


# ─────────────────────────────────────────────────────────────────
# Full 3D Attention U-Net
# ─────────────────────────────────────────────────────────────────

class AttentionUNet3D(nn.Module):
    """
    3D Attention U-Net generator for brain MRI reconstruction.

    Architecture:
      Encoder: 4× DoubleConv + MaxPool (128→64→32→16→8)
      Bottleneck: DoubleConv + self-attention (512 tokens @ feature_size*8 dim)
      Decoder: 4× bilinear up + skip + DoubleConv (8→16→32→64→128)
      Output: 1×1×1 conv + Tanh

    Args:
        in_channels:    number of input channels (2 for dual T1+T2)
        out_channels:   number of output channels (2 for dual T1+T2)
        feature_size:   base channel multiplier (24=T4, 36=L4, 48=A100)
        use_checkpoint: gradient checkpointing (saves VRAM, costs time)
    """
    def __init__(self, in_channels: int, out_channels: int,
                 feature_size: int = 24, use_checkpoint: bool = True):
        super().__init__()
        f = feature_size
        self.use_checkpoint = use_checkpoint

        # Encoder
        self.inc   = DoubleConv3D(in_channels, f)       # 128³ → f
        self.down1 = Down3D(f,     f * 2)               # 64³  → 2f
        self.down2 = Down3D(f * 2, f * 4)               # 32³  → 4f
        self.down3 = Down3D(f * 4, f * 8)               # 16³  → 8f
        self.down4 = Down3D(f * 8, f * 8)               # 8³   → 8f (bottleneck)

        # Bottleneck: DoubleConv + multi-head self-attention
        # 8³ = 512 spatial tokens — trivial for attention
        num_heads = max(1, (f * 8) // 32)               # 1 head per 32 channels
        num_heads = min(num_heads, f * 8)
        # Ensure divisibility
        while (f * 8) % num_heads != 0:
            num_heads -= 1
        self.bottle_conv = DoubleConv3D(f * 8, f * 8)
        self.bottle_attn = BottleneckAttention3D(f * 8, num_heads=num_heads)

        # Decoder — skip input = encoder output + upsampled = (8f + 8f = 16f) etc.
        self.up1 = Up3D(f * 16, f * 4)                  # 16³
        self.up2 = Up3D(f * 8,  f * 2)                  # 32³
        self.up3 = Up3D(f * 4,  f)                      # 64³
        self.up4 = Up3D(f * 2,  f)                      # 128³

        # Output projection
        self.out_conv = nn.Sequential(
            nn.Conv3d(f, out_channels, kernel_size=1),
            nn.Tanh(),
        )

    def _bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bottle_conv(x)
        x = self.bottle_attn(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        e0 = self.inc(x)          # (B, f,   128, 128, 128)
        e1 = self.down1(e0)       # (B, 2f,   64,  64,  64)
        e2 = self.down2(e1)       # (B, 4f,   32,  32,  32)
        e3 = self.down3(e2)       # (B, 8f,   16,  16,  16)
        e4 = self.down4(e3)       # (B, 8f,    8,   8,   8)

        # Bottleneck (with optional gradient checkpointing)
        if self.use_checkpoint and self.training:
            b = grad_checkpoint(self._bottleneck, e4, use_reentrant=False)
        else:
            b = self._bottleneck(e4)

        # Decoder path (skip connections)
        d1 = self.up1(b,  e3)     # (B, 4f,   16,  16,  16)
        d2 = self.up2(d1, e2)     # (B, 2f,   32,  32,  32)
        d3 = self.up3(d2, e1)     # (B,  f,   64,  64,  64)
        d4 = self.up4(d3, e0)     # (B,  f,  128, 128, 128)

        return self.out_conv(d4)  # (B, out_ch, 128, 128, 128)


# ─────────────────────────────────────────────────────────────────
# Factory function — keeps the same interface as before
# ─────────────────────────────────────────────────────────────────

def get_swin_generator(config: dict) -> AttentionUNet3D:
    """
    Build the Attention U-Net generator from configs/default.yaml.

    Config keys used (under 'swin'):
        in_channels, out_channels, feature_size, use_checkpoint
    """
    cfg = config["swin"]
    return AttentionUNet3D(
        in_channels    = cfg["in_channels"],
        out_channels   = cfg["out_channels"],
        feature_size   = cfg["feature_size"],
        use_checkpoint = cfg["use_checkpoint"],
    )


# ─────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml, sys
    sys.path.insert(0, ".")

    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing AttentionUNet3D on {device}")

    model = get_swin_generator(config).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f}M")

    dummy = torch.randn(1, 2, 128, 128, 128).to(device)
    out = model(dummy)

    assert out.shape == dummy.shape, f"Shape mismatch: {out.shape}"
    print(f"  Input:  {dummy.shape}")
    print(f"  Output: {out.shape}")
    if device.type == "cuda":
        print(f"  VRAM:   {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("✅ AttentionUNet3D smoke test passed.")
