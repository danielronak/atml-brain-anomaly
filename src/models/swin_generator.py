"""
Swin-UNET Generator
---------------------
MONAI SwinUNETR used as the GAN generator backbone.

Why SwinUNETR for generation:
  - Shifted Window self-attention captures GLOBAL spatial dependencies
  - U-Net skip connections preserve high-frequency spatial boundaries
  - linear compute complexity (not quadratic like vanilla ViT)

Memory notes for Colab:
  - feature_size=24: safe on T4 (16GB) with batch_size=2
  - feature_size=48: needs A100 (40GB) with batch_size=2
  - use_checkpoint=True: gradient checkpointing — essential, keep ON
"""

import torch
from monai.networks.nets import SwinUNETR


def get_swin_generator(config: dict) -> SwinUNETR:
    """
    Build SwinUNETR generator from config.

    Args:
        config: loaded from configs/default.yaml

    Returns:
        SwinUNETR model
    """
    cfg = config["swin"]
    res = tuple(config["data"]["resolution"])

    model = SwinUNETR(
        img_size=res,                                  # (128, 128, 128)
        in_channels=cfg["in_channels"],                # 2 for dual T1+T2
        out_channels=cfg["out_channels"],              # 2 for dual T1+T2
        feature_size=cfg["feature_size"],              # 24 (T4) or 48 (A100)
        use_checkpoint=cfg["use_checkpoint"],          # gradient checkpointing
        spatial_dims=3,
    )

    return model


if __name__ == "__main__":
    import yaml

    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing SwinUNETR Generator on {device}")

    model = get_swin_generator(config).to(device)

    dummy = torch.randn(1, 2, 128, 128, 128).to(device)
    out = model(dummy)

    assert out.shape == dummy.shape, f"Shape mismatch: {out.shape}"
    print(f"  Input shape:  {dummy.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  VRAM used:    {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("✅ SwinUNETR Generator smoke test passed.")
