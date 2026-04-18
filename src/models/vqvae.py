"""
VQ-VAE — Vector Quantised Variational Autoencoder
---------------------------------------------------
Primary generative model for unsupervised anomaly detection.

Architecture: MONAI GenerativeModels VQ-VAE
  - 3D encoder: volume → discrete latent codes (codebook)
  - Codebook: 256 learned "anatomy tokens" of dim 32
  - 3D decoder: codes → reconstructed volume

Why VQ-VAE for anomaly detection:
  - The codebook only contains healthy anatomy patterns (learned from IXI)
  - At test time, a tumor CANNOT be represented by healthy codebook entries
  - The decoder replaces it with the closest healthy anatomy → residual = anomaly
  - Stable training (no adversarial game, no mode collapse)

Dual-modality (T1+T2):
  - in_channels=2, out_channels=2
  - Model learns inter-modal correlations between T1 and T2
  - Tumors disrupt BOTH within-modality statistics AND inter-modal correlations
  - Richer anomaly signal than single modality
"""

import torch
import torch.nn as nn
from monai.networks.nets.vqvae import VQVAE


def get_vqvae(config: dict) -> VQVAE:
    """
    Build VQ-VAE from config.

    Config keys used:
        vqvae.in_channels, vqvae.out_channels
        vqvae.num_channels, vqvae.num_res_layers
        vqvae.num_embeddings, vqvae.embedding_dim
        vqvae.commitment_cost
    """
    cfg = config["vqvae"]

    model = VQVAE(
        spatial_dims=3,
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        num_channels=cfg["num_channels"],          # [64, 128, 256]
        num_res_channels=cfg["num_channels"],       # match encoder channels
        num_res_layers=cfg["num_res_layers"],       # 2
        num_embeddings=cfg["num_embeddings"],       # 256
        embedding_dim=cfg["embedding_dim"],         # 32
        commitment_cost=cfg["commitment_cost"],     # 0.25
        decay=0.5,                                  # EMA decay for codebook updates
        epsilon=1e-5,
    )

    return model


def get_vqvae_loss(reconstruction: torch.Tensor,
                   target: torch.Tensor,
                   quantization_loss: torch.Tensor) -> dict:
    """
    VQ-VAE total loss:
        L = L1_recon + quantization_loss

    quantization_loss already contains:
        - codebook loss: ||sg[z_e] - e||^2   (move codebook toward encoder output)
        - commitment loss: beta * ||z_e - sg[e]||^2  (encourage commitment)

    Args:
        reconstruction: G(E(x)) — model output
        target: x — input volume
        quantization_loss: from model forward pass

    Returns:
        dict with individual loss components and total
    """
    recon_loss = nn.functional.l1_loss(reconstruction, target)
    total = recon_loss + quantization_loss

    return {
        "total": total,
        "recon": recon_loss,
        "quantization": quantization_loss,
    }


if __name__ == "__main__":
    # Quick smoke test — run this in Colab before training
    import yaml

    with open("configs/default.yaml") as f:
        import yaml
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing VQ-VAE on {device}")

    model = get_vqvae(config).to(device)

    # Simulate a dual-channel 128³ batch
    dummy = torch.randn(1, 2, 128, 128, 128).to(device)
    recon, quant_loss = model(dummy)

    assert recon.shape == dummy.shape, f"Shape mismatch: {recon.shape} vs {dummy.shape}"

    losses = get_vqvae_loss(recon, dummy, quant_loss)
    print(f"  Recon shape:      {recon.shape}")
    print(f"  Recon loss:       {losses['recon'].item():.4f}")
    print(f"  Quantization loss:{losses['quantization'].item():.4f}")
    print(f"  Total loss:       {losses['total'].item():.4f}")
    print(f"  VRAM used:        {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("✅ VQ-VAE smoke test passed.")
