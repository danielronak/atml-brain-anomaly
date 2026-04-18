"""
VQ-VAE — Custom 3D Implementation (dependency-free)
-----------------------------------------------------
No MONAI VQVAE dependency — fully self-contained, stable across all versions.

Architecture:
  Encoder: 3D CNN (128³ → 16³ latent)
  VectorQuantizer: discrete codebook of 256 healthy-anatomy tokens
  Decoder: 3D transposed CNN (16³ → 128³ reconstruction)

Why VQ-VAE for anomaly detection:
  - Codebook only contains healthy anatomy patterns (learned from IXI)
  - At test time, tumours CANNOT be represented by healthy codebook entries
  - Decoder replaces tumour with closest healthy anatomy → residual = anomaly
  - Stable training: no adversarial game, no mode collapse

Dual-modality (T1+T2):
  - in_channels=2 → model learns inter-modal correlations
  - Tumours violate both within-modality statistics AND T1/T2 co-occurrence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# Vector Quantizer (straight-through estimator)
# ─────────────────────────────────────────────────────────────────
class VectorQuantizer(nn.Module):
    """
    Discrete codebook quantizer with straight-through gradient estimator.

    Args:
        num_embeddings: codebook size K (number of anatomy tokens)
        embedding_dim:  codebook vector dimension D
        commitment_cost: beta — how strongly encoder commits to codebook
    """
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 commitment_cost: float = 0.25):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Uniform init in [-1/K, 1/K]
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings,
                                             1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, D, d, h, w) — encoder output
        Returns:
            z_q:      (B, D, d, h, w) — quantised (straight-through)
            loss:     scalar — codebook + commitment loss
            indices:  (B, d, h, w)  — nearest codebook entry per voxel
        """
        B, D, d, h, w = z.shape

        # Flatten spatial dims: (B*d*h*w, D)
        z_flat = z.permute(0, 2, 3, 4, 1).reshape(-1, D)

        # Squared L2 distances to all codebook entries
        # ||z - e||² = ||z||² - 2·z·eᵀ + ||e||²
        dists = (z_flat.pow(2).sum(1, keepdim=True)
                 - 2.0 * z_flat @ self.embedding.weight.T
                 + self.embedding.weight.pow(2).sum(1))

        # Nearest codebook entry
        indices = dists.argmin(1)                               # (B*d*h*w,)
        z_q = self.embedding(indices)                           # (B*d*h*w, D)
        z_q = z_q.reshape(B, d, h, w, D).permute(0, 4, 1, 2, 3)  # (B,D,d,h,w)

        # Losses
        codebook_loss   = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z_q.detach(), z)
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through: copy gradient through quantisation step
        z_q_st = z + (z_q - z).detach()

        return z_q_st, loss, indices.reshape(B, d, h, w)


# ─────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────
class ResBlock3D(nn.Module):
    """3D residual block with instance normalisation."""
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class Encoder3D(nn.Module):
    """
    Downsampling encoder: (B, in_ch, 128, 128, 128) → (B, emb_dim, 16, 16, 16)
    3 × stride-2 Conv halves spatial dims each time: 128 → 64 → 32 → 16
    """
    def __init__(self, in_channels: int, channels: list[int], embedding_dim: int,
                 num_res_layers: int = 2):
        super().__init__()
        layers = []

        # Input projection
        layers.append(nn.Conv3d(in_channels, channels[0], 3, 1, 1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Downsampling stages
        for i in range(len(channels) - 1):
            layers.append(nn.Conv3d(channels[i], channels[i+1], 4, 2, 1, bias=False))
            layers.append(nn.InstanceNorm3d(channels[i+1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Residual blocks at deepest scale
        for _ in range(num_res_layers):
            layers.append(ResBlock3D(channels[-1]))

        # Project to embedding_dim (codebook vector size)
        layers.append(nn.Conv3d(channels[-1], embedding_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────────
class Decoder3D(nn.Module):
    """
    Upsampling decoder: (B, emb_dim, 16, 16, 16) → (B, out_ch, 128, 128, 128)
    Mirror of encoder.
    """
    def __init__(self, out_channels: int, channels: list[int], embedding_dim: int,
                 num_res_layers: int = 2):
        super().__init__()
        rev_ch = list(reversed(channels))   # [256, 128, 64]
        layers = []

        # Project from embedding_dim
        layers.append(nn.Conv3d(embedding_dim, rev_ch[0], 1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Residual blocks
        for _ in range(num_res_layers):
            layers.append(ResBlock3D(rev_ch[0]))

        # Upsampling stages
        for i in range(len(rev_ch) - 1):
            layers.append(nn.ConvTranspose3d(rev_ch[i], rev_ch[i+1],
                                              kernel_size=4, stride=2, padding=1,
                                              bias=False))
            layers.append(nn.InstanceNorm3d(rev_ch[i+1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Output projection — Tanh keeps output in [-1, 1]
        layers.append(nn.Conv3d(rev_ch[-1], out_channels, 3, 1, 1))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────
# Full VQ-VAE
# ─────────────────────────────────────────────────────────────────
class VQVAE3D(nn.Module):
    """
    Full VQ-VAE for 3D brain MRI anomaly detection.

    forward() returns (reconstruction, quantization_loss) to match
    the same interface used throughout the training and evaluation pipeline.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 channels: list[int], embedding_dim: int,
                 num_embeddings: int, num_res_layers: int,
                 commitment_cost: float):
        super().__init__()
        self.encoder   = Encoder3D(in_channels, channels, embedding_dim, num_res_layers)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder   = Decoder3D(out_channels, channels, embedding_dim, num_res_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            reconstruction: same shape as x
            quant_loss:     scalar quantization + commitment loss
        """
        z     = self.encoder(x)
        z_q, quant_loss, _ = self.quantizer(z)
        recon = self.decoder(z_q)
        return recon, quant_loss


# ─────────────────────────────────────────────────────────────────
# Factory function (used by all training + evaluation code)
# ─────────────────────────────────────────────────────────────────
def get_vqvae(config: dict) -> VQVAE3D:
    """
    Build VQ-VAE from configs/default.yaml.

    Config keys used (under 'vqvae'):
        in_channels, out_channels, num_channels [list],
        num_res_layers, num_embeddings, embedding_dim, commitment_cost
    """
    cfg = config["vqvae"]
    return VQVAE3D(
        in_channels     = cfg["in_channels"],
        out_channels    = cfg["out_channels"],
        channels        = cfg["num_channels"],   # [64, 128, 256]
        embedding_dim   = cfg["embedding_dim"],  # 32
        num_embeddings  = cfg["num_embeddings"], # 256
        num_res_layers  = cfg["num_res_layers"], # 2
        commitment_cost = cfg["commitment_cost"],# 0.25
    )


# ─────────────────────────────────────────────────────────────────
# Loss helper (used in train_vqvae.py)
# ─────────────────────────────────────────────────────────────────
def get_vqvae_loss(reconstruction: torch.Tensor,
                   target: torch.Tensor,
                   quantization_loss: torch.Tensor) -> dict:
    """
    VQ-VAE total loss = L1_recon + quantization_loss

    quantization_loss contains codebook + commitment terms.
    """
    recon_loss = F.l1_loss(reconstruction, target)
    total = recon_loss + quantization_loss
    return {
        "total":        total,
        "recon":        recon_loss,
        "quantization": quantization_loss,
    }


# ─────────────────────────────────────────────────────────────────
# Quick smoke test (run directly: python src/models/vqvae.py)
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml, sys
    sys.path.insert(0, ".")

    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing VQVAE3D on {device}")

    model = get_vqvae(config).to(device)
    dummy = torch.randn(1, 2, 128, 128, 128).to(device)

    recon, quant_loss = model(dummy)
    assert recon.shape == dummy.shape, f"Shape mismatch: {recon.shape}"

    losses = get_vqvae_loss(recon, dummy, quant_loss)
    print(f"  In/Out shape:     {dummy.shape}")
    print(f"  Recon loss:       {losses['recon'].item():.4f}")
    print(f"  Quantization loss:{losses['quantization'].item():.4f}")
    if device.type == "cuda":
        print(f"  VRAM used:        {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print("✅ VQVAE3D smoke test passed.")
