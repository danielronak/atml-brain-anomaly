"""
WGAN-GP GAN Training Loop (Swin-UNET GAN + CNN Proper)
-------------------------------------------------------
Shared training logic for both Model 2 (CNN Proper) and Model 3 (Swin-UNET GAN).

The generator is passed in — everything else is identical:
  - WGAN-GP discriminator loss (gradient penalty, λ=10)
  - Two Time-scale Update Rule: LR_D = 4 × LR_G
  - L1 reconstruction term added to Generator loss (prevents mode collapse)
  - n_critic = 5 (discriminator steps per generator step)
  - Drive checkpoint with resume support

WGAN-GP now works because:
  - We're on CUDA (Colab) — create_graph=True is supported
  - InstanceNorm3d (not BatchNorm) in discriminator
  - Spectral norm on all discriminator conv layers (additional stability)
"""

import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from src.data.dataset import get_ixi_dataloaders
from src.models.patch_discriminator import PatchDiscriminator3D


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU — training will be very slow.")
    return device


def compute_gradient_penalty(D: nn.Module,
                              real: torch.Tensor,
                              fake: torch.Tensor,
                              device: torch.device,
                              lambda_gp: float = 10.0) -> torch.Tensor:
    """
    WGAN-GP gradient penalty.
    Enforces Lipschitz constraint on the discriminator.
    Works on CUDA (create_graph=True supported). Would fail on MPS.
    """
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, 1, device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    d_interp, _ = D(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(B, -1)
    gp = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def get_checkpoint_path(config: dict, model_name: str, epoch: int) -> Path:
    ckpt_dir = Path(config["data"]["checkpoint_dir"]) / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir / f"epoch_{epoch:03d}.pth"


def get_latest_checkpoint(config: dict, model_name: str) -> tuple:
    ckpt_dir = Path(config["data"]["checkpoint_dir"]) / model_name
    if not ckpt_dir.exists():
        return None, 0
    checkpoints = sorted(ckpt_dir.glob("epoch_*.pth"))
    if not checkpoints:
        return None, 0
    latest = checkpoints[-1]
    epoch = int(latest.stem.split("_")[1])
    return latest, epoch


def train_gan(generator: nn.Module,
              model_name: str,
              config: dict | None = None):
    """
    Full WGAN-GP training run.

    Args:
        generator:   Pre-built generator (SwinUNETR or Generator3D)
        model_name:  String key for checkpoints ('swin_gan' or 'cnn_gan')
        config:      Loaded YAML config dict. Loads from default if None.
    """
    if config is None:
        config = load_config()

    device = get_device()
    cfg = config["gan"]

    # ── DATA ──────────────────────────────────────────────────
    print("\n📂 Loading IXI dataloader...")
    train_loader, val_loader = get_ixi_dataloaders(config)

    # ── MODELS ────────────────────────────────────────────────
    G = generator.to(device)
    D = PatchDiscriminator3D(in_channels=config["swin"]["in_channels"]).to(device)

    total_G = sum(p.numel() for p in G.parameters()) / 1e6
    total_D = sum(p.numel() for p in D.parameters()) / 1e6
    print(f"\n🧠 Generator:     {total_G:.1f}M parameters")
    print(f"   Discriminator: {total_D:.1f}M parameters")

    # TTUR: Discriminator LR = 4× Generator LR
    opt_G = optim.Adam(G.parameters(),
                       lr=cfg["lr_g"],
                       betas=tuple(cfg["adam_betas"]))
    opt_D = optim.Adam(D.parameters(),
                       lr=cfg["lr_d"],
                       betas=tuple(cfg["adam_betas"]))

    # ── RESUME ────────────────────────────────────────────────
    latest_ckpt, start_epoch = get_latest_checkpoint(config, model_name)
    if latest_ckpt:
        print(f"\n🔄 Resuming from {latest_ckpt} (epoch {start_epoch})")
        ckpt = torch.load(latest_ckpt, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
    else:
        print("\n🆕 Starting training from scratch.")
        start_epoch = 0

    # ── TRAINING LOOP ─────────────────────────────────────────
    total_epochs = cfg["epochs"]
    n_critic = cfg["n_critic"]
    lambda_gp = cfg["lambda_gp"]
    lambda_rec = cfg["lambda_rec"]
    ckpt_every = cfg["checkpoint_every"]

    print(f"\n🚀 Training {model_name}: epochs {start_epoch+1}–{total_epochs}")
    print(f"   LR_G={cfg['lr_g']} | LR_D={cfg['lr_d']} | λ_GP={lambda_gp} | λ_rec={lambda_rec}")
    print("─" * 60)

    history = {"d_loss": [], "g_loss": [], "gp": []}

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        G.train(); D.train()

        epoch_d, epoch_g, epoch_gp = 0.0, 0.0, 0.0
        g_steps = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        for batch_idx, batch in enumerate(loop):
            real = batch["image"].to(device)
            B = real.size(0)

            # ── DISCRIMINATOR STEP ────────────────────────────
            opt_D.zero_grad()
            fake = G(real).detach()

            real_scores, _ = D(real)
            fake_scores, _ = D(fake)
            gp = compute_gradient_penalty(D, real, fake, device, lambda_gp)

            # WGAN loss: maximise E[D(real)] - E[D(fake)]
            loss_D = fake_scores.mean() - real_scores.mean() + gp
            loss_D.backward()
            opt_D.step()

            epoch_d += loss_D.item()
            epoch_gp += gp.item()

            # ── GENERATOR STEP (every n_critic batches) ───────
            if batch_idx % n_critic == 0:
                opt_G.zero_grad()
                fake_for_G = G(real)
                fake_scores_G, _ = D(fake_for_G)

                # WGAN: maximise E[D(fake)]
                adv_loss = -fake_scores_G.mean()
                # L1 reconstruction: keeps outputs close to input
                rec_loss = torch.nn.functional.l1_loss(fake_for_G, real)
                loss_G = adv_loss + lambda_rec * rec_loss

                loss_G.backward()
                opt_G.step()

                epoch_g += loss_G.item()
                g_steps += 1

                loop.set_postfix(
                    D=f"{loss_D.item():.3f}",
                    G=f"{loss_G.item():.3f}",
                    GP=f"{gp.item():.3f}",
                )

        avg_d = epoch_d / len(train_loader)
        avg_g = epoch_g / max(g_steps, 1)
        avg_gp = epoch_gp / len(train_loader)

        history["d_loss"].append(avg_d)
        history["g_loss"].append(avg_g)
        history["gp"].append(avg_gp)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:3d}/{total_epochs} | "
              f"D: {avg_d:.4f} | G: {avg_g:.4f} | GP: {avg_gp:.4f} | "
              f"Time: {elapsed:.0f}s")

        # ── STABILITY CHECK ────────────────────────────────────
        if abs(avg_d) < 0.001 and epoch > 5:
            print("  ⚠️  WARNING: Discriminator loss near zero — possible collapse!")

        # ── CHECKPOINT ────────────────────────────────────────
        if (epoch + 1) % ckpt_every == 0 or (epoch + 1) == total_epochs:
            ckpt_path = get_checkpoint_path(config, model_name, epoch + 1)
            torch.save({
                "epoch": epoch + 1,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
                "history": history,
            }, ckpt_path)
            print(f"  💾 Checkpoint: {ckpt_path}")

    # Final save
    final_dir = Path(config["data"]["checkpoint_dir"]) / model_name
    torch.save(G.state_dict(), final_dir / "generator_final.pth")
    torch.save(D.state_dict(), final_dir / "discriminator_final.pth")

    # Save history
    results_dir = Path(config["data"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{model_name}_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ {model_name} training complete.")
    return G, D, history


# ──────────────────────────────────────────────────────────────
# Entrypoints for each model
# ──────────────────────────────────────────────────────────────

def train_swin_gan():
    """Train Swin-UNET GAN (Model 3)."""
    from src.models.swin_generator import get_swin_generator
    config = load_config()
    G = get_swin_generator(config)
    return train_gan(G, model_name=config["swin"]["name"], config=config)


def train_cnn_proper():
    """Train CNN Proper GAN with WGAN-GP (Model 2)."""
    from src.models.baseline import Generator3D
    config = load_config()

    # CNN generator takes flat latent vector — not a volume-to-volume map
    # For GAN training in the proper setup, we adapt to use the volume directly
    # by wrapping in an autoencoder-style generator
    G = Generator3DProper(
        latent_dim=config["cnn"]["latent_dim"],
        in_channels=config["cnn"]["in_channels"],
    )
    return train_gan(G, model_name=config["cnn"]["name"], config=config)


class Generator3DProper(nn.Module):
    """
    CNN generator adapted for volume-to-volume translation.
    Used for the 'CNN Proper' baseline (Model 2).
    """
    def __init__(self, latent_dim: int = 128, in_channels: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, 4, 2, 1), nn.BatchNorm3d(128), nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, 4, 2, 1), nn.BatchNorm3d(256), nn.LeakyReLU(0.2),
            nn.Conv3d(256, 512, 4, 2, 1), nn.BatchNorm3d(512), nn.LeakyReLU(0.2),
            nn.Flatten(), nn.Linear(512 * 8 * 8 * 8, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4 * 4),
        )
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv3d(512, 256, 3, 1, 1), nn.BatchNorm3d(256), nn.ReLU(),
            nn.Upsample(scale_factor=2), nn.Conv3d(256, 128, 3, 1, 1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.Upsample(scale_factor=2), nn.Conv3d(128, 64, 3, 1, 1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.Upsample(scale_factor=2), nn.Conv3d(64, in_channels, 3, 1, 1), nn.Tanh(),
        )

    def forward(self, x):
        z = self.encoder(x)
        d = self.decoder(z).view(-1, 512, 4, 4, 4)
        return self.upsample(d)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "swin":
        train_swin_gan()
    elif len(sys.argv) > 1 and sys.argv[1] == "cnn":
        train_cnn_proper()
    else:
        print("Usage: python train_gan.py [swin|cnn]")
