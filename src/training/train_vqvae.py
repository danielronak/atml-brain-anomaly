"""
VQ-VAE Training Loop
---------------------
Trains the VQ-VAE on IXI healthy brain volumes (T1+T2 dual-channel).
All checkpoints saved to Google Drive.

Run via: notebooks/02_train_vqvae.ipynb

Architecture:
  IXI T1+T2 → Encoder → VQ Codebook (256 tokens) → Decoder → Reconstruction
  Loss = L1(recon, original) + commitment + codebook

Training is deterministic — no adversarial game, no mode collapse risk.
"""

import os
import time
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from tqdm import tqdm

from src.data.dataset import get_ixi_dataloaders
from src.models.vqvae import get_vqvae, get_vqvae_loss


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU found — training will be SLOW. Check Colab runtime type.")
    return device


def get_checkpoint_path(config: dict, epoch: int) -> Path:
    ckpt_dir = Path(config["data"]["checkpoint_dir"]) / config["vqvae"]["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir / f"epoch_{epoch:03d}.pth"


def get_latest_checkpoint(config: dict) -> tuple[Path | None, int]:
    """Find the latest epoch checkpoint in Drive."""
    ckpt_dir = Path(config["data"]["checkpoint_dir"]) / config["vqvae"]["name"]
    if not ckpt_dir.exists():
        return None, 0

    checkpoints = sorted(ckpt_dir.glob("epoch_*.pth"))
    if not checkpoints:
        return None, 0

    latest = checkpoints[-1]
    epoch = int(latest.stem.split("_")[1])
    return latest, epoch


_CONFIG_OVERRIDE = None  # Set by notebooks to inject Drive paths


def train(config_override: dict | None = None):
    config = config_override if config_override is not None else load_config()
    device = get_device()

    # ── DATA ──────────────────────────────────────────────────
    print("\n📂 Loading IXI dataloader...")
    train_loader, val_loader = get_ixi_dataloaders(config)

    # ── MODEL ─────────────────────────────────────────────────
    print("\n🧠 Building VQ-VAE...")
    model = get_vqvae(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params / 1e6:.1f}M")

    optimizer = optim.Adam(model.parameters(), lr=config["vqvae"]["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["vqvae"]["epochs"], eta_min=1e-5
    )

    # ── RESUME ────────────────────────────────────────────────
    latest_ckpt, start_epoch = get_latest_checkpoint(config)
    if latest_ckpt:
        print(f"\n🔄 Resuming from checkpoint: {latest_ckpt} (epoch {start_epoch})")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        print("\n🆕 Starting training from scratch.")
        start_epoch = 0

    # ── TRAINING LOOP ─────────────────────────────────────────
    total_epochs = config["vqvae"]["epochs"]
    ckpt_every = config["vqvae"]["checkpoint_every"]

    print(f"\n🚀 Training VQ-VAE: epochs {start_epoch+1}–{total_epochs}")
    print(f"   Batch size: {config['data']['batch_size']}")
    print(f"   Modality:   {config['data']['modality']}")
    print("─" * 60)

    history = {"train_recon": [], "train_quant": [], "val_recon": []}

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        model.train()

        epoch_recon = 0.0
        epoch_quant = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        for batch in loop:
            images = batch["image"].to(device)

            optimizer.zero_grad()
            reconstruction, quantization_loss = model(images)
            losses = get_vqvae_loss(reconstruction, images, quantization_loss)

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_recon += losses["recon"].item()
            epoch_quant += losses["quantization"].item()

            loop.set_postfix(
                recon=f"{losses['recon'].item():.4f}",
                quant=f"{losses['quantization'].item():.4f}",
            )

        scheduler.step()

        avg_recon = epoch_recon / len(train_loader)
        avg_quant = epoch_quant / len(train_loader)
        history["train_recon"].append(avg_recon)
        history["train_quant"].append(avg_quant)

        # ── VALIDATION ────────────────────────────────────────
        model.eval()
        val_recon = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                recon, q_loss = model(images)
                losses = get_vqvae_loss(recon, images, q_loss)
                val_recon += losses["recon"].item()

        avg_val_recon = val_recon / len(val_loader)
        history["val_recon"].append(avg_val_recon)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:3d}/{total_epochs} | "
              f"Train Recon: {avg_recon:.4f} | "
              f"Quant: {avg_quant:.4f} | "
              f"Val Recon: {avg_val_recon:.4f} | "
              f"Time: {elapsed:.0f}s")

        # ── CHECKPOINT ────────────────────────────────────────
        if (epoch + 1) % ckpt_every == 0 or (epoch + 1) == total_epochs:
            ckpt_path = get_checkpoint_path(config, epoch + 1)
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
            }, ckpt_path)
            print(f"  💾 Checkpoint saved: {ckpt_path}")

    # Save final model
    final_path = Path(config["data"]["checkpoint_dir"]) / config["vqvae"]["name"] / "final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Training complete. Final model: {final_path}")

    # Save loss history
    import json
    hist_path = Path(config["data"]["results_dir"]) / "vqvae_training_history.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"   Loss history: {hist_path}")

    return model, history


if __name__ == "__main__":
    train()
