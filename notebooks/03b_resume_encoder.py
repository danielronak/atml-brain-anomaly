# ══════════════════════════════════════════════════════════════════
# ENCODER RESUME — paste this into a new cell in Notebook 03
# Run this INSTEAD of Cell 9 when resuming after a session crash.
#
# Pre-requisites (run first):
#   Cell 1 (GPU check)  → Cell 2 (Setup + git pull) → Cell 3 (Config)
#   Then paste THIS cell. Skip Cells 4, 5, 6, 7, 8, 9.
# ══════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.optim as optim
import time
from pathlib import Path
from src.data.dataset import get_ixi_dataloaders
from src.models.swin_generator import get_swin_generator

device = torch.device("cuda")

# ── 1. Reload the trained generator from Drive ────────────────────
print("Loading Swin GAN generator from Drive...")
G_trained = get_swin_generator(config).to(device)
gen_ckpt = Path(config["data"]["checkpoint_dir"]) / "swin_gan" / "generator_final.pth"
raw = torch.load(gen_ckpt, map_location=device)
# Handle both raw state_dict and wrapped checkpoint dict
G_trained.load_state_dict(raw["model"] if isinstance(raw, dict) and "model" in raw else raw)
G_trained.eval()
print(f"✅ Generator loaded from {gen_ckpt.name}")

# ── 2. Rebuild & reload encoder from epoch 10 checkpoint ─────────
in_ch     = config["swin"]["in_channels"]
latent_dim = config["cnn"]["latent_dim"]

class Encoder3D(nn.Module):
    def __init__(self, in_channels=2, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, 4, 2, 1),  nn.InstanceNorm3d(128), nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, 4, 2, 1), nn.InstanceNorm3d(256), nn.LeakyReLU(0.2),
            nn.Conv3d(256, 512, 4, 2, 1), nn.InstanceNorm3d(512), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(512, latent_dim),
        )
    def forward(self, x): return self.net(x)

encoder = Encoder3D(in_channels=in_ch, latent_dim=latent_dim).to(device)

enc_ckpt_dir = Path(config["data"]["checkpoint_dir"]) / "swin_gan"
resume_ckpt  = enc_ckpt_dir / "encoder_epoch_10.pth"

if resume_ckpt.exists():
    encoder.load_state_dict(torch.load(resume_ckpt, map_location=device))
    start_epoch = 10
    print(f"✅ Encoder loaded from encoder_epoch_10.pth — resuming from epoch {start_epoch+1}")
else:
    start_epoch = 0
    print("⚠️  encoder_epoch_10.pth not found — starting encoder from scratch")

# ── 3. Resume training from epoch start_epoch+1 ──────────────────
cfg_enc    = config["encoder"]
enc_epochs = cfg_enc["epochs"]  # 20
train_loader, _ = get_ixi_dataloaders(config)
opt_E = optim.Adam(encoder.parameters(), lr=cfg_enc["lr"])

print(f"\nResuming encoder: epochs {start_epoch+1}–{enc_epochs}")
enc_history = []

for epoch in range(start_epoch, enc_epochs):
    encoder.train()
    t0 = time.time()
    epoch_loss = 0.0

    for batch in train_loader:
        real = batch["image"].to(device)
        opt_E.zero_grad()

        # izi_f loss: reconstruction consistency in image + feature space
        recon   = G_trained(real)                       # G's pseudo-healthy recon
        z_hat   = encoder(real)                         # E(real)
        z_recon = encoder(recon.detach())               # E(G(real))

        loss_img  = nn.functional.mse_loss(recon.detach(), real)
        loss_feat = nn.functional.mse_loss(z_recon, z_hat.detach())
        loss = loss_img + cfg_enc["kappa"] * loss_feat

        loss.backward()
        opt_E.step()
        epoch_loss += loss.item()

    avg = epoch_loss / len(train_loader)
    enc_history.append(avg)
    print(f"  Encoder {epoch+1:2d}/{enc_epochs} | Loss: {avg:.4f} | {time.time()-t0:.0f}s")

    if (epoch + 1) % 5 == 0 or (epoch + 1) == enc_epochs:
        save_path = enc_ckpt_dir / f"encoder_epoch_{epoch+1:02d}.pth"
        torch.save(encoder.state_dict(), save_path)
        print(f"  💾 Saved: {save_path.name}")

# Save final
torch.save(encoder.state_dict(), enc_ckpt_dir / "encoder_final.pth")
print(f"\n✅ Encoder done: {enc_ckpt_dir}/encoder_final.pth")

# Plot encoder loss
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.plot(range(start_epoch+1, enc_epochs+1), enc_history, color="darkorange",
         linewidth=2, marker="o", markersize=4)
plt.title("Encoder (izi_f) Training Loss", fontsize=12)
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.grid(True, alpha=0.4)
fig_path = Path(config["data"]["results_dir"]) / "swin_encoder_loss.png"
Path(config["data"]["results_dir"]).mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {fig_path}")
