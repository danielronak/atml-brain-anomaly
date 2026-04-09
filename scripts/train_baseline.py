"""
Run Baseline f-AnoGAN Training
------------------------------
This is the master script for Phase 1. It pulls together your data,
your models, and your WGAN-GP trainer.
"""
import os
import sys
import torch
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import get_dataloaders
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.training.trainer import WGANTrainer

def main():
    parser = argparse.ArgumentParser(description="Train the baseline WGAN-GP")
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    # Default exactly to your Windows path for the Jenkins local test
    parser.add_argument('--data_dir', type=str, default=r"C:\Users\Ronak Daniel\Documents\atml-brain-anomaly\data\processed", help='Data directory path')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting WGAN-GP Baseline Training on: {device}")
    
    # Use whatever path the argument parser gives us
    DATA_DIR = args.data_dir
    
    print(f"Loading data from {DATA_DIR}...")
    train_loader, val_loader, _ = get_dataloaders(DATA_DIR, batch_size=16)
    
    print("Initializing models...")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    trainer = WGANTrainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        device=device,
        lr=1e-4
    )
    
    EPOCHS = args.epochs
    print(f"Starting training loop for {EPOCHS} epoch(s)...")
    
    for epoch in range(EPOCHS):
        d_loss, g_loss = trainer.train_epoch(epoch)
        print(f"✅ Epoch {epoch} Complete | Avg D_Loss: {d_loss:.4f} | Avg G_Loss: {g_loss:.4f}")
        
    os.makedirs("outputs", exist_ok=True)
    torch.save(generator.state_dict(), "outputs/trained_generator.pth")
    torch.save(discriminator.state_dict(), "outputs/trained_discriminator.pth")
    print("🎉 Training complete! Checkpoints saved.")

if __name__ == "__main__":
    main()