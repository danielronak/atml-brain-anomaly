"""
Run Baseline f-AnoGAN Training
------------------------------
"""
import os
import torch
import argparse

from src.data.dataset import get_dataloaders
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.training.trainer import WGANTrainer

def main():
    parser = argparse.ArgumentParser(description="Train the baseline WGAN-GP")
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--data_dir', type=str, default="./data/processed", help='Data directory path')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting WGAN-GP Baseline Training on: {device}")
    
    DATA_DIR = args.data_dir
    
    print(f"Loading data from {DATA_DIR}...")
    
    # ---------------------------------------------------------
    # THE SPEED FIX: Massively increased batch size, added 4 CPU 
    # workers to fetch data, and pinned memory for fast GPU transfer.
    # ---------------------------------------------------------
    train_loader, val_loader, _ = get_dataloaders(DATA_DIR, batch_size=128, num_workers=4)
    
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