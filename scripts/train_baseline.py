"""
Run Baseline f-AnoGAN Training
------------------------------
What this does:
This is the master script for Phase 1. It pulls together your data,
your models, and your WGAN-GP trainer, and actually starts the 
learning process.
"""
import os
import sys
import torch
import argparse

# Ensure python can find the src folder from inside the scripts folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import get_dataloaders
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.training.trainer import WGANTrainer

def main():
    # 1. Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Train the baseline WGAN-GP")
    # Default to 1 epoch for local Jenkins safety checks. Azure will override this!
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    args = parser.parse_args()

    # 2. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting WGAN-GP Baseline Training on: {device}")
    
    # Cloud-safe relative data path (No local C:\ drives!)
    DATA_DIR = "./data/processed"
    
    # 3. Load Data 
    print(f"Loading data from {DATA_DIR}...")
    # Batch size of 16 is safe for local and cloud GPUs
    train_loader, val_loader, _ = get_dataloaders(DATA_DIR, batch_size=16)
    
    # 4. Initialize Models
    print("Initializing models...")
    # Make sure to push models to the correct device
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # 5. Initialize Trainer
    trainer = WGANTrainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        device=device,
        lr=1e-4
    )
    
    # 6. Run Training Loop
    EPOCHS = args.epochs
    print(f"Starting training loop for {EPOCHS} epoch(s)...")
    print("Watch for the batch progress updates!")
    
    for epoch in range(EPOCHS):
        d_loss, g_loss = trainer.train_epoch(epoch)
        print(f"✅ Epoch {epoch} Complete | Avg D_Loss: {d_loss:.4f} | Avg G_Loss: {g_loss:.4f}")
        
    # 7. Save Checkpoints
    # We save these to the gitignored outputs folder
    os.makedirs("outputs", exist_ok=True)
    torch.save(generator.state_dict(), "outputs/trained_generator.pth")
    torch.save(discriminator.state_dict(), "outputs/trained_discriminator.pth")
    print("🎉 Training complete! Checkpoints saved to outputs/ folder.")

if __name__ == "__main__":
    main()