"""
Run Baseline f-AnoGAN Training
------------------------------
What this does:
This is the master script for Phase 2. It pulls together your data,
your models, and your WGAN-GP trainer, and actually starts the 
learning process.
"""
import os
import sys
import torch

# Ensure python can find the src folder from inside the scripts folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import get_dataloaders
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.training.trainer import WGANTrainer

def main():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Phase 2 Local Test Training on: {device}")
    
    # Using your specific data path from earlier tests
    DATA_DIR = r"C:\Users\Ronak Daniel\Documents\atml-brain-anomaly\data\processed"
    
    # 2. Load Data 
    print("Loading data...")
    # We use a batch size of 16 which is safe for an RTX 3060
    train_loader, val_loader, _ = get_dataloaders(DATA_DIR, batch_size=16)
    
    # 3. Initialize Models
    print("Initializing models...")
    generator = Generator()
    discriminator = Discriminator()
    
    # 4. Initialize Trainer
    trainer = WGANTrainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        device=device,
        lr=1e-4
    )
    
    # 5. Run a tiny test loop
    EPOCHS = 1
    print(f"Starting training loop for {EPOCHS} epoch(s)...")
    print("This may take a few minutes. Watch for the batch progress updates!")
    
    for epoch in range(EPOCHS):
        d_loss, g_loss = trainer.train_epoch(epoch)
        print(f"✅ Epoch {epoch} Complete | Avg D_Loss: {d_loss:.4f} | Avg G_Loss: {g_loss:.4f}")
        
    # 6. Save a test checkpoint
    # We save these to the gitignored outputs folder
    os.makedirs("outputs", exist_ok=True)
    torch.save(generator.state_dict(), "outputs/test_generator.pth")
    torch.save(discriminator.state_dict(), "outputs/test_discriminator.pth")
    print("Test checkpoints saved to outputs/ folder.")
    print("🎉 Local training pipeline test successful!")

if __name__ == "__main__":
    main()