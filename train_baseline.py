import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.dataset import get_brats_dataloader
from src.models.baseline import Generator3D, Discriminator3D

def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")

    batch_size = 2
    latent_dim = 128
    lr = 0.0002
    epochs = 50 
    data_path = "data/raw"

    os.makedirs("checkpoints", exist_ok=True)

    print("Loading Dataloader...")
    dataloader = get_brats_dataloader(data_path, batch_size=batch_size)

    print("Initializing Models...")
    generator = Generator3D(latent_dim=latent_dim).to(device)
    discriminator = Discriminator3D().to(device)

    criterion = nn.BCELoss()
    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    print(f"Starting Training Loop for {epochs} Epochs...")
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(loop):
            real_images = batch["image"].to(device)
            current_batch_size = real_images.size(0)
            
            real_labels = torch.ones(current_batch_size, 1).to(device)
            fake_labels = torch.zeros(current_batch_size, 1).to(device)

            opt_d.zero_grad()
            validity_real, _ = discriminator(real_images)
            loss_d_real = criterion(validity_real, real_labels)
            
            noise = torch.randn(current_batch_size, latent_dim).to(device)
            fake_images = generator(noise)
            validity_fake, _ = discriminator(fake_images.detach()) 
            loss_d_fake = criterion(validity_fake, fake_labels)
            
            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            opt_d.step()

            opt_g.zero_grad()
            validity_fake_for_g, _ = discriminator(fake_images)
            loss_g = criterion(validity_fake_for_g, real_labels)
            
            loss_g.backward()
            opt_g.step()

            loop.set_postfix(Loss_D=loss_d.item(), Loss_G=loss_g.item())
            
        # Save every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"\nSaving checkpoint for Epoch {epoch + 1}...")
            torch.save(generator.state_dict(), f"checkpoints/generator_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"checkpoints/discriminator_epoch_{epoch+1}.pth")
            
    print("\nSaving Final Checkpoints...")
    torch.save(generator.state_dict(), "checkpoints/generator_baseline_final.pth")
    torch.save(discriminator.state_dict(), "checkpoints/discriminator_baseline_final.pth")
    print("Overnight training complete!")

if __name__ == "__main__":
    train()