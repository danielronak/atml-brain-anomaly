import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.dataset import get_brats_dataloader
from src.models.baseline import Generator3D, Discriminator3D
from src.models.encoder import Encoder3D

def train_encoder():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    batch_size = 2
    latent_dim = 128
    lr = 0.0002
    epochs = 20
    kappa = 1.0 
    data_path = "data/raw"

    dataloader = get_brats_dataloader(data_path, batch_size=batch_size)

    generator = Generator3D(latent_dim=latent_dim).to(device)
    discriminator = Discriminator3D().to(device)
    encoder = Encoder3D(latent_dim=latent_dim).to(device)

    try:
        generator.load_state_dict(torch.load("checkpoints/generator_baseline_final.pth", map_location=device))
        discriminator.load_state_dict(torch.load("checkpoints/discriminator_baseline_final.pth", map_location=device))
    except FileNotFoundError:
        print("ERROR: Weights not found. Finish train_baseline.py first!")
        return

    generator.eval()
    discriminator.eval()
    for param in generator.parameters(): param.requires_grad = False
    for param in discriminator.parameters(): param.requires_grad = False

    criterion = nn.MSELoss()
    opt_e = optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(loop):
            real_images = batch["image"].to(device)
            
            opt_e.zero_grad()
            z_guess = encoder(real_images)
            reconstructed_images = generator(z_guess)
            
            _, real_features = discriminator(real_images)
            _, fake_features = discriminator(reconstructed_images)
            
            loss_residual = criterion(reconstructed_images, real_images)
            loss_feature = criterion(fake_features, real_features)
            loss_e = loss_residual + (kappa * loss_feature)
            
            loss_e.backward()
            opt_e.step()
            loop.set_postfix(Loss_E=loss_e.item())
            
    torch.save(encoder.state_dict(), "checkpoints/encoder_baseline_final.pth")

if __name__ == "__main__":
    train_encoder()