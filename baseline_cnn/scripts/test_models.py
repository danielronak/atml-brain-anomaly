import torch
from src.models.baseline import Generator3D, Discriminator3D

def test_architecture():
    # Force the test onto the CPU just to verify math/shapes quickly
    device = torch.device("cpu")
    batch_size = 2
    latent_dim = 128
    
    print("Initializing Models...")
    generator = Generator3D(latent_dim=latent_dim).to(device)
    discriminator = Discriminator3D().to(device)
    
    print("\nTesting Generator...")
    # Create fake noise
    noise = torch.randn(batch_size, latent_dim).to(device)
    fake_mri = generator(noise)
    print(f"Input Noise Shape: {noise.shape}")
    print(f"Generated MRI Shape: {fake_mri.shape} -> Expected: [2, 1, 64, 64, 64]")
    
    print("\nTesting Discriminator...")
    # Pass the fake MRI into the discriminator
    validity, features = discriminator(fake_mri)
    print(f"Validity Output Shape: {validity.shape} -> Expected: [2, 1]")
    print(f"Feature Map Shape: {features.shape} -> Expected: [2, 512, 4, 4, 4]")
    
    print("\nSuccess! Forward pass completed without tensor dimension errors.")

if __name__ == "__main__":
    test_architecture()