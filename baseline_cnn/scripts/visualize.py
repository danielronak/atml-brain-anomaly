import torch
import matplotlib.pyplot as plt
from src.models.baseline import Generator3D

def visualize_final():
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize the Generator
    latent_dim = 128
    generator = Generator3D(latent_dim=latent_dim).to(device)

    # Load your FINAL 50-epoch weights!
    print("Loading final baseline checkpoints...")
    try:
        generator.load_state_dict(torch.load("checkpoints/generator_baseline_final.pth", map_location=device))
        print("Final weights loaded successfully!")
    except FileNotFoundError:
        print("ERROR: Could not find checkpoints/generator_baseline_final.pth")
        return

    # Set generator to evaluation mode
    generator.eval()

    print("Generating a fake 3D MRI volume...")
    with torch.no_grad():
        noise = torch.randn(1, latent_dim).to(device)
        fake_mri = generator(noise)

    # Extract the 3D tensor and convert it
    volume = fake_mri.squeeze().cpu().numpy()

    # Grab the exact middle slice
    middle_slice = volume[32, :, :]

    print("Plotting the final image...")
    plt.figure(figsize=(6, 6))
    plt.imshow(middle_slice, cmap='gray')
    plt.title("AI-Generated MRI Slice (Final Epoch 50)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    visualize_final()