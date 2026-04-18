"""
WGAN-GP Loss Functions for f-AnoGAN
-----------------------------------
Based on: Schlegl et al., f-AnoGAN (2019)

What this does:
Instead of a simple binary "real or fake" classification, WGAN calculates 
a "Wasserstein distance" — basically measuring how far the fake distribution 
is from the real distribution. 

The Gradient Penalty (GP) enforces a mathematical rule (Lipschitz continuity) 
that keeps the discriminator's gradients from exploding, ensuring the 
generator always gets useful feedback.
"""

import torch

def get_generator_loss(fake_validity):
    """
    The generator wants the discriminator to output high scores for fake images.
    So we minimize the negative of the discriminator's output.
    """
    return -torch.mean(fake_validity)

def get_discriminator_loss(real_validity, fake_validity):
    """
    The discriminator wants to output high scores for real images and 
    low scores for fake images.
    """
    return torch.mean(fake_validity) - torch.mean(real_validity)

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """
    Calculates the gradient penalty to enforce the Lipschitz constraint.
    This is the "GP" in WGAN-GP.
    """
    # 1. Create random weights to interpolate between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    
    # 2. Create the interpolated images
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # 3. Pass interpolates through the discriminator
    d_interpolates = discriminator(interpolates)
    
    # 4. Calculate gradients of discriminator's output with respect to the interpolates
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # 5. Calculate and return the penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ── QUICK TEST ────────────────────────────────────────────────
if __name__ == "__main__":
    from src.models.discriminator import Discriminator
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing WGAN-GP Losses on {device}...")
    
    # Dummy models and data
    D = Discriminator().to(device)
    real_imgs = torch.randn(4, 1, 128, 128).to(device)
    fake_imgs = torch.randn(4, 1, 128, 128).to(device)
    
    # Simulate discriminator outputs
    real_validity = D(real_imgs)
    fake_validity = D(fake_imgs)
    
    # Calculate losses
    g_loss = get_generator_loss(fake_validity)
    d_loss = get_discriminator_loss(real_validity, fake_validity)
    gp = compute_gradient_penalty(D, real_imgs, fake_imgs, device)
    
    print(f"Generator Loss      : {g_loss.item():.4f}")
    print(f"Discriminator Loss  : {d_loss.item():.4f}")
    print(f"Gradient Penalty    : {gp.item():.4f}")
    print(" ✅ WGAN-GP loss test passed!")