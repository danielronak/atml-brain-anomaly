"""
f-AnoGAN Stage 1 Trainer (WGAN-GP)
----------------------------------
What this does:
This script contains the training loop for Stage 1. It orchestrates the 
data, the models, and the loss functions. 

Key WGAN-GP Mechanics:
1. "n_critic = 5": The discriminator trains 5 times for every 1 generator update.
2. "lambda_gp = 10": The weight applied to the gradient penalty to ensure stability.
"""

import torch
from torch.optim import Adam
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.losses.gan_losses import get_generator_loss, get_discriminator_loss, compute_gradient_penalty

class WGANTrainer:
    def __init__(self, generator, discriminator, train_loader, device, lr=1e-4):
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.train_loader = train_loader
        self.device = device
        
        # WGAN-GP requires specific Adam optimizer settings (beta1=0.5)
        self.opt_G = Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.9))
        self.opt_D = Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.9))
        
        self.n_critic = 5
        self.lambda_gp = 10

    def train_epoch(self, epoch_idx):
        self.G.train()
        self.D.train()
        
        total_g_loss = 0
        total_d_loss = 0
        
        for batch_idx, real_imgs in enumerate(self.train_loader):
            real_imgs = real_imgs.to(self.device)
            batch_size = real_imgs.size(0)
            
            # ==========================================
            # 1. Train Discriminator (Critic) - 5 steps
            # ==========================================
            for _ in range(self.n_critic):
                self.opt_D.zero_grad()
                
                # Generate fake images from random noise
                z = torch.randn(batch_size, 512).to(self.device)
                # Reshape noise to (batch, 512, 1, 1) for the ConvTranspose layers
                z = z.view(batch_size, 512, 1, 1) 
                
                fake_imgs = self.G(z)
                
                # Get discriminator predictions
                real_validity = self.D(real_imgs)
                fake_validity = self.D(fake_imgs.detach()) # Detach so we don't train G here
                
                # Calculate WGAN loss + Gradient Penalty
                d_loss_base = get_discriminator_loss(real_validity, fake_validity)
                gp = compute_gradient_penalty(self.D, real_imgs, fake_imgs.detach(), self.device)
                
                d_loss = d_loss_base + (self.lambda_gp * gp)
                
                d_loss.backward()
                self.opt_D.step()
                
            total_d_loss += d_loss.item()
                
            # ==========================================
            # 2. Train Generator - 1 step
            # ==========================================
            self.opt_G.zero_grad()
            
            # Generate a new batch of fake images
            z = torch.randn(batch_size, 512).to(self.device)
            z = z.view(batch_size, 512, 1, 1)
            fake_imgs = self.G(z)
            
            # Trick the discriminator
            fake_validity = self.D(fake_imgs)
            g_loss = get_generator_loss(fake_validity)
            
            g_loss.backward()
            self.opt_G.step()
            
            total_g_loss += g_loss.item()
            
            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch_idx}] Batch [{batch_idx}/{len(self.train_loader)}] "
                      f"| D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
                
        # Return average losses for the epoch
        return total_d_loss / len(self.train_loader), total_g_loss / len(self.train_loader)

# ── QUICK SYNTAX TEST ────────────────────────────────────────────────
if __name__ == "__main__":
    print("Syntax check passed for WGANTrainer!")