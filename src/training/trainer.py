import torch
import torch.nn as nn
import torch.optim as optim

class WGANTrainer:
    def __init__(self, generator, discriminator, train_loader, device, lr=0.0002, z_dim=512):
        self.device = device
        self.z_dim = z_dim
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.dataloader = train_loader
        
        # WGAN-GP requires specific beta values for Adam
        self.opt_G = optim.Adam(self.G.parameters(), lr=lr, betas=(0.0, 0.9))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr, betas=(0.0, 0.9))
        
        self.lambda_gp = 10.0 # Standard Gradient Penalty weight
        self.n_critic = 5 # Train Discriminator 5 times for every 1 Generator step

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty for WGAN-GP"""
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.D(interpolates)
        
        fake = torch.ones(real_samples.size(0), 1, device=self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty * self.lambda_gp

    def train_epoch(self, epoch_idx):
        self.G.train()
        self.D.train()
        
        total_g_loss = 0
        total_d_loss = 0
        
        for batch_idx, data in enumerate(self.dataloader):
            real_imgs = data[0].to(self.device)
            batch_size = real_imgs.size(0)
            
            # ==========================================
            # 1. TRAIN CRITIC (DISCRIMINATOR)
            # ==========================================
            self.opt_D.zero_grad()
            
            z = torch.randn(batch_size, self.z_dim, 1, 1, device=self.device)
            fake_imgs = self.G(z)
            
            # WGAN Math: Maximize distance between Real and Fake
            real_validity = self.D(real_imgs)
            fake_validity = self.D(fake_imgs.detach())
            
            gradient_penalty = self.compute_gradient_penalty(real_imgs.data, fake_imgs.data)
            
            # Critic loss: -Mean(Real) + Mean(Fake) + GP
            loss_D = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty
            loss_D.backward()
            self.opt_D.step()
            
            total_d_loss += loss_D.item()

            # ==========================================
            # 2. TRAIN GENERATOR (Every n_critic steps)
            # ==========================================
            if batch_idx % self.n_critic == 0:
                self.opt_G.zero_grad()
                
                # Generator loss: -Mean(Critic(Fake))
                fake_imgs_for_G = self.G(z)
                loss_G = -torch.mean(self.D(fake_imgs_for_G))
                
                loss_G.backward()
                self.opt_G.step()
                total_g_loss += loss_G.item()
            
            if batch_idx % 10 == 0:
                print(f"🚀 Epoch [{epoch_idx}] Batch [{batch_idx}/{len(self.dataloader)}] "
                      f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item() if batch_idx % self.n_critic == 0 else 0:.4f}")
                
        avg_d_loss = total_d_loss / len(self.dataloader)
        avg_g_loss = total_g_loss / (len(self.dataloader) / self.n_critic)
        
        return avg_d_loss, avg_g_loss