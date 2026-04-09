import torch
import torch.nn as nn
import torch.optim as optim
import os

# Import your architectures
from src.models.generator import Generator
from src.models.encoder import Encoder

# NOTE: Import your specific dataloader function here!
# from src.data.dataset import get_dataloader 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Phase 2 Encoder Training on {device}...")

    # ==========================================
    # 1. SETUP THE FROZEN GENERATOR
    # ==========================================
    # We must initialize it with the exact same dimensions as Phase 1
    G = Generator(z_dim=512, features=32, out_channels=1).to(device)
    
    # Path to the weights Azure will produce (we will download these later)
    g_weights_path = "outputs/trained_generator.pth" 
    
    if not os.path.exists(g_weights_path):
        print(f"⚠️ Warning: {g_weights_path} not found. Waiting for Azure Phase 1 to finish!")
        return

    # Load weights and freeze the model
    G.load_state_dict(torch.load(g_weights_path, map_location=device))
    G.eval() # Set to evaluation mode
    for param in G.parameters():
        param.requires_grad = False # LOCK THE WEIGHTS
        
    print("✅ Frozen Phase 1 Generator loaded successfully.")

    # ==========================================
    # 2. SETUP THE NEW ENCODER
    # ==========================================
    E = Encoder(in_channels=1, features=32, z_dim=512).to(device)
    E.train()
    
    # Standard Adam optimizer for the Encoder
    optimizer_E = optim.Adam(E.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Mean Squared Error: Compares pixel-by-pixel differences
    criterion = nn.MSELoss()

    # ==========================================
    # 3. LOAD DATA
    # ==========================================
    # Replace this with your actual dataloader variable
    # train_loader = get_dataloader(batch_size=16) 
    
    epochs = 50 # Phase 2 usually trains much faster than Phase 1

    # ==========================================
    # 4. PHASE 2 TRAINING LOOP
    # ==========================================
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, data in enumerate(train_loader):
            real_imgs = data[0].to(device)
            
            # The 3D -> 4D DataLoader Patch
            if real_imgs.dim() == 3:
                real_imgs = real_imgs.unsqueeze(1)
                
            optimizer_E.zero_grad()
            
            # Step A: Encoder maps the real image to a latent vector 'z'
            z_pred = E(real_imgs)
            
            # Step B: Frozen Generator builds a fake image from that 'z'
            reconstructed_imgs = G(z_pred)
            
            # Step C: Calculate how much detail was lost in translation
            loss = criterion(reconstructed_imgs, real_imgs)
            
            loss.backward()
            optimizer_E.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}] | Reconstruction Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_loader)
        print(f"🏁 Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f}")

    # Save the Phase 2 weights!
    os.makedirs("outputs", exist_ok=True)
    torch.save(E.state_dict(), "outputs/trained_encoder.pth")
    print("🎉 Phase 2 Training Complete. Encoder weights saved!")

if __name__ == "__main__":
    main()