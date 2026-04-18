import torch
from src.data.dataset import get_brats_dataloader

def test():
    # Point this to where your raw data is sitting
    data_path = "data/raw" 
    
    print("Initializing DataLoader...")
    loader = get_brats_dataloader(data_dir=data_path, batch_size=2)
    
    print("Fetching one batch of 3D MRIs...")
    # Grab the first batch
    batch = next(iter(loader))
    images = batch["image"]
    
    print("Success!")
    print(f"Batch Tensor Shape: {images.shape}")
    print(f"Batch Tensor Type: {images.dtype}")
    print("Expected Shape: [Batch, Channel, Depth, Height, Width] -> [2, 1, 64, 64, 64]")

if __name__ == "__main__":
    test()