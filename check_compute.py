import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch

def check_hardware():
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check for NVIDIA CUDA (RTX 3050)
    if torch.cuda.is_available():
        print("CUDA is available! Using NVIDIA GPU.")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
        
    # Check for Apple Silicon (M5 Mac)
    elif torch.backends.mps.is_available():
        print("MPS is available! Using Apple Silicon GPU.")
        device = torch.device("mps")
        
    else:
        print("Neither CUDA nor MPS detected. Using CPU.")
        device = torch.device("cpu")
        
    # Run a simple tensor operation to confirm it works
    try:
        x = torch.rand(5, 5).to(device)
        y = torch.rand(5, 5).to(device)
        z = x @ y
        print(f"Success! Performed matrix multiplication on {device}")
    except Exception as e:
        print(f"Error during tensor operation: {e}")

if __name__ == "__main__":
    check_hardware()