import matplotlib.pyplot as plt
import numpy as np

def plot_thesis_curves():
    # ==========================================
    # 1. PHASE 1: GAN LOSSES (Full 50 Epochs)
    # ==========================================
    epochs_gan = np.arange(1, 51) 
    
    # Extracted exactly from your terminal output
    loss_d = [
        0.00375, 0.00126, 2.73e-5, 0.00444, 0.989, 1.8, 0.167, 0.262, 0.0032, 0.196, 
        0.88, 0.0165, 0.0716, 0.0137, 0.404, 0.00215, 0.00452, 0.568, 0.00481, 6.62e-6, 
        0.0237, 0.000106, 0.0296, 0.0155, 0.0625, 0.217, 0.00554, 0.00441, 0.0181, 0.00151, 
        0.00195, 0.00731, 0.0261, 0.0016, 0.0139, 1.97, 0.00274, 0.0135, 0.00378, 0.00341, 
        0.00625, 0.0, 1.6, 0.0121, 0.000722, 0.016, 0.0325, 0.154, 0.0, 1.01e-6
    ]
    
    loss_g = [
        5.36, 6.07, 15.1, 4.46, 4.22, 1.14, 6.43, 2.26, 7.44, 7.92, 
        9.59, 5.21, 4.75, 3.38, 1.45, 5.89, 6.09, 0.394, 4.98, 19.6, 
        8.74, 8.78, 4.52, 5.05, 6.67, 11.4, 5.49, 5.64, 12.8, 16.0, 
        7.12, 6.08, 7.92, 20.8, 4.18, 0.000821, 7.78, 4.57, 8.62, 5.41, 
        44.1, 42.4, 31.0, 5.0, 6.71, 4.24, 6.45, 4.77, 40.9, 37.3
    ]
    
    plt.figure(figsize=(12, 6))
    
    # Plotting Generator and Discriminator
    plt.plot(epochs_gan, loss_d, label='Discriminator Loss (Loss_D)', color='#1f77b4', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs_gan, loss_g, label='Generator Loss (Loss_G)', color='#d62728', linewidth=2, marker='x', markersize=4)
    
    # Highlight the Mode Collapse zone
    plt.axvspan(40, 50, color='red', alpha=0.1, label='Mode Collapse Zone')
    
    plt.title('Phase 1: GAN Training Loss (Standard 3D CNN Baseline)', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    
    # Save the GAN plot
    plt.savefig('gan_loss_curve.png', dpi=300)
    print("Saved Phase 1 plot as 'gan_loss_curve.png'")
    

    # ==========================================
    # 2. PHASE 2: ENCODER LOSSES (MSE)
    # ==========================================
    epochs_enc = np.arange(1, 21)
    loss_e = [
        0.167, 0.121, 0.0901, 0.468, 0.242, 
        0.131, 0.127, 0.0579, 0.305, 0.167, 
        0.170, 0.139, 0.136, 0.262, 0.140, 
        0.0818, 0.144, 0.0712, 0.148, 0.103
    ]
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_enc, loss_e, label='Encoder Mapping Loss (MSE)', color='green', linewidth=2, marker='s')
    
    # Trendline
    z = np.polyfit(epochs_enc, loss_e, 1)
    p = np.poly1d(z)
    plt.plot(epochs_enc, p(epochs_enc), "k--", label='Overall Learning Trend', alpha=0.6)

    plt.title('Phase 2: Encoder Mapping Loss (izi_f Architecture)', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the Encoder plot
    plt.savefig('encoder_loss_curve.png', dpi=300)
    print("Saved Phase 2 plot as 'encoder_loss_curve.png'")
    

if __name__ == "__main__":
    plot_thesis_curves()