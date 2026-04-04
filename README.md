# Hybrid Swin-UNET GAN for Unsupervised Brain Tumor Detection

A novel unsupervised anomaly detection system for brain MRI using a 
Hybrid Swin-UNET GAN architecture with perceptual loss.

## Overview
This project builds on f-AnoGAN (Schlegl et al., 2019) by replacing 
the CNN generator with a Swin Transformer-based U-Net, enabling 
anatomically coherent reconstruction and sharper anomaly maps.

## Architecture
- **Generator**: Swin-UNET (Transformer backbone + U-Net skip connections)
- **Discriminator**: CNN PatchGAN
- **Loss**: Adversarial + L1 Reconstruction + Perceptual (VGG16)
- **Dataset**: BraTS 2021 Brain MRI

## Authors
- Ronak Daniel
- Neel Jaiswal

## Status
🔧 Phase 1 — Environment & Repository Setup
